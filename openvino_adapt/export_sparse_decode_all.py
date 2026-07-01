from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .export_hot_expert_pack import expert_ids_from_plan
from .sparse_decode_wrappers import (
    AddMoEResidual,
    DecodeDenseLayer,
    DecodeMoEAttentionGate,
    DecodeMoEHotGatherFusedLayer,
    FinalNormArgmax,
    FinalNormHead,
    FinalNormTopK,
)
from .sparse_moe import ExpertOnly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all split sparse decode-one graphs.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--output-dir", default="openvino_models/sparse_decode_past677")
    parser.add_argument("--past-len", type=int, default=677)
    parser.add_argument("--expert-fp16", action="store_true", help="Compress expert MLP IR weights to FP16.")
    parser.add_argument("--layer-fp16", action="store_true", help="Compress layer attention/cache IR weights to FP16. Not recommended yet.")
    parser.add_argument("--fused-hot-gather", action="store_true", help="Export one fused attention + hot gather MoE graph per MoE layer.")
    parser.add_argument("--hot-plan-json", default="", help="Hot expert plan JSON for --fused-hot-gather.")
    parser.add_argument("--max-experts", type=int, default=0, help="Optional top-N from hot plan.")
    parser.add_argument("--final-topk", type=int, default=0, help="Also export final_norm_topkK.xml for fast greedy generation.")
    parser.add_argument("--final-argmax", action="store_true", help="Also export final_norm_argmax.xml for fastest greedy generation.")
    parser.add_argument("--final-only", action="store_true", help="Only export final head/top-k graphs and update metadata.")
    parser.add_argument("--only-layers", nargs="*", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def save_model(module: torch.nn.Module, example, path: Path, fp16: bool, skip_existing: bool) -> None:
    if skip_existing and path.exists() and path.with_suffix(".bin").exists():
        print(f"skip existing: {path}", flush=True)
        return
    import openvino as ov

    path.parent.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(module.eval(), example_input=example)
    ov.save_model(ov_model, path, compress_to_fp16=fp16)
    print(f"saved: {path.resolve()}", flush=True)


def main() -> int:
    from transformers import AutoModel

    args = parse_args()
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    out_dir = Path(args.output_dir)

    hidden = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float32)
    position_ids = torch.full((1, 1), args.past_len, dtype=torch.long)
    mask = torch.zeros((1, 1, 1, args.past_len + 1), dtype=torch.float32)
    past_key = torch.zeros((1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim), dtype=torch.float32)
    past_value = torch.zeros_like(past_key)

    layers = args.only_layers if args.only_layers is not None else list(range(model.config.num_hidden_layers))
    metadata = {
        "model": args.model,
        "past_len": args.past_len,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_key_value_heads": model.config.num_key_value_heads,
        "v_head_dim": model.config.v_head_dim,
        "expert_fp16": bool(args.expert_fp16),
        "layer_fp16": bool(args.layer_fp16),
        "fused_hot_gather": bool(args.fused_hot_gather),
        "hot_plan_json": args.hot_plan_json,
        "max_experts": int(args.max_experts),
        "layers": {},
    }
    metadata_path = out_dir / "metadata.json"
    if args.final_only and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not args.final_only:
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            layer_dir = out_dir / f"layer_{layer_idx:02d}"
            if hasattr(layer.mlp, "gate") and args.fused_hot_gather:
                if not args.hot_plan_json:
                    raise ValueError("--fused-hot-gather requires --hot-plan-json")
                expert_ids = expert_ids_from_plan(Path(args.hot_plan_json), layer_idx, args.max_experts)
                save_model(
                    DecodeMoEHotGatherFusedLayer(layer, model.config, expert_ids),
                    (hidden, position_ids, mask, past_key, past_value),
                    layer_dir / "fused_hot_gather_layer.xml",
                    args.layer_fp16,
                    args.skip_existing,
                )
                metadata["layers"][str(layer_idx)] = {
                    "kind": "fused_hot_moe",
                    "experts": len(layer.mlp.experts),
                    "expert_ids": expert_ids,
                    "expert_count": len(expert_ids),
                }
            elif hasattr(layer.mlp, "gate"):
                save_model(
                    DecodeMoEAttentionGate(layer, model.config),
                    (hidden, position_ids, mask, past_key, past_value),
                    layer_dir / "attention_gate.xml",
                    args.layer_fp16,
                    args.skip_existing,
                )
                save_model(AddMoEResidual(), (hidden, hidden, hidden), layer_dir / "add_moe_residual.xml", False, args.skip_existing)
                experts_dir = layer_dir / "experts"
                for expert_id, expert in enumerate(layer.mlp.experts):
                    save_model(
                        ExpertOnly(expert),
                        hidden.reshape(-1, hidden.shape[-1]),
                        experts_dir / f"expert_{expert_id:02d}.xml",
                        args.expert_fp16,
                        args.skip_existing,
                    )
                metadata["layers"][str(layer_idx)] = {"kind": "moe", "experts": len(layer.mlp.experts)}
            else:
                save_model(
                    DecodeDenseLayer(layer, model.config),
                    (hidden, position_ids, mask, past_key, past_value),
                    layer_dir / "dense_layer.xml",
                    args.layer_fp16,
                    args.skip_existing,
                )
                metadata["layers"][str(layer_idx)] = {"kind": "dense"}

    save_model(FinalNormHead(model.model.norm, model.lm_head), hidden, out_dir / "final_norm_head.xml", False, args.skip_existing)
    if args.final_argmax:
        save_model(FinalNormArgmax(model.model.norm, model.lm_head), hidden, out_dir / "final_norm_argmax.xml", False, args.skip_existing)
        metadata["final_argmax"] = True
    if args.final_topk > 0:
        save_model(
            FinalNormTopK(model.model.norm, model.lm_head, args.final_topk),
            hidden,
            out_dir / f"final_norm_topk{args.final_topk}.xml",
            False,
            args.skip_existing,
        )
        metadata["final_topk_k"] = int(args.final_topk)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metadata: {metadata_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
