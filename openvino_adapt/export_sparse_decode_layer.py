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
    DecodeMoEFusedLayer,
    DecodeMoEHotGatherFusedLayer,
    FinalNormArgmax,
    FinalNormHead,
    FinalNormTopK,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export split decode layer graphs for sparse MoE host dispatch.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--past-len", type=int, default=677)
    parser.add_argument("--output-dir", default="openvino_models/sparse_decode_layer1")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fused", action="store_true", help="Export one packed MoE layer graph instead of split host-dispatch graphs.")
    parser.add_argument("--fused-hot-gather", action="store_true", help="Export attention + hot gather MoE + residual as one graph.")
    parser.add_argument("--hot-plan-json", default="", help="Hot expert plan JSON for --fused-hot-gather.")
    parser.add_argument("--max-experts", type=int, default=0, help="Optional top-N from hot plan.")
    parser.add_argument("--final-head", action="store_true", help="Export final_norm_head.xml too.")
    parser.add_argument("--final-topk", type=int, default=0, help="Also export final_norm_topkK.xml.")
    parser.add_argument("--final-argmax", action="store_true", help="Also export final_norm_argmax.xml.")
    return parser.parse_args()


def save_model(module: torch.nn.Module, example, path: Path, fp16: bool) -> None:
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
    layer = model.model.layers[args.layer]
    hidden = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float32)
    position_ids = torch.full((1, 1), args.past_len, dtype=torch.long)
    mask = torch.zeros((1, 1, 1, args.past_len + 1), dtype=torch.float32)
    past_key = torch.zeros((1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim), dtype=torch.float32)
    past_value = torch.zeros_like(past_key)

    out_dir = Path(args.output_dir)
    expert_ids: list[int] | None = None
    if hasattr(layer.mlp, "gate") and args.fused_hot_gather:
        if not args.hot_plan_json:
            raise ValueError("--fused-hot-gather requires --hot-plan-json")
        expert_ids = expert_ids_from_plan(Path(args.hot_plan_json), args.layer, args.max_experts)
        save_model(
            DecodeMoEHotGatherFusedLayer(layer, model.config, expert_ids),
            (hidden, position_ids, mask, past_key, past_value),
            out_dir / "fused_hot_gather_layer.xml",
            args.fp16,
        )
        kind = "fused_hot_moe"
    elif hasattr(layer.mlp, "gate") and args.fused:
        save_model(
            DecodeMoEFusedLayer(layer, model.config),
            (hidden, position_ids, mask, past_key, past_value),
            out_dir / "fused_layer.xml",
            args.fp16,
        )
        kind = "fused_moe"
    elif hasattr(layer.mlp, "gate"):
        save_model(
            DecodeMoEAttentionGate(layer, model.config),
            (hidden, position_ids, mask, past_key, past_value),
            out_dir / "attention_gate.xml",
            args.fp16,
        )
        save_model(AddMoEResidual(), (hidden, hidden, hidden), out_dir / "add_moe_residual.xml", args.fp16)
        kind = "moe"
    else:
        save_model(
            DecodeDenseLayer(layer, model.config),
            (hidden, position_ids, mask, past_key, past_value),
            out_dir / "dense_layer.xml",
            args.fp16,
        )
        kind = "dense"

    if args.final_head:
        save_model(FinalNormHead(model.model.norm, model.lm_head), hidden, out_dir / "final_norm_head.xml", args.fp16)
    if args.final_argmax:
        save_model(FinalNormArgmax(model.model.norm, model.lm_head), hidden, out_dir / "final_norm_argmax.xml", args.fp16)
    if args.final_topk > 0:
        save_model(FinalNormTopK(model.model.norm, model.lm_head, args.final_topk), hidden, out_dir / f"final_norm_topk{args.final_topk}.xml", args.fp16)

    metadata = {
        "model": args.model,
        "layer": args.layer,
        "kind": kind,
        "past_len": args.past_len,
        "fp16": bool(args.fp16),
        "hidden_size": model.config.hidden_size,
        "num_key_value_heads": model.config.num_key_value_heads,
        "v_head_dim": model.config.v_head_dim,
    }
    if expert_ids is not None:
        metadata["expert_ids"] = expert_ids
        metadata["expert_count"] = len(expert_ids)
        metadata["hot_plan_json"] = args.hot_plan_json
    if args.final_topk > 0:
        metadata["final_topk_k"] = int(args.final_topk)
    if args.final_argmax:
        metadata["final_argmax"] = True
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metadata: {(out_dir / 'metadata.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
