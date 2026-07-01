from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .export_hot_expert_pack import expert_ids_from_plan
from .sparse_decode_wrappers import DecodeHotGatherBlock, FinalNormArgmax, FinalNormHead, FinalNormTopK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fused multi-layer hot-gather decode blocks.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--output-dir", default="openvino_models/fused_hot_gather_blocks_past677_top61_fp16")
    parser.add_argument("--past-len", type=int, default=677)
    parser.add_argument("--hot-plan-json", required=True)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--final-topk", type=int, default=0, help="Also export final_norm_topkK.xml for fast greedy generation.")
    parser.add_argument("--final-argmax", action="store_true", help="Also export final_norm_argmax.xml for fastest greedy generation.")
    parser.add_argument("--fp16", action="store_true")
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
    if args.block_size < 1:
        raise ValueError("--block-size must be >= 1")
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

    layer_ids = list(range(model.config.num_hidden_layers))
    hot_experts_by_layer = {
        layer_id: expert_ids_from_plan(Path(args.hot_plan_json), layer_id)
        for layer_id in layer_ids
        if hasattr(model.model.layers[layer_id].mlp, "gate")
    }
    metadata = {
        "model": args.model,
        "past_len": args.past_len,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_key_value_heads": model.config.num_key_value_heads,
        "v_head_dim": model.config.v_head_dim,
        "fp16": bool(args.fp16),
        "block_size": int(args.block_size),
        "hot_plan_json": args.hot_plan_json,
        "layers": {},
        "blocks": [],
    }

    for block_index, start in enumerate(range(0, len(layer_ids), args.block_size)):
        block_layer_ids = layer_ids[start : start + args.block_size]
        block_layers = [model.model.layers[layer_id] for layer_id in block_layer_ids]
        block_dir = out_dir / f"block_{block_layer_ids[0]:02d}_{block_layer_ids[-1]:02d}"
        example = (hidden, position_ids, mask) + tuple(
            item for _layer_id in block_layer_ids for item in (past_key, past_value)
        )
        save_model(
            DecodeHotGatherBlock(block_layers, model.config, block_layer_ids, hot_experts_by_layer),
            example,
            block_dir / "decode_block.xml",
            args.fp16,
            args.skip_existing,
        )
        metadata["blocks"].append(
            {
                "index": block_index,
                "dir": block_dir.name,
                "layers": block_layer_ids,
            }
        )
        for layer_id in block_layer_ids:
            layer = model.model.layers[layer_id]
            if hasattr(layer.mlp, "gate"):
                metadata["layers"][str(layer_id)] = {
                    "kind": "block_hot_moe",
                    "experts": len(layer.mlp.experts),
                    "expert_ids": hot_experts_by_layer[layer_id],
                    "expert_count": len(hot_experts_by_layer[layer_id]),
                }
            else:
                metadata["layers"][str(layer_id)] = {"kind": "block_dense"}

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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "blocks": len(metadata["blocks"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
