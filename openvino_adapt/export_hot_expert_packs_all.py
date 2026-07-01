from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .export_hot_expert_pack import expert_ids_from_plan
from .sparse_moe import HotExpertGatherPack, HotExpertPack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hot expert pack graphs for every MoE decode layer.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--plan-json", default="outputs_openvino_ngram_smoke/expert_precision_plan_4samples_stock_fp4.json")
    parser.add_argument("--output-dir", default="openvino_models/hot_expert_packs")
    parser.add_argument("--max-experts", type=int, default=0, help="0 keeps only plan reason=hot; otherwise take top-N by score per layer.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gather", action="store_true", help="Gather only routed top-k experts inside each pack graph.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    import openvino as ov
    from transformers import AutoModel

    args = parse_args()
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "plan_json": args.plan_json,
        "max_experts": int(args.max_experts),
        "fp16": bool(args.fp16),
        "pack_mode": "gather" if args.gather else "dense_pack",
        "layers": {},
    }
    hidden = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float32)

    for layer_id, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, "experts"):
            continue
        layer_dir = root / f"layer_{layer_id:02d}"
        xml = layer_dir / "hot_expert_pack.xml"
        metadata_path = layer_dir / "metadata.json"
        expert_ids = expert_ids_from_plan(Path(args.plan_json), layer_id, args.max_experts)
        layer_dir.mkdir(parents=True, exist_ok=True)
        if not (args.skip_existing and xml.exists() and xml.with_suffix(".bin").exists()):
            topk = len(layer.mlp.gate(hidden)[0].reshape(1, -1)[0])
            topk_idx = torch.zeros((1, topk), dtype=torch.long)
            topk_weight = torch.zeros((1, topk), dtype=torch.float32)
            module_cls = HotExpertGatherPack if args.gather else HotExpertPack
            ov_model = ov.convert_model(module_cls(layer.mlp, expert_ids).eval(), example_input=(hidden, topk_idx, topk_weight))
            ov.save_model(ov_model, xml, compress_to_fp16=args.fp16)
            print(f"saved: {xml.resolve()}", flush=True)
        metadata = {
            "model": args.model,
            "layer": layer_id,
            "expert_ids": expert_ids,
            "expert_count": len(expert_ids),
            "max_experts": int(args.max_experts),
            "hidden_size": model.config.hidden_size,
            "topk": 6,
            "fp16": bool(args.fp16),
            "pack_mode": "gather" if args.gather else "dense_pack",
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["layers"][str(layer_id)] = metadata

    (root / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(root), "layers": len(summary["layers"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
