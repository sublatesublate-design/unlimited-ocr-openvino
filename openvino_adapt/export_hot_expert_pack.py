from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .sparse_moe import HotExpertGatherPack, HotExpertPack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one layer's hot experts as a packed MoE OpenVINO graph.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--plan-json", default="outputs_openvino_ngram_smoke/expert_precision_plan_4samples_stock_fp4.json")
    parser.add_argument("--expert-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-experts", type=int, default=0, help="0 keeps only plan reason=hot; otherwise take top-N by score.")
    parser.add_argument("--output-dir", default="openvino_models/hot_expert_pack_layer1")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gather", action="store_true", help="Gather only routed top-k experts inside the pack graph.")
    return parser.parse_args()


def expert_ids_from_plan(path: Path, layer_id: int, max_experts: int = 0) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    layer = data.get("layers", {}).get(str(layer_id))
    if layer is not None and "expert_ids" in layer:
        expert_ids = [int(expert_id) for expert_id in layer["expert_ids"]]
        return sorted(expert_ids[:max_experts] if max_experts > 0 else expert_ids)
    plan = data["plan"][str(layer_id)]["experts"]
    if max_experts <= 0:
        return sorted(int(expert_id) for expert_id, info in plan.items() if info.get("reason") == "hot")
    ranked = sorted(
        plan.items(),
        key=lambda item: (
            float(item[1].get("score", 0.0)),
            int(item[1].get("count", 0)),
            float(item[1].get("weight_sum", 0.0)),
            -int(item[0]),
        ),
        reverse=True,
    )
    return sorted(int(expert_id) for expert_id, _ in ranked[:max_experts])


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
    layer = model.model.layers[args.layer]
    if not hasattr(layer.mlp, "experts"):
        raise ValueError(f"Layer {args.layer} is not a routed MoE layer.")

    expert_ids = (
        sorted(args.expert_ids)
        if args.expert_ids is not None
        else expert_ids_from_plan(Path(args.plan_json), args.layer, args.max_experts)
    )
    module_cls = HotExpertGatherPack if args.gather else HotExpertPack
    module = module_cls(layer.mlp, expert_ids).eval()
    hidden = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float32)
    topk = len(layer.mlp.gate(hidden)[0].reshape(1, -1)[0])
    topk_idx = torch.zeros((1, topk), dtype=torch.long)
    topk_weight = torch.zeros((1, topk), dtype=torch.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(module, example_input=(hidden, topk_idx, topk_weight))
    ov.save_model(ov_model, out_dir / "hot_expert_pack.xml", compress_to_fp16=args.fp16)
    metadata = {
        "model": args.model,
        "layer": args.layer,
        "expert_ids": expert_ids,
        "expert_count": len(expert_ids),
        "max_experts": int(args.max_experts),
        "hidden_size": model.config.hidden_size,
        "topk": topk,
        "fp16": bool(args.fp16),
        "pack_mode": "gather" if args.gather else "dense_pack",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
