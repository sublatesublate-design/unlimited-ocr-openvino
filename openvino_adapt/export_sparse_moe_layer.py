from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .sparse_moe import ExpertOnly, MoEGateOnly, SharedExpertsOnly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one MoE layer as host-dispatched OpenVINO subgraphs.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output-dir", default="openvino_models/sparse_moe_layer0")
    parser.add_argument("--experts", nargs="*", type=int, default=[0, 1, 2], help="Expert ids to export. Use --all-experts for 0..63.")
    parser.add_argument("--all-experts", action="store_true")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def save_model(module: torch.nn.Module, example: torch.Tensor, path: Path, fp16: bool) -> None:
    import openvino as ov

    path.parent.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(module, example_input=example)
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
    mlp = model.model.layers[args.layer].mlp
    if not hasattr(mlp, "gate") or not hasattr(mlp, "experts"):
        raise ValueError(f"Layer {args.layer} uses {type(mlp).__name__}, not a routed MoE layer.")
    hidden = torch.zeros((1, args.tokens, model.config.hidden_size), dtype=torch.float32)
    out_dir = Path(args.output_dir)

    experts = list(range(len(mlp.experts))) if args.all_experts else args.experts
    metadata = {
        "model": args.model,
        "layer": args.layer,
        "hidden_size": model.config.hidden_size,
        "tokens": args.tokens,
        "fp16": bool(args.fp16),
        "experts": experts,
        "top_k": model.config.num_experts_per_tok,
        "n_routed_experts": model.config.n_routed_experts,
        "n_shared_experts": model.config.n_shared_experts,
    }

    save_model(MoEGateOnly(mlp).eval(), hidden, out_dir / "gate.xml", args.fp16)
    if getattr(model.config, "n_shared_experts", None) is not None:
        save_model(SharedExpertsOnly(mlp).eval(), hidden, out_dir / "shared_experts.xml", args.fp16)
    for expert_id in experts:
        save_model(ExpertOnly(mlp.experts[expert_id]).eval(), hidden.reshape(-1, hidden.shape[-1]), out_dir / f"expert_{expert_id:02d}.xml", args.fp16)

    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metadata: {(out_dir / 'metadata.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
