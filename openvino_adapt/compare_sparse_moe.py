from __future__ import annotations

import argparse
import json

import torch

from .sparse_moe import host_sparse_moe_forward
from .wrappers import DecoderPrefillWithKV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare host-dispatched sparse MoE against existing MoE paths.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def main() -> int:
    from transformers import AutoModel

    args = parse_args()
    torch.manual_seed(args.seed)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()

    mlp = model.model.layers[args.layer].mlp
    hidden_size = model.config.hidden_size
    hidden = torch.randn(1, args.tokens, hidden_size, dtype=torch.float32)
    if args.dtype == "bfloat16":
        hidden = hidden.to(torch.bfloat16)
        model = model.to(torch.bfloat16)
        mlp = model.model.layers[args.layer].mlp

    with torch.no_grad():
        official = mlp(hidden)
        dense = DecoderPrefillWithKV._dense_moe_forward(mlp, hidden)
        host_sparse = host_sparse_moe_forward(mlp, hidden)

    payload = {
        "layer": args.layer,
        "tokens": args.tokens,
        "dtype": args.dtype,
        "official_vs_dense_max_abs": float((official.float() - dense.float()).abs().max()),
        "official_vs_host_sparse_max_abs": float((official.float() - host_sparse.float()).abs().max()),
        "dense_vs_host_sparse_max_abs": float((dense.float() - host_sparse.float()).abs().max()),
        "dense_vs_host_sparse_mean_abs": float((dense.float() - host_sparse.float()).abs().mean()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
