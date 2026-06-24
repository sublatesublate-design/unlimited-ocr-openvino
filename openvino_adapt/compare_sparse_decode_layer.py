from __future__ import annotations

import argparse
import json

import torch

from .sparse_decode_wrappers import DecodeMoEAttentionGate
from .sparse_moe import host_sparse_moe_forward
from .wrappers import DecoderDecodeOneStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare split sparse decode layer against dense decode layer math.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--past-len", type=int, default=677)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    from transformers import AutoModel

    args = parse_args()
    torch.manual_seed(args.seed)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    layer = model.model.layers[args.layer]
    if not hasattr(layer.mlp, "gate"):
        raise ValueError(f"Layer {args.layer} is not MoE")

    hidden = torch.randn(1, 1, model.config.hidden_size)
    position_ids = torch.full((1, 1), args.past_len, dtype=torch.long)
    mask = torch.zeros((1, 1, 1, args.past_len + 1), dtype=torch.float32)
    past_key = torch.randn(1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim) * 0.01
    past_value = torch.randn(1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim) * 0.01

    dense_decode = DecoderDecodeOneStep(model)
    split = DecodeMoEAttentionGate(layer, model.config).eval()
    with torch.no_grad():
        dense_hidden, dense_key, dense_value = dense_decode._layer_forward(
            layer, hidden, position_ids, mask, past_key, past_value
        )
        attn_residual, moe_input, topk_idx, topk_weight, shared_out, key_new, value_new = split(
            hidden, position_ids, mask, past_key, past_value
        )
        routed_plus_shared = host_sparse_moe_forward(layer.mlp, moe_input)
        routed_only = routed_plus_shared - layer.mlp.shared_experts(moe_input)
        split_hidden = attn_residual + shared_out + routed_only

    payload = {
        "layer": args.layer,
        "past_len": args.past_len,
        "topk_idx": topk_idx.reshape(-1, topk_idx.shape[-1]).tolist(),
        "hidden_max_abs": float((dense_hidden - split_hidden).abs().max()),
        "hidden_mean_abs": float((dense_hidden - split_hidden).abs().mean()),
        "key_max_abs": float((dense_key - key_new).abs().max()),
        "value_max_abs": float((dense_value - value_new).abs().max()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
