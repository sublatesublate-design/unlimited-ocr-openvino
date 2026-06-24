from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .wrappers import DecoderDecodeOneStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one split sparse decode layer using OpenVINO subgraphs.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--decode-layer-dir", default="openvino_models/sparse_decode_layer1")
    parser.add_argument("--moe-dir", default="openvino_models/sparse_moe_layer1_seed0")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--past-len", type=int, default=677)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    import openvino as ov
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

    hidden = torch.randn(1, 1, model.config.hidden_size)
    position_ids = torch.full((1, 1), args.past_len, dtype=torch.long)
    mask = torch.zeros((1, 1, 1, args.past_len + 1), dtype=torch.float32)
    past_key = torch.randn(1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim) * 0.01
    past_value = torch.randn(1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim) * 0.01

    with torch.no_grad():
        dense_decode = DecoderDecodeOneStep(model)
        ref_hidden, ref_key, ref_value = dense_decode._layer_forward(
            layer, hidden, position_ids, mask, past_key, past_value
        )

    core = ov.Core()
    decode_dir = Path(args.decode_layer_dir)
    moe_dir = Path(args.moe_dir)
    attention_gate = core.compile_model(decode_dir / "attention_gate.xml", args.device)
    add_residual = core.compile_model(decode_dir / "add_moe_residual.xml", args.device)

    attention_outputs = attention_gate([hidden.numpy(), position_ids.numpy(), mask.numpy(), past_key.numpy(), past_value.numpy()])
    attn_residual = attention_outputs[attention_gate.output(0)]
    moe_input = attention_outputs[attention_gate.output(1)]
    topk_idx = attention_outputs[attention_gate.output(2)]
    topk_weight = attention_outputs[attention_gate.output(3)]
    shared_out = attention_outputs[attention_gate.output(4)]
    key_new = attention_outputs[attention_gate.output(5)]
    value_new = attention_outputs[attention_gate.output(6)]
    flat = moe_input.reshape(-1, moe_input.shape[-1]).astype(np.float32)
    topk_idx_flat = topk_idx.reshape(flat.shape[0], -1)
    topk_weight_flat = topk_weight.reshape(flat.shape[0], -1).astype(np.float32)
    missing = sorted({int(expert_id) for expert_id in topk_idx_flat.reshape(-1) if not (moe_dir / f"expert_{int(expert_id):02d}.xml").exists()})
    if missing:
        print(json.dumps({"missing_experts": missing, "topk_idx": topk_idx_flat.tolist()}, ensure_ascii=False, indent=2))
        return 2

    routed = np.zeros_like(flat, dtype=np.float32)
    expert_cache = {}
    for token_index in range(flat.shape[0]):
        token = flat[token_index : token_index + 1]
        token_out = np.zeros_like(token, dtype=np.float32)
        for route_index in range(topk_idx_flat.shape[1]):
            expert_id = int(topk_idx_flat[token_index, route_index])
            if expert_id not in expert_cache:
                expert_cache[expert_id] = core.compile_model(moe_dir / f"expert_{expert_id:02d}.xml", args.device)
            expert_out = next(iter(expert_cache[expert_id]([token]).values())).astype(np.float32)
            token_out += expert_out * topk_weight_flat[token_index, route_index]
        routed[token_index : token_index + 1] = token_out
    routed = routed.reshape(moe_input.shape)

    split_hidden = next(iter(add_residual([attn_residual, shared_out, routed]).values()))
    payload = {
        "layer": args.layer,
        "device": args.device,
        "topk_idx": topk_idx_flat.tolist(),
        "compiled_experts": sorted(expert_cache.keys()),
        "hidden_max_abs": float(np.max(np.abs(ref_hidden.numpy() - split_hidden))),
        "hidden_mean_abs": float(np.mean(np.abs(ref_hidden.numpy() - split_hidden))),
        "key_max_abs": float(np.max(np.abs(ref_key.numpy() - key_new))),
        "value_max_abs": float(np.max(np.abs(ref_value.numpy() - value_new))),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
