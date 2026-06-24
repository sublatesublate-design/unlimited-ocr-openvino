from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .sparse_decode_wrappers import DecodeMoEAttentionGate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile routed expert usage for sparse decode quantization policy.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--past-len", type=int, default=677)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default="outputs_openvino_ngram_smoke/expert_route_profile.json")
    parser.add_argument("--hot-ratio", type=float, default=0.10, help="Fraction of experts per layer kept as hot.")
    parser.add_argument("--cold-ratio", type=float, default=0.50, help="Fraction of experts per layer marked as cold.")
    return parser.parse_args()


def route_one_layer(model, layer, hidden, position_ids, attention_mask, past_key, past_value):
    splitter = DecodeMoEAttentionGate(layer, model.config).eval()
    with torch.no_grad():
        attn_residual, moe_input, topk_idx, topk_weight, shared_out, key_new, value_new = splitter(
            hidden, position_ids, attention_mask, past_key, past_value
        )
        # Continue with the original MoE for realistic downstream hidden states.
        hidden = attn_residual + layer.mlp(moe_input)
    return hidden, key_new, value_new, topk_idx, topk_weight


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

    n_layers = model.config.num_hidden_layers
    n_experts = model.config.n_routed_experts
    stats = {
        str(layer_id): {
            "count": [0 for _ in range(n_experts)],
            "weight_sum": [0.0 for _ in range(n_experts)],
        }
        for layer_id in range(n_layers)
        if hasattr(model.model.layers[layer_id].mlp, "gate")
    }

    attention_mask = torch.zeros((1, 1, 1, args.past_len + 1), dtype=torch.float32)
    position_ids = torch.full((1, 1), args.past_len, dtype=torch.long)
    for sample_idx in range(args.samples):
        hidden = torch.randn(1, 1, model.config.hidden_size)
        past_keys = [
            torch.randn(1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim) * 0.01
            for _ in range(n_layers)
        ]
        past_values = [
            torch.randn(1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim) * 0.01
            for _ in range(n_layers)
        ]
        for layer_id, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, "gate"):
                hidden, _, _, topk_idx, topk_weight = route_one_layer(
                    model,
                    layer,
                    hidden,
                    position_ids,
                    attention_mask,
                    past_keys[layer_id],
                    past_values[layer_id],
                )
                topk_idx = topk_idx.reshape(-1).tolist()
                topk_weight = topk_weight.reshape(-1).float().tolist()
                layer_stats = stats[str(layer_id)]
                for expert_id, weight in zip(topk_idx, topk_weight):
                    layer_stats["count"][int(expert_id)] += 1
                    layer_stats["weight_sum"][int(expert_id)] += float(weight)
            else:
                with torch.no_grad():
                    # Dense layer 0 only. Use official layer path for downstream hidden state.
                    from .wrappers import DecoderDecodeOneStep

                    wrapper = DecoderDecodeOneStep(model)
                    hidden, _, _ = wrapper._layer_forward(
                        layer,
                        hidden,
                        position_ids,
                        attention_mask,
                        past_keys[layer_id],
                        past_values[layer_id],
                    )

    policy: dict[str, dict] = {}
    hot_n = max(1, int(n_experts * args.hot_ratio))
    cold_n = max(1, int(n_experts * args.cold_ratio))
    for layer_id, layer_stats in stats.items():
        scored = []
        for expert_id, (count, weight_sum) in enumerate(zip(layer_stats["count"], layer_stats["weight_sum"])):
            scored.append(
                {
                    "expert": expert_id,
                    "count": count,
                    "weight_sum": weight_sum,
                    "score": count + weight_sum,
                }
            )
        ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
        cold_ranked = sorted(scored, key=lambda item: item["score"])
        hot = [item["expert"] for item in ranked[:hot_n]]
        cold = [item["expert"] for item in cold_ranked[:cold_n]]
        unused = [item["expert"] for item in scored if item["count"] == 0]
        policy[layer_id] = {
            "hot_keep_int8_or_fp16": hot,
            "cold_int4_or_experimental_lowbit": cold,
            "unused_in_profile": unused,
            "ranked": ranked,
        }

    payload = {
        "model": args.model,
        "samples": args.samples,
        "past_len": args.past_len,
        "note": "INT2 is not available in local NNCF/OpenVINO. Treat cold/unused experts as candidates for INT4/FP4/NF4 or a custom INT2 kernel experiment.",
        "policy": policy,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
