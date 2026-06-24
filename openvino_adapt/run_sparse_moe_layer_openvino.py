from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .sparse_moe import host_sparse_moe_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one host-dispatched sparse MoE layer from OpenVINO subgraphs.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--artifact-dir", default="openvino_models/sparse_moe_layer1_subset")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    import openvino as ov
    from transformers import AutoModel

    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    torch.manual_seed(args.seed)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    mlp = model.model.layers[args.layer].mlp
    hidden = torch.randn(1, args.tokens, model.config.hidden_size, dtype=torch.float32)
    hidden_np = hidden.numpy()

    core = ov.Core()
    compile_start = time.time()
    gate = core.compile_model(artifact_dir / "gate.xml", args.device)
    shared = core.compile_model(artifact_dir / "shared_experts.xml", args.device)
    base_compile_seconds = time.time() - compile_start
    gate_start = time.time()
    topk_idx_np, topk_weight_np = list(gate([hidden_np]).values())
    gate_seconds = time.time() - gate_start
    topk_idx = topk_idx_np.reshape(-1, topk_idx_np.shape[-1])
    topk_weight = topk_weight_np.reshape(-1, topk_weight_np.shape[-1]).astype(np.float32)
    flat = hidden_np.reshape(-1, hidden_np.shape[-1])

    missing = sorted({int(expert_id) for expert_id in topk_idx.reshape(-1) if not (artifact_dir / f"expert_{int(expert_id):02d}.xml").exists()})
    if missing:
        print(json.dumps({"missing_experts": missing, "topk_idx": topk_idx.tolist()}, ensure_ascii=False, indent=2))
        return 2

    expert_cache = {}
    expert_compile_seconds: dict[int, float] = {}
    expert_infer_seconds: dict[int, float] = {}
    y = np.zeros_like(flat, dtype=np.float32)
    for token_index in range(flat.shape[0]):
        token = flat[token_index : token_index + 1]
        token_out = np.zeros_like(token, dtype=np.float32)
        for route_index in range(topk_idx.shape[1]):
            expert_id = int(topk_idx[token_index, route_index])
            if expert_id not in expert_cache:
                expert_compile_start = time.time()
                expert_cache[expert_id] = core.compile_model(artifact_dir / f"expert_{expert_id:02d}.xml", args.device)
                expert_compile_seconds[expert_id] = time.time() - expert_compile_start
            expert_start = time.time()
            expert_out = next(iter(expert_cache[expert_id]([token]).values()))
            expert_infer_seconds[expert_id] = expert_infer_seconds.get(expert_id, 0.0) + (time.time() - expert_start)
            token_out += expert_out.astype(np.float32) * topk_weight[token_index, route_index]
        y[token_index : token_index + 1] = token_out
    shared_start = time.time()
    shared_out = next(iter(shared([hidden_np]).values())).reshape(flat.shape).astype(np.float32)
    shared_seconds = time.time() - shared_start
    y = (y + shared_out).reshape(hidden_np.shape)

    with torch.no_grad():
        torch_ref = host_sparse_moe_forward(mlp, hidden).float().numpy()

    payload = {
        "layer": args.layer,
        "tokens": args.tokens,
        "device": args.device,
        "topk_idx": topk_idx.tolist(),
        "compiled_experts": sorted(expert_cache.keys()),
        "base_compile_seconds": base_compile_seconds,
        "gate_seconds": gate_seconds,
        "shared_seconds": shared_seconds,
        "expert_compile_seconds": {str(k): v for k, v in sorted(expert_compile_seconds.items())},
        "expert_infer_seconds": {str(k): v for k, v in sorted(expert_infer_seconds.items())},
        "total_expert_compile_seconds": float(sum(expert_compile_seconds.values())),
        "total_expert_infer_seconds": float(sum(expert_infer_seconds.values())),
        "max_abs_diff": float(np.max(np.abs(torch_ref - y))),
        "mean_abs_diff": float(np.mean(np.abs(torch_ref - y))),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
