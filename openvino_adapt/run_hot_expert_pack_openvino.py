from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .sparse_moe import HotExpertPack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and validate a hot expert pack OpenVINO graph.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--artifact-dir", default="openvino_models/hot_expert_pack_layer1")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> int:
    import openvino as ov
    from transformers import AutoModel

    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    expert_ids = [int(x) for x in metadata["expert_ids"]]

    torch.manual_seed(args.seed)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    mlp = model.model.layers[args.layer].mlp
    hidden = torch.randn(1, 1, model.config.hidden_size)
    with torch.no_grad():
        topk_idx, topk_weight, _ = mlp.gate(hidden)
        ref = HotExpertPack(mlp, expert_ids).eval()(hidden, topk_idx, topk_weight)

    core = ov.Core()
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
    compile_start = time.perf_counter()
    compiled = core.compile_model(artifact_dir / "hot_expert_pack.xml", args.device)
    compile_seconds = time.perf_counter() - compile_start

    feeds = [hidden.numpy(), topk_idx.numpy(), topk_weight.numpy()]
    infer_seconds: list[float] = []
    out = None
    for _ in range(args.runs):
        infer_start = time.perf_counter()
        result = compiled(feeds)
        infer_seconds.append(time.perf_counter() - infer_start)
        out = next(iter(result.values()))
    assert out is not None

    selected = sorted({int(x) for x in topk_idx.reshape(-1).tolist()})
    hot_hits = sorted(set(selected) & set(expert_ids))
    payload = {
        "artifact_dir": str(artifact_dir),
        "device": args.device,
        "layer": args.layer,
        "expert_ids": expert_ids,
        "selected_experts": selected,
        "hot_hits": hot_hits,
        "runs": args.runs,
        "compile_seconds": compile_seconds,
        "infer_seconds": infer_seconds,
        "infer_seconds_mean": float(np.mean(infer_seconds)),
        "infer_seconds_min": float(np.min(infer_seconds)),
        "max_abs": float(np.max(np.abs(ref.numpy() - out))),
        "mean_abs": float(np.mean(np.abs(ref.numpy() - out))),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
