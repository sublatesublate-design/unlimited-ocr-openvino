from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .wrappers import DecoderDecodeOneStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fused sparse MoE decode layer OpenVINO graph.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--artifact-dir", default="openvino_models/fused_sparse_decode_layer1")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--past-len", type=int, default=677)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-json", default="")
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
        ref_hidden, ref_key, ref_value = DecoderDecodeOneStep(model)._layer_forward(
            layer, hidden, position_ids, mask, past_key, past_value
        )

    core = ov.Core()
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})

    compile_start = time.perf_counter()
    compiled = core.compile_model(Path(args.artifact_dir) / "fused_layer.xml", args.device)
    compile_seconds = time.perf_counter() - compile_start

    feeds = [hidden.numpy(), position_ids.numpy(), mask.numpy(), past_key.numpy(), past_value.numpy()]
    infer_seconds: list[float] = []
    outputs = None
    for _ in range(args.runs):
        infer_start = time.perf_counter()
        result = compiled(feeds)
        infer_seconds.append(time.perf_counter() - infer_start)
        outputs = [result[compiled.output(index)] for index in range(len(compiled.outputs))]
    assert outputs is not None
    ov_hidden, ov_key, ov_value = outputs

    payload = {
        "artifact_dir": str(Path(args.artifact_dir)),
        "device": args.device,
        "layer": args.layer,
        "runs": args.runs,
        "compile_seconds": compile_seconds,
        "infer_seconds": infer_seconds,
        "infer_seconds_mean": float(np.mean(infer_seconds)),
        "infer_seconds_min": float(np.min(infer_seconds)),
        "hidden_max_abs": float(np.max(np.abs(ref_hidden.numpy() - ov_hidden))),
        "hidden_mean_abs": float(np.mean(np.abs(ref_hidden.numpy() - ov_hidden))),
        "key_max_abs": float(np.max(np.abs(ref_key.numpy() - ov_key))),
        "value_max_abs": float(np.max(np.abs(ref_value.numpy() - ov_value))),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
