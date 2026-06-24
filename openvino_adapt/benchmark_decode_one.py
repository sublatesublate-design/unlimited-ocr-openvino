from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .artifacts import decode_prior_len, model_metadata
from .runtime import make_core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure decoder_decode_one compile and one-step inference time.")
    parser.add_argument("--model", required=True, help="Path to decoder_decode_one.xml.")
    parser.add_argument("--device", default="CPU", help="CPU, GPU, AUTO:GPU,CPU, HETERO:GPU,CPU, etc.")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--past-len", type=int, default=0, help="Override prior KV length. Defaults to model metadata/static shape.")
    parser.add_argument("--position-id", type=int, default=550)
    parser.add_argument("--token-id", type=int, default=128818)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--compile-only", action="store_true", help="Stop after compile_model returns.")
    return parser.parse_args()


def element_numpy_dtype(element_type) -> np.dtype:
    text = str(element_type)
    if "int64" in text:
        return np.int64
    if "int32" in text:
        return np.int32
    if "float16" in text:
        return np.float16
    return np.float32


def concrete_dim(dim, fallback: int) -> int:
    return int(dim.get_length()) if dim.is_static else fallback


def make_inputs(model, past_len: int, token_id: int, position_id: int) -> list[np.ndarray]:
    inputs: list[np.ndarray] = []
    for index, input_port in enumerate(model.inputs):
        dtype = element_numpy_dtype(input_port.get_element_type())
        shape = input_port.partial_shape
        if index == 0:
            inputs.append(np.asarray([[token_id]], dtype=dtype))
        elif index == 1:
            inputs.append(np.asarray([[position_id]], dtype=dtype))
        elif index == 2:
            mask = np.zeros((1, 1, 1, past_len + 1), dtype=dtype)
            inputs.append(mask)
        else:
            resolved = [concrete_dim(dim, fallback) for dim, fallback in zip(shape, [1, 10, past_len, 128])]
            inputs.append(np.zeros(resolved, dtype=dtype))
    return inputs


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print(f"[decode_diag] read_model {model_path}", flush=True)
    core = make_core(args.cache_dir)
    model = core.read_model(model_path)
    past_len = args.past_len or decode_prior_len(model_path)
    if past_len is None:
        meta = model_metadata(model_path)
        raise ValueError(f"Cannot infer past length for {model_path}; metadata={meta}")

    payload = {
        "model": str(model_path),
        "device": args.device,
        "cache_dir": args.cache_dir,
        "past_len": past_len,
        "runs": args.runs,
        "warmups": args.warmups,
        "metadata": model_metadata(model_path),
    }

    print(f"[decode_diag] compile start device={args.device}", flush=True)
    compile_start = time.time()
    compiled = core.compile_model(model, args.device)
    payload["compile_seconds"] = time.time() - compile_start
    try:
        payload["execution_devices"] = list(compiled.get_property("EXECUTION_DEVICES"))
    except Exception as exc:
        payload["execution_devices_error"] = f"{type(exc).__name__}: {exc}"
    print(f"[decode_diag] compile done seconds={payload['compile_seconds']:.3f}", flush=True)

    if args.compile_only:
        payload["infer_seconds"] = []
        payload["infer_seconds_mean"] = 0.0
        payload["output_shapes"] = []
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    print(f"[decode_diag] build inputs past_len={past_len}", flush=True)
    feeds = make_inputs(model, past_len, args.token_id, args.position_id)
    for _ in range(args.warmups):
        print("[decode_diag] warmup", flush=True)
        compiled(feeds)

    infer_times: list[float] = []
    output_shapes = None
    for run_index in range(args.runs):
        print(f"[decode_diag] infer start run={run_index + 1}", flush=True)
        infer_start = time.time()
        outputs = compiled(feeds)
        infer_times.append(time.time() - infer_start)
        print(f"[decode_diag] infer done run={run_index + 1} seconds={infer_times[-1]:.3f}", flush=True)
        if output_shapes is None:
            output_shapes = [list(value.shape) for value in outputs.values()]

    payload["infer_seconds"] = infer_times
    payload["infer_seconds_mean"] = float(np.mean(infer_times)) if infer_times else 0.0
    payload["output_shapes"] = output_shapes
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
