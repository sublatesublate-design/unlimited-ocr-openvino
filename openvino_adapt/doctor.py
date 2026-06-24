from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from .artifacts import (
    ArtifactSet,
    decode_prior_len,
    model_metadata,
    prefill_seq_len,
    validate_prompt_and_decode_lengths,
)
from .logits_processors import SlidingWindowNoRepeatNgram, select_greedy_token
from .run_prefill_no_cache import encode_prompt_for_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Unlimited-OCR OpenVINO adapter readiness.")
    parser.add_argument("--base-model-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument("--prefill-model", default="openvino_models/unlimited_ocr_kv_dense_prefill277/decoder_prefill_kv.xml")
    parser.add_argument("--decode-model", default="openvino_models/unlimited_ocr_kv_dense/decoder_decode_one.xml")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--page-count", type=int, default=1, help="Number of page images represented in the prompt.")
    parser.add_argument("--ring-window", type=int, default=128)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--image", default="", help="Optional image for a one-token runtime check.")
    parser.add_argument("--run-one-token", action="store_true", help="Compile models and generate one token.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def module_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not installed"
    module = __import__(name)
    return getattr(module, "__version__", "installed")


def bin_size_mb(xml_path: Path) -> float | None:
    bin_path = xml_path.with_suffix(".bin")
    if not bin_path.exists():
        return None
    return round(bin_path.stat().st_size / 1024 / 1024, 2)


def artifact_report(artifacts: ArtifactSet) -> dict:
    paths = {
        "embed_tokens": artifacts.embed_tokens,
        "vision_tokens": artifacts.vision_tokens,
        "prefill_model": artifacts.prefill_model,
        "decode_model": artifacts.decode_model,
        "tokenizer_json": artifacts.tokenizer_dir / "tokenizer.json",
    }
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "bin_mb": bin_size_mb(path) if path.suffix == ".xml" else None,
            "metadata": model_metadata(path) if path.suffix == ".xml" else {},
        }
        for name, path in paths.items()
    }


def check_logits_processor() -> None:
    import numpy as np

    processor = SlidingWindowNoRepeatNgram(ngram_size=3, window=32)
    sequence = [1, 2, 3, 1, 2]
    scores = np.zeros(8, dtype=np.float32)
    token = select_greedy_token(scores, sequence, processor)
    if token == 3:
        raise AssertionError("no-repeat-ngram failed to ban repeated token")


def runtime_one_token(args: argparse.Namespace, artifacts: ArtifactSet, tokenizer) -> dict:
    if not args.image:
        raise ValueError("--run-one-token requires --image")
    if args.page_count != 1:
        raise ValueError("--run-one-token currently accepts --page-count 1; use run_ocr_openvino --continuous for multi-page runtime.")

    import openvino as ov

    from .run_generate_openvino import generate_from_compiled

    core = ov.Core()
    compile_start = time.time()
    embed = core.compile_model(artifacts.embed_tokens, args.device)
    vision = core.compile_model(artifacts.vision_tokens, args.device)
    prefill = core.compile_model(artifacts.prefill_model, args.device)
    decode = core.compile_model(artifacts.decode_model, args.device)
    compile_seconds = time.time() - compile_start

    result = generate_from_compiled(
        embed=embed,
        vision=vision,
        prefill=prefill,
        decode=decode,
        tokenizer=tokenizer,
        image=args.image,
        prompt=args.prompt,
        max_new_tokens=1,
        ring_window=args.ring_window,
    )
    return {
        "compile_seconds": compile_seconds,
        "decode_seconds": result["decode_seconds"],
        "tokens_per_second": result["tokens_per_second"],
        "generated_ids": result["generated_ids"],
        "generated_text": result["text"],
    }


def build_report(args: argparse.Namespace) -> dict:
    import openvino as ov
    from transformers import AutoTokenizer

    artifacts = ArtifactSet.from_paths(args.base_model_dir, args.prefill_model, args.decode_model, args.tokenizer)
    report: dict = {
        "python": sys.version.split()[0],
        "packages": {
            "torch": module_version("torch"),
            "transformers": module_version("transformers"),
            "openvino": module_version("openvino"),
            "huggingface_hub": module_version("huggingface_hub"),
            "safetensors": module_version("safetensors"),
            "optimum": module_version("optimum"),
            "nncf": module_version("nncf"),
        },
        "openvino_devices": ov.Core().available_devices,
        "artifacts": artifact_report(artifacts),
        "checks": {},
    }

    artifacts.require_files()
    tokenizer = AutoTokenizer.from_pretrained(artifacts.tokenizer_dir, trust_remote_code=True)
    input_ids, _ = encode_prompt_for_pages(tokenizer, args.prompt, page_count=args.page_count)
    prompt_len = int(input_ids.shape[1])
    validate_prompt_and_decode_lengths(artifacts, prompt_len, args.ring_window)
    check_logits_processor()

    report["checks"] = {
        "prompt_tokens": prompt_len,
        "page_count": args.page_count,
        "prefill_seq_len": prefill_seq_len(artifacts.prefill_model),
        "decode_prior_len": decode_prior_len(artifacts.decode_model),
        "ring_window": args.ring_window,
        "prompt_shape_matches": True,
        "logits_processor": "ok",
    }
    if args.run_one_token:
        report["one_token_runtime"] = runtime_one_token(args, artifacts, tokenizer)
    return report


def print_human(report: dict) -> None:
    print("Unlimited-OCR OpenVINO doctor")
    print(f"python: {report['python']}")
    print("packages:")
    for name, version in report["packages"].items():
        print(f"  {name}: {version}")
    print(f"openvino_devices: {', '.join(report['openvino_devices'])}")
    print("artifacts:")
    for name, item in report["artifacts"].items():
        size = "" if item["bin_mb"] is None else f" ({item['bin_mb']} MB bin)"
        print(f"  {name}: {'ok' if item['exists'] else 'missing'} {item['path']}{size}")
        if item["metadata"]:
            print(f"    metadata: {json.dumps(item['metadata'], ensure_ascii=False)}")
    print("checks:")
    for name, value in report["checks"].items():
        print(f"  {name}: {value}")
    if "one_token_runtime" in report:
        print("one_token_runtime:")
        for name, value in report["one_token_runtime"].items():
            print(f"  {name}: {value}")
    print("doctor: ok")


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
