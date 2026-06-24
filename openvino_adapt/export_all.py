from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .run_prefill_no_cache import encode_prompt_for_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the full Unlimited-OCR OpenVINO artifact set.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--page-count", type=int, default=1, help="Number of page images inserted at the single <image> slot.")
    parser.add_argument("--base-output-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument("--kv-output-prefix", default="openvino_models/unlimited_ocr_kv_dense")
    parser.add_argument("--ring-window", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--moe-impl", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--fp16", action="store_true", help="Save OpenVINO IR weights compressed to FP16.")
    parser.add_argument("--skip-base", action="store_true", help="Do not export projector/embed/vision graphs.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    input_ids, _ = encode_prompt_for_pages(tokenizer, args.prompt, page_count=args.page_count)
    seq_len = input_ids.shape[1]
    past_len = seq_len + args.ring_window - 1
    py = sys.executable

    if not args.skip_base:
        for component in ("projector", "embed_tokens", "vision_tokens"):
            cmd = [
                py,
                "-m",
                "openvino_adapt.export_openvino",
                "--model",
                args.model,
                "--component",
                component,
                "--allow-weight-download",
                "--output-dir",
                args.base_output_dir,
                "--device",
                args.device,
            ]
            if args.fp16:
                cmd.append("--fp16")
            run(cmd)

    prefill_dir = f"{args.kv_output_prefix}_prefill{seq_len}"
    decode_dir = f"{args.kv_output_prefix}_past{past_len}"
    prefill_cmd = [
        py,
        "-m",
        "openvino_adapt.export_openvino",
        "--model",
        args.model,
        "--component",
        "decoder_prefill_kv",
        "--seq-len",
        str(seq_len),
        "--allow-weight-download",
        "--output-dir",
        prefill_dir,
        "--device",
        args.device,
        "--moe-impl",
        args.moe_impl,
    ]
    decode_cmd = [
        py,
        "-m",
        "openvino_adapt.export_openvino",
        "--model",
        args.model,
        "--component",
        "decoder_decode_one",
        "--past-len",
        str(past_len),
        "--allow-weight-download",
        "--output-dir",
        decode_dir,
        "--device",
        args.device,
        "--moe-impl",
        args.moe_impl,
    ]
    if args.fp16:
        prefill_cmd.append("--fp16")
        decode_cmd.append("--fp16")
    run(prefill_cmd)
    run(decode_cmd)
    profile = {
        "model": args.model,
        "prompt": args.prompt,
        "page_count": args.page_count,
        "prompt_seq_len": seq_len,
        "decode_past_len": past_len,
        "ring_window": args.ring_window,
        "moe_impl": args.moe_impl,
        "compressed_fp16": bool(args.fp16),
        "base_output_dir": args.base_output_dir,
        "prefill_model": str(Path(prefill_dir, "decoder_prefill_kv.xml")),
        "decode_model": str(Path(decode_dir, "decoder_decode_one.xml")),
    }
    Path(prefill_dir).mkdir(parents=True, exist_ok=True)
    Path(decode_dir).mkdir(parents=True, exist_ok=True)
    Path(prefill_dir, "artifact_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(decode_dir, "artifact_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prompt_seq_len: {seq_len}")
    print(f"decode_past_len: {past_len}")
    print(f"base_output_dir: {Path(args.base_output_dir).resolve()}")
    print(f"prefill_model: {Path(prefill_dir, 'decoder_prefill_kv.xml').resolve()}")
    print(f"decode_model: {Path(decode_dir, 'decoder_decode_one.xml').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
