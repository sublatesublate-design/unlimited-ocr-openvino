from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .run_prefill_no_cache import encode_prompt_for_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute prompt/page-count shapes for OpenVINO export.")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--page-count", type=int, default=1)
    parser.add_argument("--ring-window", type=int, default=128)
    parser.add_argument("--kv-output-prefix", default="openvino_models/unlimited_ocr_kv_dense")
    parser.add_argument("--moe-impl", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def build_profile(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    input_ids, image_slice = encode_prompt_for_pages(tokenizer, args.prompt, page_count=args.page_count)
    seq_len = int(input_ids.shape[1])
    past_len = seq_len + args.ring_window - 1
    prefill_model = Path(f"{args.kv_output_prefix}_prefill{seq_len}") / "decoder_prefill_kv.xml"
    decode_model = Path(f"{args.kv_output_prefix}_past{past_len}") / "decoder_decode_one.xml"
    export_cmd = [
        sys.executable,
        "-m",
        "openvino_adapt.export_all",
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--prompt",
        args.prompt,
        "--page-count",
        str(args.page_count),
        "--ring-window",
        str(args.ring_window),
        "--kv-output-prefix",
        args.kv_output_prefix,
        "--moe-impl",
        args.moe_impl,
        "--skip-base",
    ]
    if args.fp16:
        export_cmd.append("--fp16")
    return {
        "prompt": args.prompt,
        "page_count": args.page_count,
        "prompt_tokens": seq_len,
        "image_token_span": [image_slice.start, image_slice.stop],
        "ring_window": args.ring_window,
        "decode_prior_len": past_len,
        "prefill_model": str(prefill_model),
        "decode_model": str(decode_model),
        "export_command": export_cmd,
    }


def main() -> int:
    args = parse_args()
    profile = build_profile(args)
    if args.json:
        print(json.dumps(profile, ensure_ascii=False, indent=2))
        return 0

    print(f"prompt_tokens: {profile['prompt_tokens']}")
    print(f"page_count: {profile['page_count']}")
    print(f"image_token_span: {profile['image_token_span']}")
    print(f"decode_prior_len: {profile['decode_prior_len']}")
    print(f"prefill_model: {profile['prefill_model']}")
    print(f"decode_model: {profile['decode_model']}")
    print("export_command:")
    print(subprocess.list2cmdline(profile["export_command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
