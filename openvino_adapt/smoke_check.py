from __future__ import annotations

import argparse

import numpy as np

from .artifacts import ArtifactSet, decode_prior_len, prefill_seq_len, validate_prompt_and_decode_lengths
from .logits_processors import SlidingWindowNoRepeatNgram, select_greedy_token
from .run_prefill_no_cache import encode_prompt_for_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast non-inference smoke checks for the OpenVINO adapter.")
    parser.add_argument("--base-model-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument("--prefill-model", default="openvino_models/unlimited_ocr_kv_dense_prefill277/decoder_prefill_kv.xml")
    parser.add_argument("--decode-model", default="openvino_models/unlimited_ocr_kv_dense/decoder_decode_one.xml")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--page-count", type=int, default=1)
    parser.add_argument("--ring-window", type=int, default=128)
    return parser.parse_args()


def check_logits_processor() -> None:
    processor = SlidingWindowNoRepeatNgram(ngram_size=3, window=32)
    sequence = [1, 2, 3, 1, 2]
    scores = np.zeros(8, dtype=np.float32)
    token = select_greedy_token(scores, sequence, processor)
    if token == 3:
        raise AssertionError("no-repeat-ngram failed to ban repeated token")


def main() -> int:
    from transformers import AutoTokenizer

    args = parse_args()
    artifacts = ArtifactSet.from_paths(args.base_model_dir, args.prefill_model, args.decode_model, args.tokenizer)
    artifacts.require_files()
    tokenizer = AutoTokenizer.from_pretrained(artifacts.tokenizer_dir, trust_remote_code=True)
    input_ids, _ = encode_prompt_for_pages(tokenizer, args.prompt, page_count=args.page_count)
    validate_prompt_and_decode_lengths(artifacts, input_ids.shape[1], args.ring_window)
    check_logits_processor()
    print(f"prompt_tokens: {input_ids.shape[1]}")
    print(f"prefill_seq_len: {prefill_seq_len(artifacts.prefill_model)}")
    print(f"decode_prior_len: {decode_prior_len(artifacts.decode_model)}")
    print("smoke_check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
