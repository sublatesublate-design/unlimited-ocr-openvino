from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import ArtifactSet, decode_prior_len, model_metadata, prefill_seq_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Unlimited-OCR OpenVINO adaptation artifacts.")
    parser.add_argument("--base-model-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument("--prefill-model", default="openvino_models/unlimited_ocr_kv_dense_prefill277/decoder_prefill_kv.xml")
    parser.add_argument("--decode-model", default="openvino_models/unlimited_ocr_kv_dense/decoder_decode_one.xml")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = ArtifactSet.from_paths(args.base_model_dir, args.prefill_model, args.decode_model, args.tokenizer)
    artifacts.require_files()
    for label, path in [
        ("embed_tokens", artifacts.embed_tokens),
        ("vision_tokens", artifacts.vision_tokens),
        ("prefill_model", artifacts.prefill_model),
        ("decode_model", artifacts.decode_model),
    ]:
        size_mb = (Path(path).with_suffix(".bin").stat().st_size / (1024 * 1024)) if Path(path).with_suffix(".bin").exists() else 0
        print(f"{label}: {path} ({size_mb:.2f} MB bin)")
        meta = model_metadata(Path(path))
        if meta:
            print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"prefill_seq_len: {prefill_seq_len(artifacts.prefill_model)}")
    print(f"decode_prior_len: {decode_prior_len(artifacts.decode_model)}")
    print(f"tokenizer: {artifacts.tokenizer_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
