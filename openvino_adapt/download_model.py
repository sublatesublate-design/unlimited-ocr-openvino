from __future__ import annotations

import argparse


DEFAULT_PATTERNS = [
    "config.json",
    "configuration_deepseek_v2.py",
    "conversation.py",
    "deepencoder.py",
    "modeling_deepseekv2.py",
    "modeling_unlimitedocr.py",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the minimal Unlimited-OCR HF snapshot for adaptation.")
    parser.add_argument("--model", default="baidu/Unlimited-OCR")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", default="", help="Optional explicit local directory. Empty means HF cache.")
    return parser.parse_args()


def main() -> int:
    from huggingface_hub import snapshot_download

    args = parse_args()
    kwargs = {
        "repo_id": args.model,
        "revision": args.revision,
        "allow_patterns": DEFAULT_PATTERNS,
    }
    if args.local_dir:
        kwargs["local_dir"] = args.local_dir
    path = snapshot_download(**kwargs)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
