from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small Unlimited-OCR PyTorch baseline on one image.",
    )
    parser.add_argument("--image", required=True, help="Path to a single image.")
    parser.add_argument("--output-dir", default="research/baseline_outputs")
    parser.add_argument("--model", default="baidu/Unlimited-OCR")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=128)
    parser.add_argument(
        "--allow-weight-download",
        action="store_true",
        help="Required guard. Without this flag the script only validates arguments.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="CUDA is the official path; CPU may be very slow and can fail for this model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    if not args.allow_weight_download:
        print("Arguments look valid.")
        print("This script did not download weights.")
        print("Re-run with --allow-weight-download when you are ready for the large model download.")
        return 0

    import torch
    from transformers import AutoModel, AutoTokenizer

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = "cuda"
    else:
        device = "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    ).eval()
    model = model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_kwargs = dict(
        tokenizer=tokenizer,
        prompt="<image>document parsing.",
        image_file=str(image_path),
        output_path=str(output_dir),
        base_size=1024,
        image_size=args.image_size,
        crop_mode=False,
        max_length=args.max_length,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        ngram_window=args.ngram_window,
        save_results=True,
    )

    if device == "cpu":
        print("Warning: upstream infer() calls .cuda() in several places; CPU may fail without patching remote code.")

    with torch.inference_mode():
        model.infer(**infer_kwargs)

    print(f"Output written under: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
