from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .preprocess import preprocess_base_image
from .wrappers import VisionTokenExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and OpenVINO vision_tokens outputs.")
    parser.add_argument("--hf-model", default="models/Unlimited-OCR")
    parser.add_argument("--ov-model", default="openvino_models/unlimited_ocr/vision_tokens.xml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--image-size", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    from transformers import AutoModel
    import openvino as ov

    args = parse_args()
    x_np = preprocess_base_image(args.image, image_size=args.image_size)

    model = AutoModel.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
    ).eval()
    wrapper = VisionTokenExtractor(model.model).eval()
    with torch.inference_mode():
        torch_out = wrapper(torch.from_numpy(x_np)).numpy()

    core = ov.Core()
    compiled = core.compile_model(Path(args.ov_model), args.device)
    ov_out = next(iter(compiled([x_np]).values()))

    diff = np.abs(torch_out - ov_out)
    print(f"torch_shape: {torch_out.shape}")
    print(f"openvino_shape: {ov_out.shape}")
    print(f"max_abs_diff: {float(diff.max()):.8f}")
    print(f"mean_abs_diff: {float(diff.mean()):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
