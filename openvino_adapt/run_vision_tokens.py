from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .preprocess import preprocess_base_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exported Unlimited-OCR vision_tokens OpenVINO IR.")
    parser.add_argument("--model", required=True, help="Path to vision_tokens.xml.")
    parser.add_argument("--image", required=True, help="Path to a page image.")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--save-npy", default="", help="Optional output .npy path for visual token embeddings.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(model_path, args.device)
    input_tensor = preprocess_base_image(args.image, image_size=args.image_size)
    result = compiled([input_tensor])
    output = next(iter(result.values()))
    print(f"output_shape: {tuple(output.shape)}")
    print(f"output_dtype: {output.dtype}")
    print(f"output_minmax: {float(np.min(output)):.6f}, {float(np.max(output)):.6f}")
    if args.save_npy:
        out_path = Path(args.save_npy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, output)
        print(f"saved_npy: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
