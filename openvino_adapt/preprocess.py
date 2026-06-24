from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pad_to_square(image: Image.Image, image_size: int = 1024) -> Image.Image:
    fill = (127, 127, 127)
    return ImageOps.pad(image, (image_size, image_size), color=fill)


def normalize_image(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.transpose(arr, (2, 0, 1))


def preprocess_base_image(path: str | Path, image_size: int = 1024) -> np.ndarray:
    image = pad_to_square(load_rgb(path), image_size=image_size)
    return normalize_image(image)[None, ...]
