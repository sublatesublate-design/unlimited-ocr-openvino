from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .preprocess import preprocess_base_image


IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
VISION_TOKENS_PER_PAGE = 273


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenVINO no-cache prefill pass for one image.")
    parser.add_argument("--model-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--device", default="CPU")
    return parser.parse_args()


def encode_prompt_for_pages(tokenizer, prompt: str, page_count: int) -> tuple[np.ndarray, slice]:
    if page_count < 1:
        raise ValueError("page_count must be at least 1")
    if prompt.count(IMAGE_TOKEN) != 1:
        raise ValueError("prompt must contain exactly one <image> token")
    before, after = prompt.split(IMAGE_TOKEN)
    token_ids: list[int] = [tokenizer.bos_token_id]
    token_ids.extend(tokenizer.encode(before, add_special_tokens=False))
    image_start = len(token_ids)
    token_ids.extend([IMAGE_TOKEN_ID] * VISION_TOKENS_PER_PAGE * page_count)
    image_stop = len(token_ids)
    token_ids.extend(tokenizer.encode(after, add_special_tokens=False))
    return np.asarray([token_ids], dtype=np.int64), slice(image_start, image_stop)


def encode_prompt(tokenizer, prompt: str) -> tuple[np.ndarray, slice]:
    return encode_prompt_for_pages(tokenizer, prompt, page_count=1)


def main() -> int:
    import openvino as ov
    from transformers import AutoTokenizer

    args = parse_args()
    model_dir = Path(args.model_dir)
    core = ov.Core()
    embed = core.compile_model(model_dir / "embed_tokens.xml", args.device)
    vision = core.compile_model(model_dir / "vision_tokens.xml", args.device)
    decoder = core.compile_model(model_dir / "decoder_no_cache.xml", args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    input_ids, image_slice = encode_prompt(tokenizer, args.prompt)

    text_embeds = next(iter(embed([input_ids]).values()))
    image_input = preprocess_base_image(args.image)
    image_embeds = next(iter(vision([image_input]).values()))
    if image_embeds.shape[1] != image_slice.stop - image_slice.start:
        raise ValueError(f"visual token count mismatch: {image_embeds.shape[1]} vs {image_slice}")

    inputs_embeds = text_embeds.copy()
    inputs_embeds[:, image_slice, :] = image_embeds
    attention_mask = np.ones(input_ids.shape, dtype=np.int64)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)

    logits = next(iter(decoder([inputs_embeds, attention_mask, position_ids]).values()))
    next_token = int(np.argmax(logits[0, -1]))
    print(f"input_ids_shape: {input_ids.shape}")
    print(f"inputs_embeds_shape: {inputs_embeds.shape}")
    print(f"logits_shape: {logits.shape}")
    print(f"next_token_argmax: {next_token}")
    print(f"next_token_text: {tokenizer.decode([next_token])!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
