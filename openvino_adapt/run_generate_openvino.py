from __future__ import annotations

import argparse
import time
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from .artifacts import ArtifactSet, validate_prompt_and_decode_lengths
from .logits_processors import SlidingWindowNoRepeatNgram, select_greedy_token
from .preprocess import preprocess_base_image
from .runtime import add_runtime_args, compile_artifacts, component_devices, make_core
from .run_prefill_no_cache import encode_prompt_for_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy Unlimited-OCR generation with OpenVINO IRs.")
    parser.add_argument("--base-model-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument("--prefill-model", default="openvino_models/unlimited_ocr_kv_dense_prefill277/decoder_prefill_kv.xml")
    parser.add_argument("--decode-model", default="openvino_models/unlimited_ocr_kv_dense/decoder_decode_one.xml")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--ring-window", type=int, default=128)
    parser.add_argument("--eos-token-id", type=int, default=1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=128)
    add_runtime_args(parser)
    return parser.parse_args()


def causal_mask(seq_len: int) -> np.ndarray:
    return np.triu(
        np.full((1, 1, seq_len, seq_len), np.finfo(np.float32).min, dtype=np.float32),
        k=1,
    )


def decode_mask(past_len: int, active_past_len: int) -> np.ndarray:
    mask = np.zeros((1, 1, 1, past_len + 1), dtype=np.float32)
    if active_past_len < past_len:
        mask[:, :, :, active_past_len:past_len] = np.finfo(np.float32).min
    return mask


def main() -> int:
    from transformers import AutoTokenizer

    args = parse_args()
    artifacts = ArtifactSet.from_paths(args.base_model_dir, args.prefill_model, args.decode_model, args.tokenizer)
    artifacts.require_files()
    tokenizer = AutoTokenizer.from_pretrained(artifacts.tokenizer_dir, trust_remote_code=True)
    input_ids, image_slice = encode_prompt_for_pages(tokenizer, args.prompt, page_count=1)
    validate_prompt_and_decode_lengths(artifacts, input_ids.shape[1], args.ring_window)

    devices = component_devices(args)
    core = make_core(args.cache_dir)
    compiled = compile_artifacts(core, artifacts, devices)
    result = generate_from_compiled(
        embed=compiled.embed,
        vision=compiled.vision,
        prefill=compiled.prefill,
        decode=compiled.decode,
        tokenizer=tokenizer,
        image=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        ring_window=args.ring_window,
        eos_token_id=args.eos_token_id,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        ngram_window=args.ngram_window,
    )
    print(f"prompt_tokens: {result['prompt_tokens']}")
    print(f"devices: {devices.as_dict()}")
    print(f"generated_ids: {result['generated_ids']}")
    print(f"generated_text: {result['text']!r}")
    print(f"decode_seconds: {result['decode_seconds']:.2f}")
    print(f"tokens_per_second: {result['tokens_per_second']:.3f}")
    return 0


def generate_from_compiled(
    *,
    embed,
    vision,
    prefill,
    decode,
    sparse_runtime=None,
    sparse_metadata: dict | None = None,
    sparse_layers: list[int] | None = None,
    tokenizer,
    image: str | Path | None = None,
    images: Sequence[str | Path] | None = None,
    prompt: str,
    max_new_tokens: int,
    ring_window: int = 128,
    eos_token_id: int = 1,
    no_repeat_ngram_size: int = 35,
    ngram_window: int = 128,
) -> dict:
    started = time.time()
    if images is None:
        if image is None:
            raise ValueError("Either image or images must be provided.")
        images = [image]
    if not images:
        raise ValueError("images must contain at least one page image.")

    input_ids, image_slice = encode_prompt_for_pages(tokenizer, prompt, page_count=len(images))
    prompt_sequence = input_ids[0].tolist()
    seq_len = input_ids.shape[1]
    text_embeds = next(iter(embed([input_ids]).values()))
    image_inputs = np.concatenate([preprocess_base_image(page) for page in images], axis=0)
    image_embeds = next(iter(vision([image_inputs]).values())).reshape(1, -1, text_embeds.shape[-1])
    if image_embeds.shape[1] != image_slice.stop - image_slice.start:
        raise ValueError(f"visual token count mismatch: {image_embeds.shape[1]} vs {image_slice}")
    inputs_embeds = text_embeds.copy()
    inputs_embeds[:, image_slice, :] = image_embeds

    prefill_inputs = [
        inputs_embeds,
        causal_mask(seq_len),
        np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    ]
    prefill_outs = list(prefill(prefill_inputs).values())
    logits = prefill_outs[0]
    prefill_keys = prefill_outs[1::2]
    prefill_values = prefill_outs[2::2]

    generated: list[int] = []
    processor = SlidingWindowNoRepeatNgram(no_repeat_ngram_size, ngram_window)
    current_token = select_greedy_token(logits[0, -1], prompt_sequence, processor)
    generated.append(current_token)

    prefill_len = prefill_keys[0].shape[2]
    prior_capacity = prefill_len + ring_window - 1
    if sparse_runtime is not None:
        if sparse_metadata is None:
            raise ValueError("sparse_metadata is required when sparse_runtime is used.")
        sparse_past_len = int(sparse_metadata["past_len"])
        if sparse_past_len != prior_capacity:
            raise ValueError(
                f"Sparse decode artifact expects past_len={sparse_past_len}, but prompt_len={prefill_len} "
                f"and ring_window={ring_window} need {prior_capacity}."
            )
        sparse_layers = sparse_layers or [int(layer_id) for layer_id in sparse_metadata["layers"].keys()]
        sparse_step_timings: list[dict] = []
    else:
        sparse_step_timings = []
    keys = [np.zeros((1, 10, prior_capacity, 128), dtype=np.float32) for _ in range(12)]
    values = [np.zeros((1, 10, prior_capacity, 128), dtype=np.float32) for _ in range(12)]
    for layer in range(12):
        keys[layer][:, :, :prefill_len, :] = prefill_keys[layer]
        values[layer][:, :, :prefill_len, :] = prefill_values[layer]

    stored_generated = 0
    while len(generated) < max_new_tokens and current_token != eos_token_id:
        active_past = prefill_len + min(stored_generated, ring_window - 1)
        token_ids = np.asarray([[current_token]], dtype=np.int64)
        position_ids = np.asarray([[prefill_len + stored_generated]], dtype=np.int64)
        attention_mask = decode_mask(prior_capacity, active_past)
        if sparse_runtime is None:
            feeds: list[np.ndarray] = [token_ids, position_ids, attention_mask]
            for layer in range(12):
                feeds.extend([keys[layer], values[layer]])

            decode_outs = list(decode(feeds).values())
            logits = decode_outs[0]
            new_keys = decode_outs[1::2]
            new_values = decode_outs[2::2]
        else:
            from .run_sparse_decode_openvino import run_sparse_decode

            hidden = next(iter(embed([token_ids]).values()))
            logits, new_keys, new_values, timings = run_sparse_decode(
                sparse_runtime,
                sparse_metadata,
                hidden.astype(np.float32),
                position_ids,
                attention_mask,
                keys,
                values,
                sparse_layers,
                compile_all_experts=False,
            )
            sparse_step_timings.append(timings)

        if stored_generated < ring_window - 1:
            slot = prefill_len + stored_generated
        else:
            keys = [
                np.concatenate(
                    [k[:, :, :prefill_len, :], k[:, :, prefill_len + 1 :, :], np.zeros_like(k[:, :, -1:, :])],
                    axis=2,
                )
                for k in keys
            ]
            values = [
                np.concatenate(
                    [v[:, :, :prefill_len, :], v[:, :, prefill_len + 1 :, :], np.zeros_like(v[:, :, -1:, :])],
                    axis=2,
                )
                for v in values
            ]
            slot = prior_capacity - 1

        for layer in range(12):
            keys[layer][:, :, slot : slot + 1, :] = new_keys[layer]
            values[layer][:, :, slot : slot + 1, :] = new_values[layer]

        stored_generated += 1
        current_token = select_greedy_token(logits[0, -1], prompt_sequence + generated, processor)
        generated.append(current_token)

    text = tokenizer.decode(generated, skip_special_tokens=False)
    elapsed = time.time() - started
    return {
        "prompt_tokens": seq_len,
        "page_count": len(images),
        "generated_ids": generated,
        "text": text,
        "decode_seconds": elapsed,
        "tokens_per_second": (len(generated) / elapsed) if elapsed > 0 else 0.0,
        "decoder": "sparse" if sparse_runtime is not None else "dense",
        "sparse_step_timings": sparse_step_timings,
    }


if __name__ == "__main__":
    raise SystemExit(main())
