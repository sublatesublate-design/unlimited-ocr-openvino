from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .wrappers import (
    DecoderDecodeOneStep,
    DecoderNoCache,
    DecoderPrefillWithKV,
    ProjectorOnly,
    TextEmbeddings,
    VisionTokenExtractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Unlimited-OCR components to OpenVINO IR.")
    parser.add_argument("--model", default="baidu/Unlimited-OCR", help="HF model ID or local model directory.")
    parser.add_argument("--output-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument(
        "--component",
        choices=(
            "projector",
            "embed_tokens",
            "vision_tokens",
            "decoder_no_cache",
            "decoder_prefill_kv",
            "decoder_decode_one",
        ),
        default="vision_tokens",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128, help="Example sequence length for decoder_no_cache.")
    parser.add_argument("--past-len", type=int, default=404, help="Past-cache length for decoder_decode_one.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--fp16", action="store_true", help="Ask OpenVINO to save compressed FP16 IR.")
    parser.add_argument(
        "--moe-impl",
        choices=("dense", "sparse"),
        default="dense",
        help="MoE export implementation for explicit-KV decoder graphs. Sparse is experimental on CPU.",
    )
    parser.add_argument(
        "--allow-weight-download",
        action="store_true",
        help="Required guard. Without it this command only prints the planned export.",
    )
    return parser.parse_args()


def planned_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / f"{args.component}.xml"


def load_model(model_id: str, device: str) -> torch.nn.Module:
    from transformers import AutoModel

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    return model.eval().to(device)


def build_component(model: torch.nn.Module, args: argparse.Namespace) -> tuple[torch.nn.Module, object]:
    device = torch.device(args.device)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model.config._openvino_moe_impl = args.moe_impl

    if args.component == "projector":
        wrapper = ProjectorOnly(model.model.projector).eval().to(device)
        example = torch.zeros((1, 4096, 2048), dtype=dtype, device=device)
        return wrapper, example

    if args.component == "embed_tokens":
        wrapper = TextEmbeddings(model.model.embed_tokens).eval().to(device)
        example = torch.zeros((1, args.seq_len), dtype=torch.long, device=device)
        return wrapper, example

    if args.component == "vision_tokens":
        wrapper = VisionTokenExtractor(model.model).eval().to(device)
        example = torch.zeros((1, 3, args.image_size, args.image_size), dtype=dtype, device=device)
        return wrapper, example

    hidden = model.config.hidden_size
    if args.component == "decoder_no_cache":
        wrapper = DecoderNoCache(model).eval().to(device)
        example = {
            "inputs_embeds": torch.zeros((1, args.seq_len, hidden), dtype=dtype, device=device),
            "attention_mask": torch.ones((1, args.seq_len), dtype=torch.long, device=device),
            "position_ids": torch.arange(args.seq_len, dtype=torch.long, device=device).unsqueeze(0),
        }
        return wrapper, example

    if args.component == "decoder_prefill_kv":
        wrapper = DecoderPrefillWithKV(model).eval().to(device)
        mask = torch.zeros((1, 1, args.seq_len, args.seq_len), dtype=dtype, device=device)
        example = {
            "inputs_embeds": torch.zeros((1, args.seq_len, hidden), dtype=dtype, device=device),
            "attention_mask": mask,
            "position_ids": torch.arange(args.seq_len, dtype=torch.long, device=device).unsqueeze(0),
        }
        return wrapper, example

    wrapper = DecoderDecodeOneStep(model).eval().to(device)
    past = []
    for _ in range(model.config.num_hidden_layers):
        past.append(torch.zeros((1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim), dtype=dtype, device=device))
        past.append(torch.zeros((1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim), dtype=dtype, device=device))
    example = (
        torch.zeros((1, 1), dtype=torch.long, device=device),
        torch.full((1, 1), args.past_len, dtype=torch.long, device=device),
        torch.zeros((1, 1, 1, args.past_len + 1), dtype=dtype, device=device),
        *past,
    )
    return wrapper, example


def save_openvino(wrapper: torch.nn.Module, example_input: object, output_path: Path, fp16: bool) -> None:
    import openvino as ov

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        ov_model = ov.convert_model(wrapper, example_input=example_input)
    ov.save_model(ov_model, output_path, compress_to_fp16=fp16)


def save_metadata(args: argparse.Namespace, output_path: Path) -> None:
    metadata = {
        "component": args.component,
        "model": args.model,
        "image_size": args.image_size,
        "compressed_fp16": bool(args.fp16),
        "seq_len": args.seq_len if args.component in {"decoder_no_cache", "decoder_prefill_kv"} else None,
        "past_len": args.past_len if args.component == "decoder_decode_one" else None,
        "ring_window": 128 if args.component == "decoder_decode_one" else None,
        "moe_impl": args.moe_impl if args.component in {"decoder_prefill_kv", "decoder_decode_one"} else None,
        "dense_moe": (args.moe_impl == "dense") if args.component in {"decoder_prefill_kv", "decoder_decode_one"} else None,
        "sparse_topk_moe": (args.moe_impl == "sparse") if args.component in {"decoder_prefill_kv", "decoder_decode_one"} else None,
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_path = planned_path(args)

    if not args.allow_weight_download:
        print("Dry run only; no weights loaded or downloaded.")
        print(f"component: {args.component}")
        print(f"model: {args.model}")
        print(f"output: {output_path}")
        print("Re-run with --allow-weight-download when ready.")
        return 0

    model = load_model(args.model, args.device)
    wrapper, example_input = build_component(model, args)
    save_openvino(wrapper, example_input, output_path, args.fp16)
    save_metadata(args, output_path)
    print(f"saved: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
