from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .artifacts import ArtifactSet, validate_prompt_and_decode_lengths
from .runtime import add_runtime_args, compile_artifacts, component_devices, make_core, parse_ov_config
from .run_generate_openvino import generate_from_compiled
from .run_prefill_no_cache import encode_prompt_for_pages


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Unlimited-OCR OpenVINO adaptation on images or a PDF.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", default="", help="Single image path.")
    source.add_argument("--image-dir", default="", help="Directory of page images.")
    source.add_argument("--pdf", default="", help="PDF path; pages are rendered to images first.")
    parser.add_argument("--output-dir", default="outputs_openvino")
    parser.add_argument("--base-model-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument("--prefill-model", default="openvino_models/unlimited_ocr_kv_dense_prefill277/decoder_prefill_kv.xml")
    parser.add_argument("--decode-model", default="openvino_models/unlimited_ocr_kv_dense/decoder_decode_one.xml")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    parser.add_argument("--decoder", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--sparse-artifact-dir", default="openvino_models/sparse_decode_past677")
    parser.add_argument("--sparse-device", default="", help="OpenVINO device for sparse layer graphs. Defaults to --decode-device/--device.")
    parser.add_argument("--sparse-expert-device", default="", help="OpenVINO device for sparse expert graphs. Defaults to --sparse-device.")
    parser.add_argument("--sparse-hot-pack-dir", default="", help="Optional root/single-layer dir containing hot_expert_pack.xml.")
    parser.add_argument("--sparse-hot-pack-device", default="", help="OpenVINO device for hot expert pack graphs.")
    parser.add_argument("--sparse-precompile-static", action="store_true", help="Compile sparse layer/add/final/hot-pack graphs before generation.")
    parser.add_argument("--sparse-precompile-all-experts", action="store_true", help="Also compile all fallback expert graphs before generation.")
    parser.add_argument("--sparse-final-argmax", action="store_true", help="Use final_norm_argmax.xml for sparse greedy generation when present.")
    parser.add_argument("--sparse-final-topk", type=int, default=0, help="Use final_norm_topkK.xml for sparse greedy generation when present.")
    parser.add_argument("--sparse-config", nargs="*", default=[], help="Extra sparse OpenVINO compile config as KEY=VALUE pairs.")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ring-window", type=int, default=128)
    parser.add_argument("--eos-token-id", type=int, default=1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=128)
    parser.add_argument("--pdf-dpi", type=int, default=200)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run all pages in one model context. Requires a prefill graph exported for this page count.",
    )
    add_runtime_args(parser)
    return parser.parse_args()


def collect_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def pdf_to_images(pdf_path: str | Path, dpi: int) -> tuple[tempfile.TemporaryDirectory, list[Path]]:
    import fitz

    tmp = tempfile.TemporaryDirectory(prefix="unlimited_ocr_pdf_")
    out_dir = Path(tmp.name)
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths: list[Path] = []
    for index, page in enumerate(doc, 1):
        out = out_dir / f"page_{index:04d}.png"
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return tmp, paths


def source_images(args: argparse.Namespace) -> tuple[tempfile.TemporaryDirectory | None, list[Path]]:
    if args.image:
        return None, [Path(args.image)]
    if args.image_dir:
        return None, collect_images(args.image_dir)
    return pdf_to_images(args.pdf, args.pdf_dpi)


def split_continuous_pages(text: str, page_count: int) -> list[str]:
    if "<PAGE>" not in text:
        return [text]
    pages = [part.strip() for part in text.split("<PAGE>")[1:]]
    return pages[:page_count] if pages else [text]


def main() -> int:
    from transformers import AutoTokenizer

    args = parse_args()
    artifacts = ArtifactSet.from_paths(args.base_model_dir, args.prefill_model, args.decode_model, args.tokenizer)
    if args.decoder == "dense":
        artifacts.require_files()
    else:
        missing = [
            path
            for path in [
                artifacts.embed_tokens,
                artifacts.vision_tokens,
                artifacts.prefill_model,
                artifacts.tokenizer_dir / "tokenizer.json",
                Path(args.sparse_artifact_dir) / "metadata.json",
            ]
            if not path.exists()
        ]
        if missing:
            joined = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Missing OpenVINO sparse OCR artifacts:\n{joined}")

    tokenizer = AutoTokenizer.from_pretrained(artifacts.tokenizer_dir, trust_remote_code=True)

    tmp, images = source_images(args)
    if not images:
        raise FileNotFoundError("No images found for OCR.")
    prompt_pages = len(images) if args.continuous else 1
    input_ids, _ = encode_prompt_for_pages(tokenizer, args.prompt, page_count=prompt_pages)
    if args.decoder == "dense":
        validate_prompt_and_decode_lengths(artifacts, input_ids.shape[1], args.ring_window)
    else:
        from .artifacts import prefill_seq_len

        expected_prefill = prefill_seq_len(artifacts.prefill_model)
        if expected_prefill is not None and expected_prefill != input_ids.shape[1]:
            raise ValueError(
                f"Prefill graph expects sequence length {expected_prefill}, but this prompt needs {input_ids.shape[1]}."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    devices = component_devices(args)
    core = make_core(args.cache_dir)
    if args.decoder == "dense":
        compiled = compile_artifacts(core, artifacts, devices)
        sparse_runtime = None
        sparse_metadata = None
    else:
        from .runtime import CompiledArtifacts
        from .run_sparse_decode_openvino import SparseDecodeRuntime

        sparse_dir = Path(args.sparse_artifact_dir)
        sparse_metadata = json.loads((sparse_dir / "metadata.json").read_text(encoding="utf-8"))
        needed_prior = input_ids.shape[1] + args.ring_window - 1
        if int(sparse_metadata["past_len"]) != needed_prior:
            raise ValueError(
                f"Sparse artifact expects past_len={sparse_metadata['past_len']}, but prompt_len={input_ids.shape[1]} "
                f"and ring_window={args.ring_window} need {needed_prior}."
            )
        compiled = CompiledArtifacts(
            embed=core.compile_model(artifacts.embed_tokens, devices.embed),
            vision=core.compile_model(artifacts.vision_tokens, devices.vision),
            prefill=core.compile_model(artifacts.prefill_model, devices.prefill),
            decode=None,
            devices=devices,
        )
        sparse_device = args.sparse_device or devices.decode
        sparse_runtime = SparseDecodeRuntime(
            sparse_dir,
            sparse_device,
            args.sparse_expert_device or sparse_device,
            args.cache_dir,
            args.sparse_hot_pack_dir,
            args.sparse_hot_pack_device or args.sparse_expert_device or sparse_device,
            parse_ov_config(args.sparse_config),
            args.sparse_final_topk,
            args.sparse_final_argmax,
        )
        sparse_precompile = {}
        if args.sparse_precompile_static:
            layers = [int(layer_id) for layer_id in sparse_metadata["layers"].keys()]
            sparse_precompile = sparse_runtime.precompile_static(
                sparse_metadata,
                layers,
                args.sparse_precompile_all_experts,
            )

    manifest = {
        "source": args.image or args.image_dir or args.pdf,
        "pages": [],
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "ngram_window": args.ngram_window,
        "device": args.device,
        "devices": devices.as_dict(),
        "cache_dir": args.cache_dir,
        "mode": "continuous" if args.continuous else "per_page",
        "decoder": args.decoder,
        "sparse_artifact_dir": args.sparse_artifact_dir if args.decoder == "sparse" else "",
        "sparse_device": args.sparse_device or devices.decode if args.decoder == "sparse" else "",
        "sparse_expert_device": args.sparse_expert_device or args.sparse_device or devices.decode if args.decoder == "sparse" else "",
        "sparse_hot_pack_dir": args.sparse_hot_pack_dir if args.decoder == "sparse" else "",
        "sparse_hot_pack_device": args.sparse_hot_pack_device or args.sparse_expert_device or args.sparse_device or devices.decode if args.decoder == "sparse" else "",
        "sparse_config": parse_ov_config(args.sparse_config) if args.decoder == "sparse" else {},
        "sparse_final_argmax": args.sparse_final_argmax if args.decoder == "sparse" else False,
        "sparse_final_topk": args.sparse_final_topk if args.decoder == "sparse" else 0,
        "sparse_precompile": sparse_precompile if args.decoder == "sparse" else {},
    }
    combined_parts: list[str] = []
    try:
        if args.continuous:
            print(f"[continuous] {len(images)} pages")
            result = generate_from_compiled(
                embed=compiled.embed,
                vision=compiled.vision,
                prefill=compiled.prefill,
                decode=compiled.decode,
                sparse_runtime=sparse_runtime,
                sparse_metadata=sparse_metadata,
                tokenizer=tokenizer,
                images=images,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                ring_window=args.ring_window,
                eos_token_id=args.eos_token_id,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                ngram_window=args.ngram_window,
            )
            raw_output_name = "continuous.md"
            (output_dir / raw_output_name).write_text(result["text"], encoding="utf-8")
            page_texts = split_continuous_pages(result["text"], len(images))
            unsegmented_output = len(page_texts) == 1 and len(images) > 1
            for page_index, image in enumerate(images, 1):
                text = page_texts[page_index - 1] if page_index <= len(page_texts) else ""
                page_name = f"page_{page_index:04d}.md"
                (output_dir / page_name).write_text(text, encoding="utf-8")
                if not unsegmented_output:
                    combined_parts.append(f"<!-- page {page_index}: {image.name} -->\n{text}\n")
                manifest["pages"].append({"page": page_index, "image": os.fspath(image), "output": page_name})
            if unsegmented_output:
                combined_parts.append(result["text"])
            manifest["continuous_result"] = {
                "prompt_tokens": result["prompt_tokens"],
                "generated_tokens": len(result["generated_ids"]),
                "decode_seconds": result["decode_seconds"],
                "tokens_per_second": result["tokens_per_second"],
                "segmented_pages": not unsegmented_output,
                "raw_output": raw_output_name,
            }
        else:
            for page_index, image in enumerate(images, 1):
                print(f"[{page_index}/{len(images)}] {image}")
                result = generate_from_compiled(
                    embed=compiled.embed,
                    vision=compiled.vision,
                    prefill=compiled.prefill,
                    decode=compiled.decode,
                    sparse_runtime=sparse_runtime,
                    sparse_metadata=sparse_metadata,
                    tokenizer=tokenizer,
                    image=image,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    ring_window=args.ring_window,
                    eos_token_id=args.eos_token_id,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    ngram_window=args.ngram_window,
                )
                text = result["text"]
                page_name = f"page_{page_index:04d}.md"
                (output_dir / page_name).write_text(text, encoding="utf-8")
                combined_parts.append(f"<!-- page {page_index}: {image.name} -->\n{text}\n")
                manifest["pages"].append(
                    {
                        "page": page_index,
                        "image": os.fspath(image),
                        "output": page_name,
                        "prompt_tokens": result["prompt_tokens"],
                        "generated_tokens": len(result["generated_ids"]),
                        "decode_seconds": result["decode_seconds"],
                        "tokens_per_second": result["tokens_per_second"],
                    }
                )
        (output_dir / "combined.md").write_text("\n".join(combined_parts), encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if tmp is not None:
            tmp.cleanup()

    print(f"wrote: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
