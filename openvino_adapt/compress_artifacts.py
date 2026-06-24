from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .artifacts import ArtifactSet, model_metadata


MODE_CHOICES = (
    "int8_asym",
    "int8_sym",
    "int4_asym",
    "int4_sym",
    "nf4",
    "cb4",
    "fp4",
    "mxfp4",
    "mxfp8_e4m3",
    "fp8_e4m3",
    "nvfp4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress OpenVINO IR weights with NNCF.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-model", default="", help="One OpenVINO .xml file to compress.")
    source.add_argument("--artifact-set", action="store_true", help="Compress embed, vision, prefill, and decode artifacts.")
    parser.add_argument("--base-model-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument("--prefill-model", default="openvino_models/unlimited_ocr_kv_dense_prefill277/decoder_prefill_kv.xml")
    parser.add_argument("--decode-model", default="openvino_models/unlimited_ocr_kv_dense/decoder_decode_one.xml")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    parser.add_argument("--output-model", default="", help="Output .xml path for --input-model.")
    parser.add_argument("--output-dir", default="", help="Output directory for --artifact-set.")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="int8_asym")
    parser.add_argument("--ratio", type=float, default=None, help="Compression ratio. Defaults to 1.0 for INT4, NNCF default for INT8.")
    parser.add_argument("--group-size", type=int, default=None, help="Group size. Defaults to 64 for INT4, NNCF default for INT8.")
    parser.add_argument("--all-layers", action="store_true", help="Ask NNCF to compress all supported layers.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def nncf_mode(mode: str):
    from nncf import CompressWeightsMode

    enum_name = mode.upper()
    if not hasattr(CompressWeightsMode, enum_name):
        available = ", ".join(sorted(item.name.lower() for item in CompressWeightsMode))
        raise ValueError(f"NNCF mode {mode!r} is unavailable in this environment. Available modes: {available}")
    return getattr(CompressWeightsMode, enum_name)


def compressed_path(input_model: Path, output_root: Path, mode: str) -> Path:
    return output_root / input_model.parent.name / f"{input_model.stem}_{mode}.xml"


def compression_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {"mode": nncf_mode(args.mode)}
    if args.mode.startswith("int4"):
        kwargs["ratio"] = 1.0 if args.ratio is None else args.ratio
        kwargs["group_size"] = 64 if args.group_size is None else args.group_size
    else:
        if args.ratio is not None:
            kwargs["ratio"] = args.ratio
        if args.group_size is not None:
            kwargs["group_size"] = args.group_size
    if args.all_layers:
        kwargs["all_layers"] = True
    return kwargs


def copy_sidecars(input_model: Path, output_model: Path, extra_metadata: dict) -> None:
    input_json = input_model.with_suffix(".json")
    metadata = model_metadata(input_model)
    metadata.update(extra_metadata)
    output_model.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    input_profile = input_model.parent / "artifact_profile.json"
    if input_profile.exists():
        profile = json.loads(input_profile.read_text(encoding="utf-8"))
        profile.update(extra_metadata)
        profile_name = f"{output_model.stem}_artifact_profile.json"
        (output_model.parent / profile_name).write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    elif input_json.exists():
        shutil.copy2(input_json, output_model.with_suffix(".source.json"))


def compress_one(input_model: Path, output_model: Path, args: argparse.Namespace) -> dict:
    if not input_model.exists():
        raise FileNotFoundError(input_model)
    if input_model.suffix.lower() != ".xml":
        raise ValueError(f"Expected an OpenVINO .xml model, got {input_model}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_model": str(input_model),
        "output_model": str(output_model),
        "mode": args.mode,
        "ratio": args.ratio,
        "group_size": args.group_size,
        "all_layers": bool(args.all_layers),
    }
    if args.dry_run:
        return payload

    import openvino as ov
    import nncf

    core = ov.Core()
    model = core.read_model(input_model)
    kwargs = compression_kwargs(args)
    compressed = nncf.compress_weights(model, **kwargs)
    ov.save_model(compressed, output_model, compress_to_fp16=False)
    copy_sidecars(
        input_model,
        output_model,
        {
            "weight_compressed": True,
            "weight_compression_mode": args.mode,
            "weight_compression_ratio": kwargs.get("ratio"),
            "weight_compression_group_size": kwargs.get("group_size"),
        },
    )
    payload["input_bin_mb"] = round(input_model.with_suffix(".bin").stat().st_size / 1024 / 1024, 2)
    payload["output_bin_mb"] = round(output_model.with_suffix(".bin").stat().st_size / 1024 / 1024, 2)
    return payload


def artifact_inputs(args: argparse.Namespace) -> list[Path]:
    artifacts = ArtifactSet.from_paths(args.base_model_dir, args.prefill_model, args.decode_model, args.tokenizer)
    artifacts.require_files()
    return [artifacts.embed_tokens, artifacts.vision_tokens, artifacts.prefill_model, artifacts.decode_model]


def main() -> int:
    args = parse_args()
    if args.artifact_set:
        output_root = Path(args.output_dir or f"openvino_models/compressed_{args.mode}")
        pairs = [(path, compressed_path(path, output_root, args.mode)) for path in artifact_inputs(args)]
    else:
        input_model = Path(args.input_model)
        output_model = Path(args.output_model) if args.output_model else input_model.with_name(f"{input_model.stem}_{args.mode}.xml")
        pairs = [(input_model, output_model)]

    results = [compress_one(src, dst, args) for src, dst in pairs]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
