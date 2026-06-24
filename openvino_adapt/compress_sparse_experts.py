from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import openvino as ov

from .compress_artifacts import compression_kwargs, nncf_mode


SUPPORTED_COPY_MODES = {"fp16", "fp32", "copy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a per-expert precision plan to split sparse decode artifacts.")
    parser.add_argument("--artifact-dir", default="openvino_models/sparse_decode_past677")
    parser.add_argument("--plan-json", default="outputs_openvino_ngram_smoke/expert_precision_plan.json")
    parser.add_argument("--output-dir", default="openvino_models/sparse_decode_past677_mixed")
    parser.add_argument("--default-mode", default="fp16", help="Mode for experts missing from the plan.")
    parser.add_argument("--unsupported-action", choices=("copy", "skip", "fail"), default="copy")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def copy_model(src_xml: Path, dst_xml: Path) -> None:
    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_xml, dst_xml)
    src_bin = src_xml.with_suffix(".bin")
    if src_bin.exists():
        shutil.copy2(src_bin, dst_xml.with_suffix(".bin"))


def copy_tree_without_experts(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if "experts" in rel.parts:
            continue
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def mode_to_args(mode: str, group_size: int | None) -> argparse.Namespace:
    effective_group_size = group_size if mode.startswith("int4") else None
    return argparse.Namespace(mode=mode, ratio=None, group_size=effective_group_size, all_layers=False)


def compress_one(core: ov.Core, src_xml: Path, dst_xml: Path, mode: str, group_size: int, dry_run: bool) -> dict:
    result = {
        "source": str(src_xml),
        "target": str(dst_xml),
        "requested_mode": mode,
        "actual_mode": mode,
    }
    if dry_run:
        return result

    import nncf

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    model = core.read_model(src_xml)
    args = mode_to_args(mode, group_size)
    kwargs = compression_kwargs(args)
    compressed = nncf.compress_weights(model, **kwargs)
    ov.save_model(compressed, dst_xml, compress_to_fp16=False)
    result["group_size"] = kwargs.get("group_size")
    result["ratio"] = kwargs.get("ratio")
    result["source_bin_mb"] = round(src_xml.with_suffix(".bin").stat().st_size / 1024 / 1024, 3)
    result["target_bin_mb"] = round(dst_xml.with_suffix(".bin").stat().st_size / 1024 / 1024, 3)
    return result


def main() -> int:
    args = parse_args()
    src_root = Path(args.artifact_dir)
    dst_root = Path(args.output_dir)
    plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))["plan"]

    if not src_root.exists():
        raise FileNotFoundError(src_root)

    if not args.dry_run:
        if dst_root.exists():
            raise FileExistsError(f"Refusing to overwrite existing output dir: {dst_root}")
        copy_tree_without_experts(src_root, dst_root)

    core = ov.Core()
    results = []
    for layer_dir in sorted(src_root.glob("layer_*")):
        experts_dir = layer_dir / "experts"
        if not experts_dir.exists():
            continue
        layer_id = str(int(layer_dir.name.split("_")[1]))
        expert_plan = plan.get(layer_id, {}).get("experts", {})
        for src_xml in sorted(experts_dir.glob("expert_*.xml")):
            expert_id = str(int(src_xml.stem.split("_")[1]))
            requested_mode = expert_plan.get(expert_id, {}).get("mode", args.default_mode)
            dst_xml = dst_root / layer_dir.name / "experts" / src_xml.name

            if requested_mode in SUPPORTED_COPY_MODES:
                result = {
                    "source": str(src_xml),
                    "target": str(dst_xml),
                    "requested_mode": requested_mode,
                    "actual_mode": "copy",
                }
                if not args.dry_run:
                    copy_model(src_xml, dst_xml)
            else:
                try:
                    nncf_mode(requested_mode)
                except ValueError:
                    if args.unsupported_action == "fail":
                        raise
                    result = {
                        "source": str(src_xml),
                        "target": str(dst_xml),
                        "requested_mode": requested_mode,
                        "actual_mode": "copy" if args.unsupported_action == "copy" else "skipped",
                        "unsupported": True,
                    }
                    if args.unsupported_action == "copy" and not args.dry_run:
                        copy_model(src_xml, dst_xml)
                else:
                    result = compress_one(core, src_xml, dst_xml, requested_mode, args.group_size, args.dry_run)

            results.append(result)

    payload = {
        "artifact_dir": str(src_root),
        "plan_json": args.plan_json,
        "output_dir": str(dst_root),
        "unsupported_action": args.unsupported_action,
        "dry_run": bool(args.dry_run),
        "results": results,
    }
    if not args.dry_run:
        (dst_root / "precision_plan_applied.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
