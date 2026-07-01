from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .artifacts import ArtifactSet, prefill_seq_len
from .benchmark_openvino import summarize_sparse_timings
from .runtime import CompiledArtifacts, add_runtime_args, component_devices, make_core, parse_ov_config
from .run_generate_openvino import generate_from_compiled
from .run_prefill_no_cache import encode_prompt_for_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep OpenVINO artifacts compiled and serve OCR jobs over JSONL stdin/stdout.")
    parser.add_argument("--base-model-dir", default="openvino_models/unlimited_ocr")
    parser.add_argument("--prefill-model", default="openvino_models/unlimited_ocr_kv_dense_prefill550/decoder_prefill_kv.xml")
    parser.add_argument("--decode-model", default="openvino_models/unlimited_ocr_kv_dense/decoder_decode_one.xml")
    parser.add_argument("--tokenizer", default="models/Unlimited-OCR")
    parser.add_argument("--decoder", choices=("dense", "sparse"), default="sparse")
    parser.add_argument("--sparse-artifact-dir", default="openvino_models/sparse_decode_past677_mixed_fp4")
    parser.add_argument("--sparse-device", default="", help="OpenVINO device for sparse layer graphs. Defaults to --decode-device/--device.")
    parser.add_argument("--sparse-expert-device", default="", help="OpenVINO device for sparse expert graphs. Defaults to --sparse-device.")
    parser.add_argument("--sparse-hot-pack-dir", default="", help="Optional root/single-layer dir containing hot_expert_pack.xml.")
    parser.add_argument("--sparse-hot-pack-device", default="", help="OpenVINO device for hot expert pack graphs.")
    parser.add_argument("--sparse-precompile-static", action="store_true", help="Compile sparse layer/add/final/hot-pack graphs before accepting jobs.")
    parser.add_argument("--sparse-precompile-all-experts", action="store_true", help="Also compile all fallback expert graphs before accepting jobs.")
    parser.add_argument("--sparse-final-argmax", action="store_true", help="Use final_norm_argmax.xml for sparse greedy generation when present.")
    parser.add_argument("--sparse-final-topk", type=int, default=0, help="Use final_norm_topkK.xml for sparse greedy generation when present.")
    parser.add_argument("--sparse-config", nargs="*", default=[], help="Extra sparse OpenVINO compile config as KEY=VALUE pairs.")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ring-window", type=int, default=128)
    parser.add_argument("--eos-token-id", type=int, default=1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=128)
    add_runtime_args(parser)
    return parser.parse_args()


def write_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def normalize_images(job: dict) -> list[str]:
    if "images" in job:
        images = job["images"]
        if not isinstance(images, list) or not images:
            raise ValueError("job.images must be a non-empty list")
        return [str(item) for item in images]
    if "image" in job:
        return [str(job["image"])]
    raise ValueError("job must include image or images")


def maybe_write_outputs(output_dir: str | None, result: dict) -> None:
    if not output_dir:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "continuous.md").write_text(result["text"], encoding="utf-8")
    manifest = {
        "prompt_tokens": result["prompt_tokens"],
        "page_count": result.get("page_count"),
        "generated_tokens": len(result["generated_ids"]),
        "decode_seconds": result["decode_seconds"],
        "tokens_per_second": result["tokens_per_second"],
        "stage_timings": result.get("stage_timings", {}),
        "decode_step_seconds": result.get("decode_step_seconds", []),
    }
    if result.get("sparse_step_timings"):
        manifest["sparse_timing_summary"] = summarize_sparse_timings(result["sparse_step_timings"])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    from transformers import AutoTokenizer

    args = parse_args()
    artifacts = ArtifactSet.from_paths(args.base_model_dir, args.prefill_model, args.decode_model, args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(artifacts.tokenizer_dir, trust_remote_code=True)
    devices = component_devices(args)
    core = make_core(args.cache_dir)

    compile_start = time.time()
    sparse_runtime = None
    sparse_metadata = None
    sparse_precompile = {}
    if args.decoder == "dense":
        artifacts.require_files()
        compiled = CompiledArtifacts(
            embed=core.compile_model(artifacts.embed_tokens, devices.embed),
            vision=core.compile_model(artifacts.vision_tokens, devices.vision),
            prefill=core.compile_model(artifacts.prefill_model, devices.prefill),
            decode=core.compile_model(artifacts.decode_model, devices.decode),
            devices=devices,
        )
    else:
        from .run_sparse_decode_openvino import SparseDecodeRuntime

        sparse_dir = Path(args.sparse_artifact_dir)
        sparse_metadata = json.loads((sparse_dir / "metadata.json").read_text(encoding="utf-8"))
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
        if args.sparse_precompile_static:
            layers = [int(layer_id) for layer_id in sparse_metadata["layers"].keys()]
            sparse_precompile = sparse_runtime.precompile_static(
                sparse_metadata,
                layers,
                args.sparse_precompile_all_experts,
            )

    write_json(
        {
            "event": "ready",
            "decoder": args.decoder,
            "devices": devices.as_dict(),
            "compile_seconds": time.time() - compile_start,
            "sparse_precompile": sparse_precompile,
            "sparse_final_argmax": args.sparse_final_argmax,
            "sparse_final_topk": args.sparse_final_topk,
        }
    )

    expected_prefill = prefill_seq_len(artifacts.prefill_model)
    for line_no, line in enumerate(sys.stdin, 1):
        line = line.strip()
        if not line:
            continue
        job_started = time.time()
        try:
            job = json.loads(line)
            images = normalize_images(job)
            prompt = str(job.get("prompt", args.prompt))
            input_ids, _ = encode_prompt_for_pages(tokenizer, prompt, page_count=len(images))
            if expected_prefill is not None and expected_prefill != input_ids.shape[1]:
                raise ValueError(
                    f"Prefill graph expects sequence length {expected_prefill}, but this job needs {input_ids.shape[1]}."
                )
            result = generate_from_compiled(
                embed=compiled.embed,
                vision=compiled.vision,
                prefill=compiled.prefill,
                decode=compiled.decode,
                sparse_runtime=sparse_runtime,
                sparse_metadata=sparse_metadata,
                tokenizer=tokenizer,
                images=images,
                prompt=prompt,
                max_new_tokens=int(job.get("max_new_tokens", args.max_new_tokens)),
                ring_window=int(job.get("ring_window", args.ring_window)),
                eos_token_id=int(job.get("eos_token_id", args.eos_token_id)),
                no_repeat_ngram_size=int(job.get("no_repeat_ngram_size", args.no_repeat_ngram_size)),
                ngram_window=int(job.get("ngram_window", args.ngram_window)),
            )
            maybe_write_outputs(job.get("output_dir"), result)
            payload = {
                "event": "result",
                "id": job.get("id", line_no),
                "ok": True,
                "images": images,
                "prompt_tokens": result["prompt_tokens"],
                "generated_tokens": len(result["generated_ids"]),
                "generated_ids": result["generated_ids"],
                "text": result["text"],
                "decode_seconds": result["decode_seconds"],
                "tokens_per_second": result["tokens_per_second"],
                "stage_timings": result.get("stage_timings", {}),
                "decode_step_seconds": result.get("decode_step_seconds", []),
                "wall_seconds": time.time() - job_started,
            }
            if result.get("sparse_step_timings"):
                payload["sparse_timing_summary"] = summarize_sparse_timings(result["sparse_step_timings"])
            write_json(payload)
        except Exception as exc:
            write_json(
                {
                    "event": "error",
                    "id": line_no,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": time.time() - job_started,
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
