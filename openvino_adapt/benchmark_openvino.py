from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .artifacts import ArtifactSet, validate_prompt_and_decode_lengths
from .runtime import CompiledArtifacts, add_runtime_args, compile_artifacts, component_devices, make_core, parse_ov_config
from .run_generate_openvino import generate_from_compiled
from .run_prefill_no_cache import encode_prompt_for_pages


def _sum_mapping(step: dict, key: str) -> float:
    values = step.get(key, {})
    return float(sum(float(value) for value in values.values())) if isinstance(values, dict) else 0.0


def summarize_sparse_timings(steps: list[dict]) -> dict:
    summary = {
        "steps": len(steps),
        "total_seconds": 0.0,
        "layer_compile_seconds": 0.0,
        "layer_infer_seconds": 0.0,
        "expert_compile_seconds": 0.0,
        "expert_infer_seconds": 0.0,
        "expert_compile_count": 0,
        "expert_call_count": 0,
        "hot_pack_compile_seconds": 0.0,
        "hot_pack_infer_seconds": 0.0,
        "fallback_expert_call_count": 0,
        "route_python_seconds": 0.0,
        "add_compile_seconds": 0.0,
        "add_infer_seconds": 0.0,
        "final_head_compile_seconds": 0.0,
        "final_head_infer_seconds": 0.0,
        "final_argmax_compile_seconds": 0.0,
        "final_argmax_infer_seconds": 0.0,
        "final_topk_compile_seconds": 0.0,
        "final_topk_infer_seconds": 0.0,
    }
    for step in steps:
        summary["total_seconds"] += float(step.get("total_seconds", 0.0))
        summary["layer_compile_seconds"] += _sum_mapping(step, "layer_compile_seconds")
        summary["layer_infer_seconds"] += _sum_mapping(step, "layer_infer_seconds")
        summary["expert_compile_seconds"] += _sum_mapping(step, "expert_compile_seconds")
        summary["expert_infer_seconds"] += _sum_mapping(step, "expert_infer_seconds")
        summary["expert_compile_count"] += int(step.get("expert_compile_count", 0))
        summary["expert_call_count"] += sum(int(value) for value in step.get("expert_call_count", {}).values())
        summary["hot_pack_compile_seconds"] += _sum_mapping(step, "hot_pack_compile_seconds")
        summary["hot_pack_infer_seconds"] += _sum_mapping(step, "hot_pack_infer_seconds")
        summary["fallback_expert_call_count"] += sum(int(value) for value in step.get("fallback_expert_call_count", {}).values())
        summary["route_python_seconds"] += _sum_mapping(step, "route_python_seconds")
        summary["add_compile_seconds"] += _sum_mapping(step, "add_compile_seconds")
        summary["add_infer_seconds"] += _sum_mapping(step, "add_infer_seconds")
        summary["final_head_compile_seconds"] += float(step.get("final_head_compile_seconds", 0.0))
        summary["final_head_infer_seconds"] += float(step.get("final_head_infer_seconds", 0.0))
        summary["final_argmax_compile_seconds"] += float(step.get("final_argmax_compile_seconds", 0.0))
        summary["final_argmax_infer_seconds"] += float(step.get("final_argmax_infer_seconds", 0.0))
        summary["final_topk_compile_seconds"] += float(step.get("final_topk_compile_seconds", 0.0))
        summary["final_topk_infer_seconds"] += float(step.get("final_topk_infer_seconds", 0.0))
    if steps:
        summary["seconds_per_sparse_step"] = summary["total_seconds"] / len(steps)
    else:
        summary["seconds_per_sparse_step"] = 0.0
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Unlimited-OCR OpenVINO generation loop.")
    parser.add_argument("--image", required=True, nargs="+", help="One or more page images. Multiple images run in one context.")
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
    parser.add_argument("--sparse-precompile-static", action="store_true", help="Compile sparse layer/add/final/hot-pack graphs before timed generation.")
    parser.add_argument("--sparse-precompile-all-experts", action="store_true", help="Also compile all fallback expert graphs before timed generation.")
    parser.add_argument("--sparse-final-argmax", action="store_true", help="Use final_norm_argmax.xml for sparse greedy generation when present.")
    parser.add_argument("--sparse-final-topk", type=int, default=0, help="Use final_norm_topkK.xml for sparse greedy generation when present.")
    parser.add_argument("--sparse-config", nargs="*", default=[], help="Extra sparse OpenVINO compile config as KEY=VALUE pairs.")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--ring-window", type=int, default=128)
    parser.add_argument("--eos-token-id", type=int, default=1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=128)
    parser.add_argument("--output-json", default="")
    add_runtime_args(parser)
    return parser.parse_args()


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
            raise FileNotFoundError(f"Missing OpenVINO sparse benchmark artifacts:\n{joined}")
    tokenizer = AutoTokenizer.from_pretrained(artifacts.tokenizer_dir, trust_remote_code=True)
    input_ids, _ = encode_prompt_for_pages(tokenizer, args.prompt, page_count=len(args.image))
    if args.decoder == "dense":
        validate_prompt_and_decode_lengths(artifacts, input_ids.shape[1], args.ring_window)
    else:
        from .artifacts import prefill_seq_len

        expected_prefill = prefill_seq_len(artifacts.prefill_model)
        if expected_prefill is not None and expected_prefill != input_ids.shape[1]:
            raise ValueError(
                f"Prefill graph expects sequence length {expected_prefill}, but this prompt needs {input_ids.shape[1]}."
            )

    devices = component_devices(args)
    core = make_core(args.cache_dir)
    compile_start = time.time()
    sparse_runtime = None
    sparse_metadata = None
    sparse_precompile = {}
    if args.decoder == "dense":
        compiled = compile_artifacts(core, artifacts, devices)
    else:
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
        if args.sparse_precompile_static:
            layers = [int(layer_id) for layer_id in sparse_metadata["layers"].keys()]
            sparse_precompile = sparse_runtime.precompile_static(
                sparse_metadata,
                layers,
                args.sparse_precompile_all_experts,
            )
    compile_seconds = time.time() - compile_start

    result = generate_from_compiled(
        embed=compiled.embed,
        vision=compiled.vision,
        prefill=compiled.prefill,
        decode=compiled.decode,
        sparse_runtime=sparse_runtime,
        sparse_metadata=sparse_metadata,
        tokenizer=tokenizer,
        images=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        ring_window=args.ring_window,
        eos_token_id=args.eos_token_id,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        ngram_window=args.ngram_window,
    )
    payload = {
        "device": args.device,
        "devices": devices.as_dict(),
        "decoder": args.decoder,
        "sparse_artifact_dir": args.sparse_artifact_dir if args.decoder == "sparse" else "",
        "cache_dir": args.cache_dir,
        "sparse_device": args.sparse_device or devices.decode,
        "sparse_expert_device": args.sparse_expert_device or args.sparse_device or devices.decode,
        "sparse_hot_pack_device": args.sparse_hot_pack_device or args.sparse_expert_device or args.sparse_device or devices.decode,
        "sparse_config": parse_ov_config(args.sparse_config),
        "sparse_final_argmax": args.sparse_final_argmax,
        "sparse_final_topk": args.sparse_final_topk,
        "images": args.image,
        "page_count": len(args.image),
        "prompt_tokens": result["prompt_tokens"],
        "generated_tokens": len(result["generated_ids"]),
        "compile_seconds": compile_seconds,
        "sparse_precompile": sparse_precompile,
        "decode_seconds": result["decode_seconds"],
        "stage_timings": result.get("stage_timings", {}),
        "decode_step_seconds": result.get("decode_step_seconds", []),
        "tokens_per_second": result["tokens_per_second"],
        "generated_text": result["text"],
    }
    if result.get("sparse_step_timings"):
        payload["sparse_timing_summary"] = summarize_sparse_timings(result["sparse_step_timings"])
        payload["sparse_step_timings"] = result["sparse_step_timings"]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
