from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OpenVINO benchmark JSON files.")
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def summarize_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sparse = data.get("sparse_timing_summary", {})
    stage = data.get("stage_timings", {})
    steps = data.get("decode_step_seconds", [])
    sparse_seconds_per_step = sparse.get("seconds_per_sparse_step")
    sparse_decode_tokens_per_second = None
    if sparse_seconds_per_step:
        sparse_decode_tokens_per_second = 1.0 / float(sparse_seconds_per_step)
    return {
        "file": str(path),
        "decoder": data.get("decoder", "dense"),
        "devices": data.get("devices", {}),
        "generated_tokens": data.get("generated_tokens"),
        "tokens_per_second": data.get("tokens_per_second"),
        "compile_seconds": data.get("compile_seconds"),
        "total_seconds": data.get("decode_seconds"),
        "vision_seconds": stage.get("vision_seconds"),
        "prefill_seconds": stage.get("prefill_seconds"),
        "decode_loop_seconds": stage.get("decode_loop_seconds"),
        "decode_step_seconds": steps,
        "sparse_total_seconds": sparse.get("total_seconds"),
        "sparse_seconds_per_step": sparse_seconds_per_step,
        "sparse_decode_tokens_per_second": sparse_decode_tokens_per_second,
        "layer_compile_seconds": sparse.get("layer_compile_seconds"),
        "layer_infer_seconds": sparse.get("layer_infer_seconds"),
        "expert_compile_seconds": sparse.get("expert_compile_seconds"),
        "expert_infer_seconds": sparse.get("expert_infer_seconds"),
        "expert_compile_count": sparse.get("expert_compile_count"),
        "expert_call_count": sparse.get("expert_call_count"),
        "hot_pack_compile_seconds": sparse.get("hot_pack_compile_seconds"),
        "hot_pack_infer_seconds": sparse.get("hot_pack_infer_seconds"),
        "fallback_expert_call_count": sparse.get("fallback_expert_call_count"),
        "route_python_seconds": sparse.get("route_python_seconds"),
        "add_infer_seconds": sparse.get("add_infer_seconds"),
        "final_head_compile_seconds": sparse.get("final_head_compile_seconds"),
        "final_head_infer_seconds": sparse.get("final_head_infer_seconds"),
        "final_argmax_compile_seconds": sparse.get("final_argmax_compile_seconds"),
        "final_argmax_infer_seconds": sparse.get("final_argmax_infer_seconds"),
        "final_topk_compile_seconds": sparse.get("final_topk_compile_seconds"),
        "final_topk_infer_seconds": sparse.get("final_topk_infer_seconds"),
    }


def main() -> int:
    args = parse_args()
    rows = [summarize_file(Path(item)) for item in args.json_files]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
