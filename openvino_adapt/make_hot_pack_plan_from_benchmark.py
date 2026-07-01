from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-layer hot expert ids from sparse benchmark selected_experts.")
    parser.add_argument("benchmark_json", nargs="+")
    parser.add_argument("--max-experts", type=int, default=16)
    parser.add_argument("--output-json", default="outputs_openvino_ngram_smoke/hot_pack_plan_from_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    sources: list[str] = []
    for item in args.benchmark_json:
        path = Path(item)
        data = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        for step in data.get("sparse_step_timings", []):
            for layer_id, experts in step.get("selected_experts", {}).items():
                for expert_id in experts:
                    counts[int(layer_id)][int(expert_id)] += 1

    layers = {}
    for layer_id, counter in sorted(counts.items()):
        ranked = sorted(counter.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)
        selected = sorted(expert_id for expert_id, _ in ranked[: args.max_experts])
        layers[str(layer_id)] = {
            "expert_ids": selected,
            "counts": {str(expert_id): count for expert_id, count in sorted(counter.items())},
        }

    payload = {
        "sources": sources,
        "max_experts": args.max_experts,
        "layers": layers,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
