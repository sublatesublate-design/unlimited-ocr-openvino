from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-expert precision plan from a sparse route profile.")
    parser.add_argument("--profile-json", default="outputs_openvino_ngram_smoke/expert_route_profile_4samples.json")
    parser.add_argument("--output-json", default="outputs_openvino_ngram_smoke/expert_precision_plan.json")
    parser.add_argument("--hot-mode", default="fp16", help="Precision label for hot experts.")
    parser.add_argument("--warm-mode", default="int8_asym", help="Precision label for routed but non-hot/non-cold experts.")
    parser.add_argument("--cold-mode", default="int4_asym", help="Precision label for cold experts.")
    parser.add_argument("--unused-mode", default="experimental_int2", help="Precision label for experts unused in the profile.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_path = Path(args.profile_json)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    policy = profile["policy"]

    plan: dict[str, dict] = {}
    summary: dict[str, int] = {}
    for layer_id, layer_policy in policy.items():
        ranked = layer_policy["ranked"]
        hot = set(layer_policy["hot_keep_int8_or_fp16"])
        cold = set(layer_policy["cold_int4_or_experimental_lowbit"])
        unused = set(layer_policy["unused_in_profile"])

        experts: dict[str, dict] = {}
        for item in ranked:
            expert_id = int(item["expert"])
            if expert_id in hot:
                mode = args.hot_mode
                reason = "hot"
            elif expert_id in unused:
                mode = args.unused_mode
                reason = "unused_in_profile"
            elif expert_id in cold:
                mode = args.cold_mode
                reason = "cold"
            else:
                mode = args.warm_mode
                reason = "warm"

            summary[mode] = summary.get(mode, 0) + 1
            experts[str(expert_id)] = {
                "mode": mode,
                "reason": reason,
                "count": int(item["count"]),
                "weight_sum": float(item["weight_sum"]),
                "score": float(item["score"]),
            }
        plan[str(layer_id)] = {"experts": experts}

    payload = {
        "source_profile": str(profile_path),
        "samples": profile.get("samples"),
        "past_len": profile.get("past_len"),
        "note": (
            "OpenVINO/NNCF in this environment does not expose INT2 weight compression. "
            "experimental_int2 is a routing/importance label for a future custom packed kernel; "
            "use int4_asym, fp4, nf4, or cb4 for stock OpenVINO experiments."
        ),
        "summary": summary,
        "plan": plan,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
