from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HF_CODE = ROOT / "hf_remote_code"


def load_json(name: str) -> dict:
    with (HF_CODE / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(name: str) -> str:
    return (HF_CODE / name).read_text(encoding="utf-8")


def find_defs(source: str) -> tuple[list[str], list[str]]:
    classes = re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.MULTILINE)
    funcs = re.findall(r"^(?:def|    def)\s+([A-Za-z_][A-Za-z0-9_]*)\(", source, re.MULTILINE)
    return classes, funcs


def count_weight_prefixes(index: dict) -> Counter:
    weights = index.get("weight_map", {})
    prefixes: Counter[str] = Counter()
    for name in weights:
        if name.startswith("model.vision_model."):
            prefixes["vision_model"] += 1
        elif name.startswith("model.sam_model."):
            prefixes["sam_model"] += 1
        elif name.startswith("model.projector."):
            prefixes["projector"] += 1
        elif name.startswith("model.embed_tokens."):
            prefixes["embed_tokens"] += 1
        elif name.startswith("model.layers."):
            prefixes["decoder_layers"] += 1
        elif name.startswith("model.norm."):
            prefixes["decoder_norm"] += 1
        elif name.startswith("lm_head."):
            prefixes["lm_head"] += 1
        else:
            prefixes["other"] += 1
    return prefixes


def count_layers(index: dict) -> tuple[list[int], list[int], list[int]]:
    names = index.get("weight_map", {})
    decoder_layers = sorted({int(m.group(1)) for k in names if (m := re.search(r"model\.layers\.(\d+)\.", k))})
    vision_layers = sorted(
        {int(m.group(1)) for k in names if (m := re.search(r"model\.vision_model\.transformer\.layers\.(\d+)\.", k))}
    )
    sam_blocks = sorted({int(m.group(1)) for k in names if (m := re.search(r"model\.sam_model\.blocks\.(\d+)\.", k))})
    return decoder_layers, vision_layers, sam_blocks


def estimate_kv_cache(config: dict) -> dict[str, float]:
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_key_value_heads"]
    head_dim = config.get("v_head_dim", hidden // config["num_attention_heads"])
    dtype_bytes = 2
    bytes_per_token = layers * 2 * heads * head_dim * dtype_bytes
    return {
        "bytes_per_token": bytes_per_token,
        "mb_per_1024_tokens": bytes_per_token * 1024 / (1024**2),
        "mb_for_ring_128": bytes_per_token * 128 / (1024**2),
        "mb_for_32768_full_cache": bytes_per_token * 32768 / (1024**2),
    }


def line_hits(source: str, patterns: list[str]) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for no, line in enumerate(source.splitlines(), 1):
        if any(pattern in line for pattern in patterns):
            hits.append((no, line.strip()))
    return hits


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    config = load_json("config.json")
    index = load_json("model.safetensors.index.json")
    unlimited = read_text("modeling_unlimitedocr.py")
    deepseek = read_text("modeling_deepseekv2.py")
    deepencoder = read_text("deepencoder.py")

    print_section("Model identity")
    print(f"architecture: {', '.join(config.get('architectures', []))}")
    print(f"auto_map.AutoModel: {config.get('auto_map', {}).get('AutoModel')}")
    print(f"model_type: {config.get('model_type')}")
    print(f"transformers_version in config: {config.get('transformers_version')}")

    print_section("Core config")
    keys = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "v_head_dim",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "max_position_embeddings",
        "sliding_window_size",
        "sliding_window",
        "vocab_size",
    ]
    for key in keys:
        print(f"{key}: {config.get(key)}")
    print(f"vision.image_size: {config.get('vision_config', {}).get('image_size')}")
    print(f"projector: {config.get('projector_config')}")

    print_section("Weight-map shape")
    for key, value in count_weight_prefixes(index).most_common():
        print(f"{key}: {value}")
    decoder_layers, vision_layers, sam_blocks = count_layers(index)
    print(f"decoder layer ids: {decoder_layers[:3]}...{decoder_layers[-3:]} ({len(decoder_layers)} total)")
    print(f"clip vision layer ids: {vision_layers[:3]}...{vision_layers[-3:]} ({len(vision_layers)} total)")
    print(f"sam block ids: {sam_blocks[:3]}...{sam_blocks[-3:]} ({len(sam_blocks)} total)")

    print_section("Python classes")
    for label, source in [
        ("modeling_unlimitedocr.py", unlimited),
        ("modeling_deepseekv2.py", deepseek),
        ("deepencoder.py", deepencoder),
    ]:
        classes, funcs = find_defs(source)
        print(f"{label}: {len(classes)} classes, {len(funcs)} defs")
        print("  classes: " + ", ".join(classes[:24]) + (" ..." if len(classes) > 24 else ""))

    print_section("R-SWA / cache markers")
    for no, line in line_hits(
        deepseek,
        [
            "SlidingWindowLlamaAttention",
            "_ring_window",
            "_prefill_length",
            "_ring_pos",
            "DynamicCache",
            "past_key_values",
        ],
    )[:80]:
        print(f"modeling_deepseekv2.py:{no}: {line}")
    for no, line in line_hits(
        unlimited,
        [
            "_ring_window",
            "config.sliding_window = None",
            "prepare_inputs_for_generation",
            "images_seq_mask",
            "infer_multi",
        ],
    )[:80]:
        print(f"modeling_unlimitedocr.py:{no}: {line}")

    print_section("KV cache estimate, bf16/fp16")
    for key, value in estimate_kv_cache(config).items():
        print(f"{key}: {value:.2f}")

    print_section("Suggested OpenVINO cut points")
    print("1. vision_prefill: sam_model + vision_model + projector -> image token embeddings")
    print("2. decoder_prefill: inputs_embeds + attention_mask -> logits + full prefill KV")
    print("3. decoder_decode_one: token_id + position_id + bounded KV -> logits + updated KV")
    print("4. host loop: keep visual/prompt KV fixed, overwrite output KV ring slots")


if __name__ == "__main__":
    main()
