from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .wrappers import DecoderDecodeOneStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run split sparse decode-one graphs and compare with PyTorch wrapper.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--artifact-dir", default="openvino_models/sparse_decode_past677")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--expert-device", default="", help="Defaults to --device.")
    parser.add_argument("--cache-dir", default="", help="Optional OpenVINO model/kernel cache directory.")
    parser.add_argument("--past-len", type=int, default=677)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="Layer ids to run. Defaults to metadata/all layers.")
    parser.add_argument("--compile-all-experts", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def output_values(compiled, feeds) -> list[np.ndarray]:
    result = compiled(feeds)
    return [result[compiled.output(index)] for index in range(len(compiled.outputs))]


class SparseDecodeRuntime:
    def __init__(self, artifact_dir: Path, device: str, expert_device: str, cache_dir: str | Path = ""):
        import openvino as ov

        self.artifact_dir = artifact_dir
        self.device = device
        self.expert_device = expert_device or device
        self.core = ov.Core()
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.core.set_property({"CACHE_DIR": str(cache_path)})
        self.compiled_layers: dict[int, object] = {}
        self.compiled_add: dict[int, object] = {}
        self.compiled_experts: dict[tuple[int, int], object] = {}
        self.final_head = None

    def compile_layer(self, layer_id: int, kind: str):
        if layer_id in self.compiled_layers:
            return self.compiled_layers[layer_id]
        layer_dir = self.artifact_dir / f"layer_{layer_id:02d}"
        model_path = layer_dir / ("dense_layer.xml" if kind == "dense" else "attention_gate.xml")
        compiled = self.core.compile_model(model_path, self.device)
        self.compiled_layers[layer_id] = compiled
        return compiled

    def compile_add(self, layer_id: int):
        if layer_id not in self.compiled_add:
            self.compiled_add[layer_id] = self.core.compile_model(
                self.artifact_dir / f"layer_{layer_id:02d}" / "add_moe_residual.xml",
                self.device,
            )
        return self.compiled_add[layer_id]

    def compile_expert(self, layer_id: int, expert_id: int):
        key = (layer_id, expert_id)
        if key not in self.compiled_experts:
            self.compiled_experts[key] = self.core.compile_model(
                self.artifact_dir / f"layer_{layer_id:02d}" / "experts" / f"expert_{expert_id:02d}.xml",
                self.expert_device,
            )
        return self.compiled_experts[key]

    def compile_final_head(self):
        if self.final_head is None:
            self.final_head = self.core.compile_model(self.artifact_dir / "final_norm_head.xml", self.device)
        return self.final_head


def run_sparse_decode(
    runtime: SparseDecodeRuntime,
    metadata: dict,
    hidden: np.ndarray,
    position_ids: np.ndarray,
    attention_mask: np.ndarray,
    past_keys: list[np.ndarray],
    past_values: list[np.ndarray],
    layers: list[int],
    compile_all_experts: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], dict]:
    timings = {
        "layer_compile_seconds": {},
        "layer_infer_seconds": {},
        "expert_compile_count": 0,
        "expert_infer_seconds": {},
        "selected_experts": {},
    }
    new_keys: list[np.ndarray] = []
    new_values: list[np.ndarray] = []
    for layer_id in layers:
        layer_meta = metadata["layers"][str(layer_id)]
        kind = layer_meta["kind"]
        compile_start = time.time()
        layer_graph = runtime.compile_layer(layer_id, kind)
        timings["layer_compile_seconds"][str(layer_id)] = time.time() - compile_start
        feeds = [hidden, position_ids, attention_mask, past_keys[layer_id], past_values[layer_id]]
        infer_start = time.time()
        values = output_values(layer_graph, feeds)
        timings["layer_infer_seconds"][str(layer_id)] = time.time() - infer_start
        if kind == "dense":
            hidden, key_new, value_new = values
            new_keys.append(key_new)
            new_values.append(value_new)
            continue

        attn_residual, moe_input, topk_idx, topk_weight, shared_out, key_new, value_new = values
        flat = moe_input.reshape(-1, moe_input.shape[-1]).astype(np.float32)
        topk_idx_flat = topk_idx.reshape(flat.shape[0], -1)
        topk_weight_flat = topk_weight.reshape(flat.shape[0], -1).astype(np.float32)
        timings["selected_experts"][str(layer_id)] = sorted({int(x) for x in topk_idx_flat.reshape(-1)})
        routed = np.zeros_like(flat, dtype=np.float32)

        if compile_all_experts:
            for expert_id in range(layer_meta["experts"]):
                before = len(runtime.compiled_experts)
                runtime.compile_expert(layer_id, expert_id)
                timings["expert_compile_count"] += len(runtime.compiled_experts) - before

        for token_index in range(flat.shape[0]):
            token = flat[token_index : token_index + 1]
            token_out = np.zeros_like(token, dtype=np.float32)
            for route_index in range(topk_idx_flat.shape[1]):
                expert_id = int(topk_idx_flat[token_index, route_index])
                before = len(runtime.compiled_experts)
                expert_graph = runtime.compile_expert(layer_id, expert_id)
                timings["expert_compile_count"] += len(runtime.compiled_experts) - before
                expert_start = time.time()
                expert_out = output_values(expert_graph, [token])[0].astype(np.float32)
                timings["expert_infer_seconds"][str(layer_id)] = timings["expert_infer_seconds"].get(str(layer_id), 0.0) + (
                    time.time() - expert_start
                )
                token_out += expert_out * topk_weight_flat[token_index, route_index]
            routed[token_index : token_index + 1] = token_out
        routed = routed.reshape(moe_input.shape)
        add_graph = runtime.compile_add(layer_id)
        hidden = output_values(add_graph, [attn_residual, shared_out, routed])[0]
        new_keys.append(key_new)
        new_values.append(value_new)

    logits = output_values(runtime.compile_final_head(), [hidden])[0]
    return logits, new_keys, new_values, timings


def main() -> int:
    from transformers import AutoModel

    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    all_layers = [int(layer_id) for layer_id in metadata["layers"].keys()]
    layers = args.layers if args.layers is not None else all_layers

    torch.manual_seed(args.seed)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    hidden = torch.randn(1, 1, model.config.hidden_size)
    position_ids = torch.full((1, 1), args.past_len, dtype=torch.long)
    attention_mask = torch.zeros((1, 1, 1, args.past_len + 1), dtype=torch.float32)
    past_keys = [
        torch.randn(1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim) * 0.01
        for _ in range(model.config.num_hidden_layers)
    ]
    past_values = [
        torch.randn(1, model.config.num_key_value_heads, args.past_len, model.config.v_head_dim) * 0.01
        for _ in range(model.config.num_hidden_layers)
    ]

    with torch.no_grad():
        wrapper = DecoderDecodeOneStep(model)
        pt_hidden = hidden
        ref_keys: list[torch.Tensor] = []
        ref_values: list[torch.Tensor] = []
        for layer_id in layers:
            layer = model.model.layers[layer_id]
            pt_hidden, key_new, value_new = wrapper._layer_forward(
                layer, pt_hidden, position_ids, attention_mask, past_keys[layer_id], past_values[layer_id]
            )
            ref_keys.append(key_new)
            ref_values.append(value_new)
        ref_logits = model.lm_head(model.model.norm(pt_hidden)).float().numpy()

    runtime = SparseDecodeRuntime(artifact_dir, args.device, args.expert_device, args.cache_dir)
    logits, keys, values, timings = run_sparse_decode(
        runtime,
        metadata,
        hidden.numpy(),
        position_ids.numpy(),
        attention_mask.numpy(),
        [x.numpy() for x in past_keys],
        [x.numpy() for x in past_values],
        layers,
        compile_all_experts=args.compile_all_experts,
    )
    payload = {
        "artifact_dir": str(artifact_dir),
        "device": args.device,
        "expert_device": args.expert_device or args.device,
        "layers": layers,
        "logits_max_abs": float(np.max(np.abs(ref_logits - logits))),
        "logits_mean_abs": float(np.mean(np.abs(ref_logits - logits))),
        "key_max_abs": float(max(np.max(np.abs(ref_keys[i].numpy() - keys[i])) for i in range(len(keys)))),
        "value_max_abs": float(max(np.max(np.abs(ref_values[i].numpy() - values[i])) for i in range(len(values)))),
        "timings": timings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
