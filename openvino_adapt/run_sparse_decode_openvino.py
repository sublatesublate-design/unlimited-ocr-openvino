from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .wrappers import DecoderDecodeOneStep
from .runtime import parse_ov_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run split sparse decode-one graphs and compare with PyTorch wrapper.")
    parser.add_argument("--model", default="models/Unlimited-OCR")
    parser.add_argument("--artifact-dir", default="openvino_models/sparse_decode_past677")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--expert-device", default="", help="Defaults to --device.")
    parser.add_argument("--hot-pack-dir", default="", help="Optional hot expert pack root or single-layer pack dir.")
    parser.add_argument("--hot-pack-device", default="", help="Defaults to --expert-device/--device.")
    parser.add_argument("--precompile-static", action="store_true", help="Compile layer/add/final/hot-pack graphs before timed decode.")
    parser.add_argument("--precompile-all-experts", action="store_true", help="Also compile all fallback expert graphs up front.")
    parser.add_argument("--cache-dir", default="", help="Optional OpenVINO model/kernel cache directory.")
    parser.add_argument("--ov-config", nargs="*", default=[], help="Extra OpenVINO compile config as KEY=VALUE pairs.")
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
    def __init__(
        self,
        artifact_dir: Path,
        device: str,
        expert_device: str,
        cache_dir: str | Path = "",
        hot_pack_dir: str | Path = "",
        hot_pack_device: str = "",
        ov_config: dict[str, str] | None = None,
        final_topk_k: int = 0,
        final_argmax: bool = False,
    ):
        import openvino as ov

        self.artifact_dir = artifact_dir
        self.device = device
        self.expert_device = expert_device or device
        self.hot_pack_dir = Path(hot_pack_dir) if hot_pack_dir else None
        self.hot_pack_device = hot_pack_device or self.expert_device
        self.ov_config = dict(ov_config or {})
        self.final_topk_k = int(final_topk_k)
        self.final_argmax_enabled = bool(final_argmax)
        self.core = ov.Core()
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.core.set_property({"CACHE_DIR": str(cache_path)})
        self.compiled_layers: dict[int, object] = {}
        self.compiled_add: dict[int, object] = {}
        self.compiled_experts: dict[tuple[int, int], object] = {}
        self.compiled_hot_packs: dict[int, object] = {}
        self.compiled_blocks: dict[int, object] = {}
        self.hot_pack_expert_ids: dict[int, set[int]] = {}
        self.final_head = None
        self.final_topk = None
        self.final_argmax = None

    def compile_layer(self, layer_id: int, kind: str):
        if layer_id in self.compiled_layers:
            return self.compiled_layers[layer_id]
        layer_dir = self.artifact_dir / f"layer_{layer_id:02d}"
        if kind == "dense":
            model_path = layer_dir / "dense_layer.xml"
        elif kind == "fused_hot_moe":
            model_path = layer_dir / "fused_hot_gather_layer.xml"
        elif kind == "fused_moe":
            model_path = layer_dir / "fused_layer.xml"
        else:
            model_path = layer_dir / "attention_gate.xml"
        compiled = self.core.compile_model(model_path, self.device, self.ov_config)
        self.compiled_layers[layer_id] = compiled
        return compiled

    def compile_block(self, block: dict):
        block_index = int(block["index"])
        if block_index not in self.compiled_blocks:
            self.compiled_blocks[block_index] = self.core.compile_model(
                self.artifact_dir / str(block["dir"]) / "decode_block.xml",
                self.device,
                self.ov_config,
            )
        return self.compiled_blocks[block_index]

    def compile_add(self, layer_id: int):
        if layer_id not in self.compiled_add:
            self.compiled_add[layer_id] = self.core.compile_model(
                self.artifact_dir / f"layer_{layer_id:02d}" / "add_moe_residual.xml",
                self.device,
                self.ov_config,
            )
        return self.compiled_add[layer_id]

    def compile_expert(self, layer_id: int, expert_id: int):
        key = (layer_id, expert_id)
        if key not in self.compiled_experts:
            self.compiled_experts[key] = self.core.compile_model(
                self.artifact_dir / f"layer_{layer_id:02d}" / "experts" / f"expert_{expert_id:02d}.xml",
                self.expert_device,
                self.ov_config,
            )
        return self.compiled_experts[key]

    def _hot_pack_paths(self, layer_id: int) -> tuple[Path, Path] | None:
        candidates = [
            self.artifact_dir / f"layer_{layer_id:02d}",
        ]
        if self.hot_pack_dir is not None:
            candidates.extend(
                [
                    self.hot_pack_dir / f"layer_{layer_id:02d}",
                    self.hot_pack_dir,
                ]
            )
        for base in candidates:
            xml = base / "hot_expert_pack.xml"
            metadata = base / "metadata.json"
            if xml.exists() and metadata.exists():
                return xml, metadata
        return None

    def compile_hot_pack(self, layer_id: int):
        paths = self._hot_pack_paths(layer_id)
        if paths is None:
            return None
        if layer_id not in self.compiled_hot_packs:
            xml, metadata_path = paths
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("layer", layer_id)) != layer_id:
                return None
            self.hot_pack_expert_ids[layer_id] = {int(expert_id) for expert_id in metadata["expert_ids"]}
            self.compiled_hot_packs[layer_id] = self.core.compile_model(xml, self.hot_pack_device, self.ov_config)
        return self.compiled_hot_packs[layer_id]

    def compile_final_head(self):
        if self.final_head is None:
            self.final_head = self.core.compile_model(self.artifact_dir / "final_norm_head.xml", self.device, self.ov_config)
        return self.final_head

    def compile_final_topk(self):
        if self.final_topk_k <= 0:
            return None
        model_path = self.artifact_dir / f"final_norm_topk{self.final_topk_k}.xml"
        if not model_path.exists():
            return None
        if self.final_topk is None:
            self.final_topk = self.core.compile_model(model_path, self.device, self.ov_config)
        return self.final_topk

    def compile_final_argmax(self):
        if not self.final_argmax_enabled:
            return None
        model_path = self.artifact_dir / "final_norm_argmax.xml"
        if not model_path.exists():
            return None
        if self.final_argmax is None:
            self.final_argmax = self.core.compile_model(model_path, self.device, self.ov_config)
        return self.final_argmax

    def precompile_static(self, metadata: dict, layers: list[int], compile_all_experts: bool = False) -> dict:
        started = time.perf_counter()
        before_experts = len(self.compiled_experts)
        compiled_hot_packs = 0
        if metadata.get("blocks"):
            for block in metadata["blocks"]:
                self.compile_block(block)
            if self.final_argmax_enabled:
                self.compile_final_argmax()
            if self.final_topk_k > 0:
                self.compile_final_topk()
            self.compile_final_head()
            return {
                "seconds": time.perf_counter() - started,
                "compiled_experts": 0,
                "compiled_hot_packs": 0,
                "compiled_blocks": len(self.compiled_blocks),
                "compiled_final_argmax": self.final_argmax is not None,
                "compiled_final_topk": self.final_topk is not None,
            }
        for layer_id in layers:
            layer_meta = metadata["layers"][str(layer_id)]
            self.compile_layer(layer_id, layer_meta["kind"])
            if layer_meta["kind"] == "moe":
                self.compile_add(layer_id)
                if self.compile_hot_pack(layer_id) is not None:
                    compiled_hot_packs += 1
                if compile_all_experts:
                    hot_pack_ids = self.hot_pack_expert_ids.get(layer_id, set())
                    for expert_id in range(layer_meta["experts"]):
                        if expert_id not in hot_pack_ids:
                            self.compile_expert(layer_id, expert_id)
        self.compile_final_head()
        if self.final_argmax_enabled:
            self.compile_final_argmax()
        if self.final_topk_k > 0:
            self.compile_final_topk()
        return {
            "seconds": time.perf_counter() - started,
            "compiled_experts": len(self.compiled_experts) - before_experts,
            "compiled_hot_packs": compiled_hot_packs,
            "compiled_final_argmax": self.final_argmax is not None,
            "compiled_final_topk": self.final_topk is not None,
        }


def run_final_head(runtime: SparseDecodeRuntime, hidden: np.ndarray, timings: dict):
    final_argmax = None
    if runtime.final_argmax_enabled:
        argmax_compile_start = time.perf_counter()
        final_argmax = runtime.compile_final_argmax()
        timings["final_argmax_compile_seconds"] = time.perf_counter() - argmax_compile_start
    if final_argmax is not None:
        argmax_infer_start = time.perf_counter()
        token_ids = output_values(final_argmax, [hidden])[0]
        timings["final_argmax_infer_seconds"] = time.perf_counter() - argmax_infer_start
        return {"argmax_indices": token_ids}

    final_topk = None
    if runtime.final_topk_k > 0:
        topk_compile_start = time.perf_counter()
        final_topk = runtime.compile_final_topk()
        timings["final_topk_compile_seconds"] = time.perf_counter() - topk_compile_start
    if final_topk is not None:
        topk_infer_start = time.perf_counter()
        values = output_values(final_topk, [hidden])
        timings["final_topk_infer_seconds"] = time.perf_counter() - topk_infer_start
        return {"topk_values": values[0], "topk_indices": values[1]}

    final_compile_start = time.perf_counter()
    final_head = runtime.compile_final_head()
    timings["final_head_compile_seconds"] = time.perf_counter() - final_compile_start
    final_infer_start = time.perf_counter()
    logits = output_values(final_head, [hidden])[0]
    timings["final_head_infer_seconds"] = time.perf_counter() - final_infer_start
    return logits


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
    total_start = time.perf_counter()
    timings = {
        "total_seconds": 0.0,
        "layer_compile_seconds": {},
        "layer_infer_seconds": {},
        "add_compile_seconds": {},
        "add_infer_seconds": {},
        "final_head_compile_seconds": 0.0,
        "final_head_infer_seconds": 0.0,
        "final_argmax_compile_seconds": 0.0,
        "final_argmax_infer_seconds": 0.0,
        "final_topk_compile_seconds": 0.0,
        "final_topk_infer_seconds": 0.0,
        "expert_compile_count": 0,
        "expert_compile_seconds": {},
        "expert_compile_seconds_by_layer": {},
        "expert_infer_seconds": {},
        "expert_call_count": {},
        "hot_pack_compile_seconds": {},
        "hot_pack_infer_seconds": {},
        "hot_pack_expert_ids": {},
        "fallback_expert_call_count": {},
        "route_python_seconds": {},
        "selected_experts": {},
    }
    new_keys: list[np.ndarray] = []
    new_values: list[np.ndarray] = []
    if metadata.get("blocks"):
        new_keys_by_layer: dict[int, np.ndarray] = {}
        new_values_by_layer: dict[int, np.ndarray] = {}
        for block in metadata["blocks"]:
            block_key = f"block_{block['layers'][0]:02d}_{block['layers'][-1]:02d}"
            compile_start = time.perf_counter()
            block_graph = runtime.compile_block(block)
            timings["layer_compile_seconds"][block_key] = time.perf_counter() - compile_start
            feeds = [hidden, position_ids, attention_mask]
            for layer_id in block["layers"]:
                feeds.extend([past_keys[int(layer_id)], past_values[int(layer_id)]])
            infer_start = time.perf_counter()
            values = output_values(block_graph, feeds)
            timings["layer_infer_seconds"][block_key] = time.perf_counter() - infer_start
            hidden = values[0]
            for offset, layer_id in enumerate(block["layers"]):
                new_keys_by_layer[int(layer_id)] = values[1 + offset * 2]
                new_values_by_layer[int(layer_id)] = values[2 + offset * 2]

        final_output = run_final_head(runtime, hidden, timings)
        timings["total_seconds"] = time.perf_counter() - total_start
        ordered_layers = sorted(int(layer_id) for layer_id in metadata["layers"].keys())
        return (
            final_output,
            [new_keys_by_layer[layer_id] for layer_id in ordered_layers],
            [new_values_by_layer[layer_id] for layer_id in ordered_layers],
            timings,
        )

    for layer_id in layers:
        layer_meta = metadata["layers"][str(layer_id)]
        kind = layer_meta["kind"]
        compile_start = time.perf_counter()
        layer_graph = runtime.compile_layer(layer_id, kind)
        timings["layer_compile_seconds"][str(layer_id)] = time.perf_counter() - compile_start
        feeds = [hidden, position_ids, attention_mask, past_keys[layer_id], past_values[layer_id]]
        infer_start = time.perf_counter()
        values = output_values(layer_graph, feeds)
        timings["layer_infer_seconds"][str(layer_id)] = time.perf_counter() - infer_start
        if kind in {"dense", "fused_hot_moe", "fused_moe"}:
            hidden, key_new, value_new = values
            new_keys.append(key_new)
            new_values.append(value_new)
            continue

        attn_residual, moe_input, topk_idx, topk_weight, shared_out, key_new, value_new = values
        flat = moe_input.reshape(-1, moe_input.shape[-1]).astype(np.float32)
        topk_idx_flat = topk_idx.reshape(flat.shape[0], -1)
        topk_weight_flat = topk_weight.reshape(flat.shape[0], -1).astype(np.float32)
        timings["selected_experts"][str(layer_id)] = sorted({int(x) for x in topk_idx_flat.reshape(-1)})
        routed = np.zeros_like(moe_input, dtype=np.float32)
        hot_pack = None
        hot_pack_ids: set[int] = set()
        hot_compile_start = time.perf_counter()
        hot_pack = runtime.compile_hot_pack(layer_id)
        hot_compile_elapsed = time.perf_counter() - hot_compile_start
        if hot_pack is not None:
            hot_pack_ids = runtime.hot_pack_expert_ids[layer_id]
            timings["hot_pack_compile_seconds"][str(layer_id)] = hot_compile_elapsed
            timings["hot_pack_expert_ids"][str(layer_id)] = sorted(hot_pack_ids)
            hot_infer_start = time.perf_counter()
            routed = output_values(hot_pack, [moe_input, topk_idx, topk_weight])[0].astype(np.float32)
            timings["hot_pack_infer_seconds"][str(layer_id)] = time.perf_counter() - hot_infer_start
        routed_flat = routed.reshape(flat.shape)

        if compile_all_experts:
            for expert_id in range(layer_meta["experts"]):
                if expert_id in hot_pack_ids:
                    continue
                before = len(runtime.compiled_experts)
                expert_compile_start = time.perf_counter()
                runtime.compile_expert(layer_id, expert_id)
                timings["expert_compile_count"] += len(runtime.compiled_experts) - before
                if len(runtime.compiled_experts) > before:
                    key = f"{layer_id}:{expert_id}"
                    compile_elapsed = time.perf_counter() - expert_compile_start
                    timings["expert_compile_seconds"][key] = compile_elapsed
                    timings["expert_compile_seconds_by_layer"][str(layer_id)] = (
                        timings["expert_compile_seconds_by_layer"].get(str(layer_id), 0.0) + compile_elapsed
                    )

        route_start = time.perf_counter()
        for token_index in range(flat.shape[0]):
            token = flat[token_index : token_index + 1]
            token_out = np.zeros_like(token, dtype=np.float32)
            for route_index in range(topk_idx_flat.shape[1]):
                expert_id = int(topk_idx_flat[token_index, route_index])
                if expert_id in hot_pack_ids:
                    continue
                before = len(runtime.compiled_experts)
                expert_compile_start = time.perf_counter()
                expert_graph = runtime.compile_expert(layer_id, expert_id)
                timings["expert_compile_count"] += len(runtime.compiled_experts) - before
                if len(runtime.compiled_experts) > before:
                    key = f"{layer_id}:{expert_id}"
                    compile_elapsed = time.perf_counter() - expert_compile_start
                    timings["expert_compile_seconds"][key] = compile_elapsed
                    timings["expert_compile_seconds_by_layer"][str(layer_id)] = (
                        timings["expert_compile_seconds_by_layer"].get(str(layer_id), 0.0) + compile_elapsed
                    )
                expert_start = time.perf_counter()
                expert_out = output_values(expert_graph, [token])[0].astype(np.float32)
                timings["expert_infer_seconds"][str(layer_id)] = timings["expert_infer_seconds"].get(str(layer_id), 0.0) + (
                    time.perf_counter() - expert_start
                )
                timings["expert_call_count"][str(layer_id)] = timings["expert_call_count"].get(str(layer_id), 0) + 1
                timings["fallback_expert_call_count"][str(layer_id)] = (
                    timings["fallback_expert_call_count"].get(str(layer_id), 0) + 1
                )
                token_out += expert_out * topk_weight_flat[token_index, route_index]
            routed_flat[token_index : token_index + 1] += token_out
        route_elapsed = time.perf_counter() - route_start
        timings["route_python_seconds"][str(layer_id)] = (
            route_elapsed
            - timings["expert_infer_seconds"].get(str(layer_id), 0.0)
            - timings["expert_compile_seconds_by_layer"].get(str(layer_id), 0.0)
        )
        routed = routed_flat.reshape(moe_input.shape)
        add_compile_start = time.perf_counter()
        add_graph = runtime.compile_add(layer_id)
        timings["add_compile_seconds"][str(layer_id)] = time.perf_counter() - add_compile_start
        add_infer_start = time.perf_counter()
        hidden = output_values(add_graph, [attn_residual, shared_out, routed])[0]
        timings["add_infer_seconds"][str(layer_id)] = time.perf_counter() - add_infer_start
        new_keys.append(key_new)
        new_values.append(value_new)

    final_output = run_final_head(runtime, hidden, timings)
    timings["total_seconds"] = time.perf_counter() - total_start
    return final_output, new_keys, new_values, timings


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

    runtime = SparseDecodeRuntime(
        artifact_dir,
        args.device,
        args.expert_device,
        args.cache_dir,
        args.hot_pack_dir,
        args.hot_pack_device,
        parse_ov_config(args.ov_config),
    )
    precompile_payload = {}
    if args.precompile_static:
        precompile_payload = runtime.precompile_static(metadata, layers, args.precompile_all_experts)
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
        "hot_pack_dir": args.hot_pack_dir,
        "hot_pack_device": args.hot_pack_device or args.expert_device or args.device,
        "ov_config": parse_ov_config(args.ov_config),
        "precompile": precompile_payload,
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
