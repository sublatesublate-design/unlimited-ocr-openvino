from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KVSpec:
    layers: int = 12
    batch: int = 1
    kv_heads: int = 10
    head_dim: int = 128
    dtype: np.dtype = np.dtype(np.float16)


class RingKVCache:
    """Host-owned KV cache for Unlimited-OCR R-SWA decoding.

    The first `prefill_length` slots are permanent visual/prompt cache. The last
    `ring_window` slots hold generated-token KV and are overwritten in a ring.
    """

    def __init__(self, prefill_length: int, ring_window: int = 128, spec: KVSpec | None = None):
        self.prefill_length = int(prefill_length)
        self.ring_window = int(ring_window)
        self.spec = spec or KVSpec()
        self.ring_pos = 0
        self.keys = [
            np.zeros(self.layer_shape, dtype=self.spec.dtype)
            for _ in range(self.spec.layers)
        ]
        self.values = [
            np.zeros(self.layer_shape, dtype=self.spec.dtype)
            for _ in range(self.spec.layers)
        ]

    @property
    def capacity(self) -> int:
        return self.prefill_length + self.ring_window

    @property
    def layer_shape(self) -> tuple[int, int, int, int]:
        return (self.spec.batch, self.spec.kv_heads, self.capacity, self.spec.head_dim)

    def load_prefill(self, layer: int, key: np.ndarray, value: np.ndarray) -> None:
        if key.shape[-2] != self.prefill_length or value.shape[-2] != self.prefill_length:
            raise ValueError("prefill KV length does not match cache prefill_length")
        self.keys[layer][:, :, : self.prefill_length, :] = key.astype(self.spec.dtype, copy=False)
        self.values[layer][:, :, : self.prefill_length, :] = value.astype(self.spec.dtype, copy=False)

    def update_generated(self, layer: int, key: np.ndarray, value: np.ndarray) -> int:
        slot = self.prefill_length + self.ring_pos
        self.keys[layer][:, :, slot : slot + 1, :] = key.astype(self.spec.dtype, copy=False)
        self.values[layer][:, :, slot : slot + 1, :] = value.astype(self.spec.dtype, copy=False)
        return slot

    def advance(self, tokens: int = 1) -> None:
        self.ring_pos = (self.ring_pos + int(tokens)) % self.ring_window

    def as_openvino_inputs(self, key_prefix: str = "past_key", value_prefix: str = "past_value") -> dict[str, np.ndarray]:
        inputs: dict[str, np.ndarray] = {}
        for layer in range(self.spec.layers):
            inputs[f"{key_prefix}.{layer}"] = self.keys[layer]
            inputs[f"{value_prefix}.{layer}"] = self.values[layer]
        return inputs


def main() -> int:
    cache = RingKVCache(prefill_length=305)
    print(f"capacity: {cache.capacity}")
    print(f"layer_shape: {cache.layer_shape}")
    print(f"first_slot: {cache.prefill_length + cache.ring_pos}")
    cache.advance()
    print(f"second_slot: {cache.prefill_length + cache.ring_pos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
