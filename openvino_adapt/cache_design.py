from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVShape:
    layers: int = 12
    batch: int = 1
    kv_heads: int = 10
    head_dim: int = 128
    dtype_bytes: int = 2

    @property
    def bytes_per_token(self) -> int:
        return self.layers * 2 * self.batch * self.kv_heads * self.head_dim * self.dtype_bytes

    def mib(self, tokens: int) -> float:
        return self.bytes_per_token * tokens / (1024**2)


@dataclass
class RingCacheState:
    prefill_length: int
    ring_window: int = 128
    ring_pos: int = 0

    @property
    def capacity(self) -> int:
        return self.prefill_length + self.ring_window

    def slot_for_next_token(self) -> int:
        return self.prefill_length + self.ring_pos

    def advance(self, tokens: int = 1) -> None:
        self.ring_pos = (self.ring_pos + tokens) % self.ring_window


def main() -> int:
    shape = KVShape()
    print(f"bytes_per_token: {shape.bytes_per_token}")
    for tokens in (128, 1024, 4096, 32768):
        print(f"{tokens}_tokens_mib: {shape.mib(tokens):.2f}")
    one_page_visual_tokens = 273
    example_prompt_tokens = 32
    state = RingCacheState(prefill_length=one_page_visual_tokens + example_prompt_tokens)
    print(f"one_page_visual_tokens_1024_base: {one_page_visual_tokens}")
    print(f"example_prefill_with_32_text_tokens: {state.prefill_length}")
    print(f"example_capacity_for_one_1024_page: {state.capacity}")
    print(f"first_decode_slot: {state.slot_for_next_token()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
