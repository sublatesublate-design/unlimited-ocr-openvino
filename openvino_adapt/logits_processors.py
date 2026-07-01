from __future__ import annotations

import numpy as np


class SlidingWindowNoRepeatNgram:
    """Host-side equivalent of Unlimited-OCR's sliding no-repeat ngram processor."""

    def __init__(self, ngram_size: int, window: int, whitelist_token_ids: list[int] | None = None):
        self.ngram_size = int(ngram_size)
        self.window = int(window)
        self.whitelist = set(whitelist_token_ids or [])

    @property
    def enabled(self) -> bool:
        return self.ngram_size > 0 and self.window > 0

    def banned_tokens(self, sequence: list[int]) -> set[int]:
        if not self.enabled or len(sequence) < self.ngram_size:
            return set()
        search_start = max(0, len(sequence) - self.window)
        search_end = len(sequence) - self.ngram_size + 1
        if search_end <= search_start:
            return set()
        current_prefix = tuple(sequence[-(self.ngram_size - 1) :]) if self.ngram_size > 1 else tuple()
        banned: set[int] = set()
        for index in range(search_start, search_end):
            ngram = sequence[index : index + self.ngram_size]
            if self.ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                banned.add(ngram[-1])
        banned.difference_update(self.whitelist)
        return banned

    def apply(self, sequence: list[int], scores: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return scores
        for token_id in self.banned_tokens(sequence):
            if 0 <= token_id < scores.shape[-1]:
                scores[token_id] = -np.inf
        return scores


def select_greedy_token(
    logits: np.ndarray,
    sequence: list[int],
    processor: SlidingWindowNoRepeatNgram | None = None,
) -> int:
    scores = np.array(logits, copy=True)
    if processor is not None:
        scores = processor.apply(sequence, scores)
    return int(np.argmax(scores))


def select_greedy_token_from_topk(
    token_ids: np.ndarray,
    sequence: list[int],
    processor: SlidingWindowNoRepeatNgram | None = None,
) -> int:
    candidates = [int(token_id) for token_id in np.asarray(token_ids).reshape(-1)]
    if not candidates:
        raise ValueError("top-k token list is empty")
    if processor is None or not processor.enabled:
        return candidates[0]
    banned = processor.banned_tokens(sequence)
    for token_id in candidates:
        if token_id not in banned:
            return token_id
    return candidates[0]
