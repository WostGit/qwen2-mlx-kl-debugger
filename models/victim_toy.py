"""Toy victim with fixed vocabulary and deterministic prompt-conditioned logits."""

from __future__ import annotations

import hashlib
import numpy as np

from models.interfaces import stable_softmax


TOY_VOCAB = [
    " the", " a", " to", " and", " is", " of", " in", " that", " for", " with",
    " on", " as", " by", " from", " at", " be", " this", " it", " an", " not",
]


class ToyVictim:
    def __init__(self) -> None:
        self.vocab = TOY_VOCAB
        self.vocab_size = len(TOY_VOCAB)

    def next_token_full_probs(self, prompt: str) -> np.ndarray:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        logits = rng.normal(loc=0.0, scale=1.0, size=self.vocab_size)
        logits[seed % self.vocab_size] += 2.5
        return stable_softmax(logits)
