from __future__ import annotations

import hashlib

import numpy as np

from models.interfaces import stable_softmax


class ToyVictim:
    """Deterministic toy victim producing a full logit vector from prompt text."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size

    def _seed(self, prompt: str) -> int:
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return int(h[:16], 16)

    def full_softmax(self, prompt: str) -> np.ndarray:
        rng = np.random.default_rng(self._seed(prompt))
        logits = rng.normal(loc=0.0, scale=1.0, size=self.vocab_size)
        # Add a weak structured signal from suffix token-ish feature.
        last_char = ord(prompt[-1]) if prompt else 0
        logits[last_char % self.vocab_size] += 1.25
        return stable_softmax(logits)
