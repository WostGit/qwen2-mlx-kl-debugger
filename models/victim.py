from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ToyVictim:
    vocab_size: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def next_token_logits(self, prompt: str) -> np.ndarray:
        prompt_hash = abs(hash(prompt)) % (2**32)
        local_rng = np.random.default_rng(prompt_hash)
        logits = local_rng.normal(loc=0.0, scale=1.0, size=self.vocab_size)
        logits += np.linspace(0.5, -0.5, self.vocab_size)
        return logits.astype(np.float64)


class QwenMLXVictim:
    """Victim wrapper returning full next-token logits from mlx-lm."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load  # delayed import for toy-only runs

        self._model, self._tokenizer = load(self.model_name)

    @property
    def tokenizer(self):
        self._lazy_load()
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        self._lazy_load()
        return int(self._tokenizer.vocab_size)

    def next_token_logits(self, prompt: str) -> np.ndarray:
        self._lazy_load()
        import mlx.core as mx

        ids: List[int] = self._tokenizer.encode(prompt)
        if len(ids) == 0:
            ids = [self._tokenizer.eos_token_id]
        x = mx.array([ids])
        out = self._model(x)
        # shape: [batch, seq, vocab]
        logits = np.array(mx.eval(out[:, -1, :])[0], dtype=np.float64)
        if logits.ndim != 1:
            raise RuntimeError(f"Expected 1D logits, got shape={logits.shape}")
        return logits

    def decode_token(self, token_id: int) -> str:
        self._lazy_load()
        try:
            return self._tokenizer.decode([int(token_id)])
        except Exception:
            return f"<tok:{token_id}>"

    def top_tokens(self, probs: np.ndarray, k: int = 10) -> Dict[str, list]:
        k = min(k, probs.size)
        idx = np.argpartition(-probs, k - 1)[:k]
        idx = idx[np.argsort(-probs[idx])]
        return {
            "token_ids": idx.tolist(),
            "tokens": [self.decode_token(i) for i in idx.tolist()],
            "probs": probs[idx].tolist(),
        }
