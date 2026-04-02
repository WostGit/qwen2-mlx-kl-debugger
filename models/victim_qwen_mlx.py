from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from models.interfaces import stable_softmax


@dataclass
class QwenMLXVictim:
    model_id: str = "mlx-community/Qwen2-0.5B-Instruct-4bit"

    def __post_init__(self) -> None:
        from mlx_lm import load  # type: ignore

        self.model, self.tokenizer = load(self.model_id)

    def _extract_logits_obj(self, model_output: Any) -> Any:
        if hasattr(model_output, "logits"):
            return model_output.logits
        if isinstance(model_output, (tuple, list)) and model_output:
            return model_output[0]
        return model_output

    def get_last_token_id(self, prompt: str) -> int:
        encoded = self.tokenizer(prompt, return_tensors="np")
        input_ids = np.asarray(encoded["input_ids"])
        return int(input_ids[0, -1])

    def full_softmax(self, prompt: str) -> np.ndarray:
        import mlx.core as mx  # type: ignore

        encoded = self.tokenizer(prompt, return_tensors="np")
        input_ids = np.asarray(encoded["input_ids"])
        mx_inputs = mx.array(input_ids)

        model_output = self.model(mx_inputs)
        logits_obj = self._extract_logits_obj(model_output)
        logits_np = np.asarray(logits_obj)
        if logits_np.ndim != 3:
            raise ValueError(f"Unexpected logits shape from MLX model: {logits_np.shape}")
        next_token_logits = logits_np[0, -1, :]
        return stable_softmax(next_token_logits)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def decode_tokens(self, token_ids: list[int]) -> list[str]:
        return [self.tokenizer.decode([tid]) for tid in token_ids]
