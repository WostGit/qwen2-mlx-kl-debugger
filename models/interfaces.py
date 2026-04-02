from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

InterfaceKind = Literal["argmax", "topk", "probs"]


@dataclass(frozen=True)
class InterfaceConfig:
    kind: InterfaceKind
    topk: int = 10


def to_interface_view(full_softmax: np.ndarray, config: InterfaceConfig) -> np.ndarray:
    """Derive an interface view from the *same* full-softmax vector."""
    probs = np.asarray(full_softmax, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"Expected 1D probability vector, got shape={probs.shape}")
    if config.kind == "probs":
        return probs.copy()

    out = np.zeros_like(probs)
    if config.kind == "argmax":
        out[int(np.argmax(probs))] = 1.0
        return out

    if config.kind == "topk":
        k = int(config.topk)
        if k <= 0:
            raise ValueError("topk must be positive")
        k = min(k, probs.shape[0])
        top_idx = np.argpartition(probs, -k)[-k:]
        selected = probs[top_idx]
        mass = selected.sum()
        if mass <= 0 or not np.isfinite(mass):
            # Let caller catch and fail with explicit diagnostics.
            out[top_idx] = np.nan
            return out
        out[top_idx] = selected / mass
        return out

    raise ValueError(f"Unknown interface kind: {config.kind}")


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp_x = np.exp(shifted)
    z = exp_x.sum()
    if z <= 0 or not np.isfinite(z):
        return np.full_like(exp_x, np.nan)
    return exp_x / z


def top_tokens(probs: np.ndarray, top_n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    k = min(top_n, probs.shape[0])
    idx = np.argpartition(probs, -k)[-k:]
    sorted_idx = idx[np.argsort(probs[idx])[::-1]]
    return sorted_idx, probs[sorted_idx]
