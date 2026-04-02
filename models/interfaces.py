"""Interface view builders derived from a single full victim softmax."""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid softmax denominator: {denom}")
    probs = exps / denom
    return probs.astype(np.float64)


def interface_from_full_softmax(
    full_softmax: np.ndarray,
    interface: str,
    topk: int = 10,
) -> np.ndarray:
    """Derive an interface view from a full-support probability vector."""
    p = np.asarray(full_softmax, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"Expected 1D probability vector, got shape={p.shape}")

    s = float(np.sum(p))
    if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
        raise ValueError(f"full_softmax must sum to 1, got sum={s}")

    if interface == "probs":
        return p.copy()

    if interface == "argmax":
        out = np.zeros_like(p)
        out[int(np.argmax(p))] = 1.0
        return out

    if interface == "topk":
        k = max(1, min(int(topk), p.shape[0]))
        idx = np.argpartition(p, -k)[-k:]
        out = np.zeros_like(p)
        mass = float(np.sum(p[idx]))
        if mass <= EPS:
            raise ValueError(f"Top-k mass too small for renormalization: {mass}")
        out[idx] = p[idx] / mass
        return out

    raise ValueError(f"Unknown interface: {interface}")


def top_tokens(probs: np.ndarray, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    k = max(1, min(int(n), probs.shape[0]))
    idx = np.argpartition(probs, -k)[-k:]
    idx = idx[np.argsort(probs[idx])[::-1]]
    return idx, probs[idx]


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), both over same full support."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch for KL: p={p.shape}, q={q.shape}")

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Negative probabilities are invalid for KL")

    p_sum = float(np.sum(p))
    q_sum = float(np.sum(q))
    if abs(p_sum - 1.0) > 1e-6 or abs(q_sum - 1.0) > 1e-6:
        raise ValueError(f"Probability sum check failed: p_sum={p_sum}, q_sum={q_sum}")

    q_safe = np.clip(q, EPS, None)
    p_safe = np.clip(p, EPS, None)
    kl = float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))
    if not np.isfinite(kl):
        raise ValueError(f"KL is non-finite: {kl}")
    return kl
