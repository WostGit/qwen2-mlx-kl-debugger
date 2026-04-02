from __future__ import annotations

import numpy as np


EPS = 1e-12


def top1_agreement(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    return float(int(np.argmax(victim_probs) == np.argmax(student_probs)))


def kl_divergence(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    if victim_probs.shape != student_probs.shape:
        raise ValueError("KL shape mismatch")
    if not (np.isfinite(victim_probs).all() and np.isfinite(student_probs).all()):
        raise FloatingPointError("Non-finite values in KL inputs")
    if np.any(victim_probs < 0) or np.any(student_probs < 0):
        raise FloatingPointError("Negative probabilities in KL inputs")

    p = victim_probs / victim_probs.sum()
    q = student_probs / student_probs.sum()

    # p support drives expectation; q must be positive where p > 0
    bad = (p > 0) & (q <= 0)
    if np.any(bad):
        raise FloatingPointError("Student has zero probability on victim support")

    ratio = np.log(np.clip(p, EPS, None) / np.clip(q, EPS, None))
    out = np.sum(p * ratio)
    if not np.isfinite(out):
        raise FloatingPointError("KL became non-finite")
    return float(out)
