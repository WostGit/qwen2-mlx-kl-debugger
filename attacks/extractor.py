from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from attacks.metrics import kl_divergence, top1_agreement
from models.student import StudentModel


@dataclass
class ExtractionConfig:
    family: str
    interface: str
    budget: int
    topk: int
    debug_llm: bool
    output_dir: Path


def _top10(probs: np.ndarray) -> Dict[str, List[float]]:
    k = min(10, probs.size)
    idx = np.argpartition(-probs, k - 1)[:k]
    idx = idx[np.argsort(-probs[idx])]
    return {"token_ids": idx.tolist(), "probs": probs[idx].tolist()}


def run_extraction(
    prompts: List[str],
    victim_logits_fn: Callable[[str], np.ndarray],
    interface_builder: Callable[[np.ndarray], np.ndarray],
    vocab_size: int,
    config: ExtractionConfig,
    token_decoder: Optional[Callable[[int], str]] = None,
) -> pd.DataFrame:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    student = StudentModel(vocab_size=vocab_size)

    debug_jsonl = config.output_dir / f"debug_{config.family}_{config.interface}_b{config.budget}.jsonl"
    debug_csv = config.output_dir / f"debug_{config.family}_{config.interface}_b{config.budget}.csv"

    jsonl_fp = debug_jsonl.open("w", encoding="utf-8") if config.debug_llm else None
    csv_fp = debug_csv.open("w", newline="", encoding="utf-8") if config.debug_llm else None
    csv_writer = None
    if csv_fp is not None:
        csv_writer = csv.DictWriter(
            csv_fp,
            fieldnames=[
                "prompt_id",
                "interface",
                "budget",
                "victim_sum",
                "student_sum",
                "victim_has_nan",
                "student_has_nan",
                "victim_has_inf",
                "student_has_inf",
                "vocab_aligned",
                "kl_skipped_reason",
            ],
        )
        csv_writer.writeheader()

    train_prompts = prompts[: config.budget]
    eval_prompts = prompts

    for p in train_prompts:
        logits = victim_logits_fn(p)
        full = _softmax_or_fail(logits)
        view = interface_builder(full)
        student.train_on_distribution(view)

    rows = []
    for i, p in enumerate(eval_prompts):
        victim_logits = victim_logits_fn(p)
        victim_full = _softmax_or_fail(victim_logits)
        student_probs = student.predict_distribution()

        reason = ""
        try:
            kl = kl_divergence(victim_full, student_probs)
        except Exception as exc:
            reason = str(exc)
            kl = np.nan

        agreement = top1_agreement(victim_full, student_probs)

        row = {
            "family": config.family,
            "interface": config.interface,
            "budget": config.budget,
            "topk": config.topk,
            "prompt_id": i,
            "top1_agreement": agreement,
            "kl_divergence": kl,
            "kl_skipped_reason": reason,
        }
        rows.append(row)

        if config.debug_llm:
            v_top = _top10(victim_full)
            s_top = _top10(student_probs)
            if token_decoder is not None:
                v_tokens = [token_decoder(t) for t in v_top["token_ids"]]
                s_tokens = [token_decoder(t) for t in s_top["token_ids"]]
            else:
                v_tokens = [str(t) for t in v_top["token_ids"]]
                s_tokens = [str(t) for t in s_top["token_ids"]]

            dbg = {
                "prompt_id": i,
                "prompt": p,
                "interface": config.interface,
                "budget": config.budget,
                "victim_top10_token_ids": v_top["token_ids"],
                "victim_top10_tokens": v_tokens,
                "victim_top10_probs": v_top["probs"],
                "student_top10_token_ids": s_top["token_ids"],
                "student_top10_tokens": s_tokens,
                "student_top10_probs": s_top["probs"],
                "victim_sum": float(victim_full.sum()),
                "student_sum": float(student_probs.sum()),
                "victim_has_nan": bool(np.isnan(victim_full).any()),
                "student_has_nan": bool(np.isnan(student_probs).any()),
                "victim_has_inf": bool(np.isinf(victim_full).any()),
                "student_has_inf": bool(np.isinf(student_probs).any()),
                "vocab_aligned": bool(victim_full.shape == student_probs.shape),
                "kl_skipped_reason": reason,
            }
            jsonl_fp.write(json.dumps(dbg) + "\n")
            csv_writer.writerow(
                {
                    "prompt_id": i,
                    "interface": config.interface,
                    "budget": config.budget,
                    "victim_sum": dbg["victim_sum"],
                    "student_sum": dbg["student_sum"],
                    "victim_has_nan": dbg["victim_has_nan"],
                    "student_has_nan": dbg["student_has_nan"],
                    "victim_has_inf": dbg["victim_has_inf"],
                    "student_has_inf": dbg["student_has_inf"],
                    "vocab_aligned": dbg["vocab_aligned"],
                    "kl_skipped_reason": reason,
                }
            )

    if jsonl_fp:
        jsonl_fp.close()
    if csv_fp:
        csv_fp.close()

    df = pd.DataFrame(rows)
    return df


def _softmax_or_fail(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    ex = np.exp(z)
    s = ex.sum()
    probs = ex / s
    if not np.isfinite(probs).all():
        raise FloatingPointError("Victim softmax produced NaN/Inf")
    if np.any(probs < 0):
        raise FloatingPointError("Victim softmax produced negative probabilities")
    return probs.astype(np.float64)
