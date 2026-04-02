from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from models.interfaces import InterfaceConfig, to_interface_view, top_tokens
from models.student import LastTokenStudent


@dataclass
class EvalRow:
    family: str
    interface: str
    budget: int
    topk: int
    prompt_id: int
    split: str
    top1_agreement: float
    kl_divergence: float | None
    kl_skipped_reason: str


def kl_divergence(student: np.ndarray, victim: np.ndarray) -> float:
    mask = victim > 0
    if np.any((student[mask] <= 0) | ~np.isfinite(student[mask])):
        return float("nan")
    return float(np.sum(victim[mask] * (np.log(victim[mask]) - np.log(student[mask]))))


def _json_safe(obj: object) -> object:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_extraction(
    *,
    family: str,
    interface_config: InterfaceConfig,
    budget: int,
    topk: int,
    train_prompts: list[str],
    eval_prompts: list[str],
    victim,
    debug_llm: bool,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_jsonl = output_dir / f"debug_{family}_{interface_config.kind}_b{budget}_k{topk}.jsonl"
    debug_csv = output_dir / f"debug_{family}_{interface_config.kind}_b{budget}_k{topk}.csv"

    train_subset = train_prompts[:budget]
    last_toks: list[int] = []
    views: list[np.ndarray] = []

    for prompt in train_subset:
        full = victim.full_softmax(prompt)
        view = to_interface_view(full, interface_config)
        if hasattr(victim, "get_last_token_id"):
            last_toks.append(victim.get_last_token_id(prompt))
        else:
            last_toks.append(ord(prompt[-1]) % full.shape[0] if prompt else 0)
        views.append(view)

    vocab_size = len(views[0])
    student = LastTokenStudent(vocab_size=vocab_size)
    student.fit(last_toks, views)

    rows: list[EvalRow] = []
    debug_records: list[dict] = []

    for i, prompt in enumerate(eval_prompts):
        victim_full = victim.full_softmax(prompt)
        if hasattr(victim, "get_last_token_id"):
            lt = victim.get_last_token_id(prompt)
        else:
            lt = ord(prompt[-1]) % vocab_size if prompt else 0
        student_probs = student.predict_proba(lt)

        victim_argmax = int(np.argmax(victim_full))
        student_argmax = int(np.argmax(student_probs))
        agree = float(victim_argmax == student_argmax)

        reason = ""
        kl = kl_divergence(student_probs, victim_full)
        if not np.isfinite(kl):
            reason = "non_finite_student_or_kl"
            kl_value = None
        else:
            kl_value = kl

        rows.append(
            EvalRow(
                family=family,
                interface=interface_config.kind,
                budget=budget,
                topk=topk,
                prompt_id=i,
                split="eval",
                top1_agreement=agree,
                kl_divergence=kl_value,
                kl_skipped_reason=reason,
            )
        )

        if debug_llm:
            v_idx, v_prob = top_tokens(victim_full, top_n=10)
            s_idx, s_prob = top_tokens(student_probs, top_n=10)
            decode = getattr(victim, "decode_tokens", None)
            v_tok = decode(v_idx.tolist()) if decode else [str(int(x)) for x in v_idx]
            s_tok = decode(s_idx.tolist()) if decode else [str(int(x)) for x in s_idx]
            debug_records.append(
                {
                    "family": family,
                    "interface": interface_config.kind,
                    "budget": budget,
                    "topk": topk,
                    "prompt_id": i,
                    "victim_top10_token_ids": v_idx.tolist(),
                    "victim_top10_tokens": v_tok,
                    "victim_top10_probs": v_prob.tolist(),
                    "student_top10_token_ids": s_idx.tolist(),
                    "student_top10_tokens": s_tok,
                    "student_top10_probs": s_prob.tolist(),
                    "victim_prob_sum": float(victim_full.sum()),
                    "student_prob_sum": float(student_probs.sum()),
                    "victim_has_nan": bool(np.isnan(victim_full).any()),
                    "student_has_nan": bool(np.isnan(student_probs).any()),
                    "victim_has_inf": bool(np.isinf(victim_full).any()),
                    "student_has_inf": bool(np.isinf(student_probs).any()),
                    "vocab_alignment_ok": bool(victim_full.shape == student_probs.shape),
                    "kl_skipped_reason": reason,
                }
            )

    df = pd.DataFrame([r.__dict__ for r in rows])

    if family == "qwen":
        bad = df["kl_divergence"].isna().sum()
        if bad > 0:
            raise RuntimeError(
                f"Qwen KL has {bad} unexpected NaN rows; failing fast instead of zero-filling."
            )

    if debug_llm:
        with debug_jsonl.open("w", encoding="utf-8") as f:
            for record in debug_records:
                f.write(json.dumps(record, default=_json_safe) + "\n")

        with debug_csv.open("w", newline="", encoding="utf-8") as f:
            if debug_records:
                writer = csv.DictWriter(f, fieldnames=list(debug_records[0].keys()))
                writer.writeheader()
                writer.writerows(debug_records)
            else:
                f.write("\n")

    return df
