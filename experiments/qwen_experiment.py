from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from attacks.extractor import ExtractionConfig, run_extraction
from attacks.interfaces import argmax_view, probs_view, topk_view
from experiments.common import BUDGETS
from models.victim import QwenMLXVictim


def default_prompts(n: int = 128) -> List[str]:
    return [
        f"Prompt {i}: The quick brown fox jumps over the lazy"
        for i in range(n)
    ]


def run_qwen(output_dir: Path, model_name: str, debug_llm: bool = False) -> pd.DataFrame:
    victim = QwenMLXVictim(model_name=model_name)
    prompts = default_prompts(128)

    rows = []
    for b in BUDGETS:
        b_eff = min(b, len(prompts))
        specs = [
            ("argmax", argmax_view, 1),
            ("topk", lambda p: topk_view(p, 10), 10),
            ("probs", probs_view, victim.vocab_size),
        ]
        for interface, fn, k in specs:
            cfg = ExtractionConfig(
                family="qwen",
                interface=interface,
                budget=b_eff,
                topk=k,
                debug_llm=debug_llm,
                output_dir=output_dir,
            )
            df = run_extraction(
                prompts=prompts,
                victim_logits_fn=victim.next_token_logits,
                interface_builder=fn,
                vocab_size=victim.vocab_size,
                config=cfg,
                token_decoder=victim.decode_token,
            )
            _assert_no_unexpected_nan(df, interface, b_eff)
            rows.append(df)

    fixed_budget = min(256, len(prompts))
    for k in [2, 5, 10, 20, 50]:
        cfg = ExtractionConfig(
            family="qwen",
            interface="topk",
            budget=fixed_budget,
            topk=k,
            debug_llm=debug_llm,
            output_dir=output_dir,
        )
        df = run_extraction(
            prompts=prompts,
            victim_logits_fn=victim.next_token_logits,
            interface_builder=lambda p, kk=k: topk_view(p, kk),
            vocab_size=victim.vocab_size,
            config=cfg,
            token_decoder=victim.decode_token,
        )
        _assert_no_unexpected_nan(df, f"topk@{k}", fixed_budget)
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def _assert_no_unexpected_nan(df: pd.DataFrame, interface: str, budget: int) -> None:
    has_nan = df["kl_divergence"].isna()
    if not has_nan.any():
        return
    reasons = set(df.loc[has_nan, "kl_skipped_reason"].tolist())
    raise RuntimeError(
        f"Qwen KL path produced NaN rows for interface={interface}, budget={budget}, reasons={sorted(reasons)}"
    )
