from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from attacks.extractor import ExtractionConfig, run_extraction
from attacks.interfaces import argmax_view, probs_view, topk_view
from experiments.common import BUDGETS
from models.victim import ToyVictim


def default_prompts(n: int = 128) -> List[str]:
    return [f"toy prompt {i}: classify latent token" for i in range(n)]


def run_toy(output_dir: Path, debug_llm: bool = False) -> pd.DataFrame:
    victim = ToyVictim(vocab_size=64, seed=7)
    prompts = default_prompts(128)

    rows = []
    for b in BUDGETS:
        b_eff = min(b, len(prompts))
        for interface, fn, k in [
            ("argmax", argmax_view, 1),
            ("topk", lambda p: topk_view(p, 10), 10),
            ("probs", probs_view, victim.vocab_size),
        ]:
            cfg = ExtractionConfig(
                family="toy",
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
                token_decoder=lambda i: f"tok_{i}",
            )
            rows.append(df)
    return pd.concat(rows, ignore_index=True)
