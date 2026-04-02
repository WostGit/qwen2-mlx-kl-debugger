from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


BUDGETS = [64, 128, 256, 512, 1024]


def write_outputs(df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "results_raw.csv"
    summary_path = output_dir / "results_summary.csv"

    df.to_csv(raw_path, index=False)
    summary = (
        df.groupby(["family", "interface", "budget", "topk"], dropna=False)
        .agg(
            top1_agreement_mean=("top1_agreement", "mean"),
            kl_divergence_mean=("kl_divergence", "mean"),
            n=("prompt_id", "count"),
            skipped_kl=("kl_skipped_reason", lambda x: int((x != "").sum())),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    return raw_path, summary_path
