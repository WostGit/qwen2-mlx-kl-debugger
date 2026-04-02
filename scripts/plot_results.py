from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/results_summary.csv")
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for metric in ["top1_agreement_mean", "kl_divergence_mean"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for interface, sdf in summary.groupby("interface"):
            if interface == "topk" and sdf["topk"].nunique() > 1:
                # only keep budget sweep line for canonical topk=10
                sdf = sdf[sdf["topk"] == 10]
            series = sdf.groupby("budget")[metric].mean().sort_index()
            ax.plot(series.index, series.values, marker="o", label=interface)
        ax.set_title(metric)
        ax.set_xlabel("budget")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        out = outdir / f"plot_{metric}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
