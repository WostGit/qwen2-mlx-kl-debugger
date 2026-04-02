from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import write_outputs
from experiments.qwen_experiment import run_qwen
from experiments.toy_experiment import run_toy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["toy", "qwen", "all"], default="all")
    p.add_argument("--qwen-model", default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--debug-llm", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)

    frames = []
    if args.family in {"toy", "all"}:
        frames.append(run_toy(out, debug_llm=args.debug_llm))
    if args.family in {"qwen", "all"}:
        frames.append(run_qwen(out, model_name=args.qwen_model, debug_llm=args.debug_llm))

    df = pd.concat(frames, ignore_index=True)
    raw_path, summary_path = write_outputs(df, out)

    print(f"Wrote raw results: {raw_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
