# qwen2-mlx-kl-debugger

Research repo for **black-box interface-leakage** on one-token next-token prediction, with a toy baseline and a real MLX victim (`Qwen/Qwen2-0.5B-Instruct`).

## What this repo enforces

- Interface views are all derived from the **same victim full softmax vector**:
  - `argmax`: one-hot at top token
  - `topk`: renormalized top-k mass
  - `probs`: full softmax
- Student is trained on chosen interface view only.
- Evaluation always computes:
  - top-1 agreement
  - real `kl_divergence(victim_full_softmax || student_pred)` on same token support
- Qwen KL path fails on NaN/inf (no silent zeroing).
- `--debug-llm` writes per-example JSONL + CSV diagnostics:
  - prompt id
  - victim/student top-10 tokens & probabilities
  - probability sums
  - NaN/inf flags
  - vocab alignment
  - explicit KL skip reason

## Layout

- `models/`: victims, interfaces, student
- `attacks/`: extraction loop
- `experiments/`: toy and Qwen sweeps
- `scripts/`: experiment entrypoint
- `results/`: output CSV/plots/debug logs
- `.github/workflows/`: macOS Actions workflow

## Local run

```bash
python -m pip install -r requirements.txt
python scripts/run_experiments.py --out-dir results
python scripts/run_experiments.py --run-qwen --debug-llm --out-dir results
```

Outputs:

- `results/results_raw.csv`
- `results/results_summary.csv`
- `results/plot_top1_agreement.png`
- `results/plot_kl_divergence.png`
- `results/debug/*.jsonl` + `results/debug/*.csv` (with `--debug-llm`)

## Sweep setup

- Budget sweep: `64, 128, 256, 512, 1024`
- Fixed-budget top-k sweep for Qwen top-k interface (`--fixed-topk-budget`, default `512`)

## GitHub Actions

Workflow: `.github/workflows/experiments-macos.yml`

- Runner: `macos-14`
- Installs minimal dependencies
- Caches Hugging Face downloads + pip cache
- Runs toy + Qwen with debug mode
- Uploads all CSV/JSONL/plot artifacts
