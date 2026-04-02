# qwen2-mlx-kl-debugger

Research repo for black-box interface leakage experiments on **GitHub Actions macOS runners**.

## Goal

Make one-token next-token extraction experiments auditable and KL-safe:

- Victim: `Qwen/Qwen2-0.5B-Instruct` via **MLX + mlx-lm**.
- Interfaces from the **same full next-token softmax**:
  - `argmax`: one-hot
  - `topk`: renormalized top-k
  - `probs`: full softmax
- Student is trained on the chosen interface.
- Evaluation computes:
  - top-1 agreement with victim
  - real KL divergence against victim full softmax (same token support)
- `--debug-llm` emits verbose JSONL/CSV artifacts and fails fast on unexpected NaN/Inf KL behavior.

## Repository layout

- `models/` - victim and interface-view construction.
- `attacks/` - student model/training helpers.
- `experiments/` - toy baseline and Qwen2 MLX experiments.
- `scripts/` - runners for budget sweeps and plotting.
- `results/` - generated CSVs/plots/debug artifacts.
- `.github/workflows/` - macOS Actions workflow.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py --debug-llm --qwen-topk 10
```

Outputs:

- `results/results_raw.csv`
- `results/results_summary.csv`
- `results/*.png`
- `results/debug/*.jsonl`
- `results/debug/*.csv`

## Notes

The workflow caches Hugging Face downloads and uploads all CSV/JSONL/plots as artifacts.
