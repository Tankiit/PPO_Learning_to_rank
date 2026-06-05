# Explanation Ranking (`explrank`)

Learning-to-rank reward models for **graded explanation quality**, with reproducible configs, fixed evaluation metrics, and optional step-level PRMs.

## Install

```bash
cd explanation-ranking
pip install -e ".[dev]"
pre-commit install
```

## Layout

| Path | Role |
|------|------|
| `configs/` | Hydra YAML — data paths, model, loss, experiments (no hardcoded paths in code) |
| `src/explrank/` | Importable package |
| `scripts/` | Thin CLIs that compose Hydra + `explrank` |
| `tests/` | Metric consistency tests (pairwise ↔ Spearman) |
| `slurm/` | Array jobs for loss comparison and step-PRM |

## Quick start

```bash
# Train with ListNet on DS-Critique (override paths via CLI)
python scripts/train.py \
  experiment=loss_comparison \
  data=ds_critique \
  loss=listnet \
  seed=42

# Step-level PRM (Extension 1)
python scripts/train.py experiment=step_prm data=ds_critique seed=42
```

## Metrics

- **Ranking**: NDCG@k, MAP, MRR, Spearman, Kendall τ — see `explrank.metrics.ranking`
- **Pairwise accuracy**: canonical implementation in `explrank.metrics.pairwise` (per-query, ties skipped on gold)
- **Separation ratio**: two documented definitions in `explrank.metrics.separation`

## Experiments

- `configs/experiment/loss_comparison.yaml` — 4 losses × 3 seeds (use `slurm/loss_comparison_array.sh`)
- `configs/experiment/step_prm.yaml` — step-level ListNet + BCE

## Tests

```bash
pytest tests/ -v
```
