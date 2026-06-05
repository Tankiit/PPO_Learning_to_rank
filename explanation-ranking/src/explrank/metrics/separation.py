"""
Score separation metrics for PPO reward viability.

Two definitions are used in this codebase (document both when reporting):

**Definition A — variance ratio (default, early stopping)**
    separation_ratio_std = mean_q( std(pred_q) / std(gold_q) )

Used in `src/evaluation/metrics.py` and Table 2 of the paper. Values near 1.0
mean the model preserves gold score spread; MSE-trained models often collapse
to ~0.01.

**Definition B — batch-level std ratio**
    separation_ratio_batch = std(all_pred_flat) / std(all_gold_flat)

Used in some training loops (`train.py`) as a single scalar over flattened
batch scores. Can differ from Definition A when list sizes vary per query.

Additional diagnostics:
    score_range — mean_q(max(pred_q) - min(pred_q))
    score_std   — mean_q(std(pred_q))
"""

from __future__ import annotations

from typing import Dict

import torch


def separation_ratio_std(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Definition A: per-query std ratio, averaged."""
    pred_std = pred_scores.std(dim=-1)
    true_std = true_scores.std(dim=-1)
    ratios = pred_std / true_std.clamp(min=eps)
    return ratios.mean().item()


def separation_ratio_batch(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Definition B: global std ratio over flattened batch."""
    return (pred_scores.std() / true_scores.std().clamp(min=eps)).item()


def score_separation_metrics(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
) -> Dict[str, float]:
    """Return all separation diagnostics."""
    ranges = pred_scores.max(dim=-1).values - pred_scores.min(dim=-1).values
    pred_std = pred_scores.std(dim=-1)
    return {
        "separation_ratio_std": separation_ratio_std(pred_scores, true_scores),
        "separation_ratio_batch": separation_ratio_batch(pred_scores, true_scores),
        "score_range": ranges.mean().item(),
        "score_std": pred_std.mean().item(),
    }
