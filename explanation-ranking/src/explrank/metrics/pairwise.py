"""
Pairwise ranking accuracy — canonical implementation.

Computes, **per query**, the fraction of gold-distinguishable pairs (i, j)
where the predicted score order matches the gold order:

    gold_i > gold_j  =>  pred_i > pred_j
    gold_i < gold_j  =>  pred_i < pred_j

Pairs with equal gold scores are skipped (ties in relevance).

This definition is consistent with Spearman/Kendall concordance on the same
query when there are no gold ties: perfect monotone predictions yield 1.0
for pairwise accuracy and Spearman ρ = 1.

Previous bugs in ad-hoc scripts often came from:
  - aggregating pairs across queries without per-query normalization
  - comparing ranks instead of scores with wrong tie-breaking
  - treating pred ties as correct when gold differs
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Union

ArrayLike = Union[np.ndarray, torch.Tensor, list]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def pairwise_accuracy(
    gold_scores: ArrayLike,
    pred_scores: ArrayLike,
    *,
    tie_break: str = "strict",
) -> float:
    """
    Pairwise accuracy for a single query (1D arrays).

    Args:
        gold_scores: ground-truth relevance scores
        pred_scores: model scores (same length)
        tie_break: 'strict' — pred tie counts as incorrect when gold differs;
                   'conservative' — pred tie counts as half-correct (not used by default)

    Returns:
        Fraction of correct pairwise orderings among non-tied gold pairs.
    """
    gold = _to_numpy(gold_scores).ravel()
    pred = _to_numpy(pred_scores).ravel()
    if gold.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gold {gold.shape} vs pred {pred.shape}")
    n = len(gold)
    if n < 2:
        return 1.0

    correct = 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if gold[i] == gold[j]:
                continue
            total += 1.0
            gold_prefers_i = gold[i] > gold[j]
            if pred[i] == pred[j]:
                if tie_break == "conservative":
                    correct += 0.5
                continue
            pred_prefers_i = pred[i] > pred[j]
            if gold_prefers_i == pred_prefers_i:
                correct += 1.0

    return float(correct / total) if total > 0 else 1.0


def mean_pairwise_accuracy(
    gold_batch: torch.Tensor,
    pred_batch: torch.Tensor,
) -> float:
    """
    Average pairwise accuracy over a batch of queries.

    Args:
        gold_batch: [N, k] true scores
        pred_batch: [N, k] predicted scores
    """
    if gold_batch.dim() != 2 or pred_batch.dim() != 2:
        raise ValueError("Expected 2D tensors [N, k]")
    n = gold_batch.shape[0]
    accs = [
        pairwise_accuracy(gold_batch[i], pred_batch[i])
        for i in range(n)
    ]
    return float(np.mean(accs)) if accs else 0.0
