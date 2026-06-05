"""NDCG, MAP, MRR, Spearman, Kendall — batched over queries."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy import stats

from explrank.metrics.pairwise import mean_pairwise_accuracy
from explrank.metrics.separation import score_separation_metrics


def ndcg_at_k(pred_scores: torch.Tensor, true_scores: torch.Tensor, k: int) -> float:
    n = pred_scores.shape[0]
    ndcg_sum = 0.0
    for i in range(n):
        _, pred_order = pred_scores[i].sort(descending=True)
        sorted_true = true_scores[i][pred_order][:k]
        positions = torch.arange(1, min(k, len(sorted_true)) + 1, dtype=torch.float32)
        discounts = torch.log2(positions + 1.0)
        gains = torch.pow(2.0, sorted_true.float()) - 1.0
        dcg = (gains / discounts).sum().item()
        ideal_sorted, _ = true_scores[i].sort(descending=True)
        ideal_sorted = ideal_sorted[:k]
        ideal_gains = torch.pow(2.0, ideal_sorted.float()) - 1.0
        ideal_dcg = (ideal_gains / discounts[: len(ideal_gains)]).sum().item()
        ndcg_sum += dcg / ideal_dcg if ideal_dcg > 0 else 1.0
    return ndcg_sum / n


def mean_average_precision(pred_scores: torch.Tensor, true_scores: torch.Tensor) -> float:
    n = pred_scores.shape[0]
    ap_sum = 0.0
    for i in range(n):
        _, pred_order = pred_scores[i].sort(descending=True)
        sorted_true = true_scores[i][pred_order]
        threshold = true_scores[i].median()
        relevant = (sorted_true > threshold).float()
        if relevant.sum() == 0:
            ap_sum += 1.0
            continue
        cum_relevant = relevant.cumsum(dim=0)
        precisions = cum_relevant / torch.arange(1, len(relevant) + 1, dtype=torch.float32)
        ap_sum += (precisions * relevant).sum().item() / relevant.sum().item()
    return ap_sum / n


def mean_reciprocal_rank(pred_scores: torch.Tensor, true_scores: torch.Tensor) -> float:
    n = pred_scores.shape[0]
    rr_sum = 0.0
    for i in range(n):
        _, pred_order = pred_scores[i].sort(descending=True)
        best_true_idx = true_scores[i].argmax()
        rank = (pred_order == best_true_idx).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            rr_sum += 1.0 / (rank[0].item() + 1)
    return rr_sum / n


def spearman_correlation(pred_scores: torch.Tensor, true_scores: torch.Tensor) -> float:
    rho_sum, valid = 0.0, 0
    for i in range(pred_scores.shape[0]):
        p = pred_scores[i].cpu().numpy()
        t = true_scores[i].cpu().numpy()
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            continue
        rho, _ = stats.spearmanr(p, t)
        if not np.isnan(rho):
            rho_sum += rho
            valid += 1
    return rho_sum / max(valid, 1)


def kendall_tau(pred_scores: torch.Tensor, true_scores: torch.Tensor) -> float:
    tau_sum, valid = 0.0, 0
    for i in range(pred_scores.shape[0]):
        p = pred_scores[i].cpu().numpy()
        t = true_scores[i].cpu().numpy()
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            continue
        tau, _ = stats.kendalltau(p, t)
        if not np.isnan(tau):
            tau_sum += tau
            valid += 1
    return tau_sum / max(valid, 1)


def compute_ranking_metrics(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
) -> Dict[str, float]:
    metrics = {
        "ndcg@1": ndcg_at_k(pred_scores, true_scores, k=1),
        "ndcg@3": ndcg_at_k(pred_scores, true_scores, k=3),
        "ndcg@5": ndcg_at_k(pred_scores, true_scores, k=5),
        "map": mean_average_precision(pred_scores, true_scores),
        "mrr": mean_reciprocal_rank(pred_scores, true_scores),
        "spearman": spearman_correlation(pred_scores, true_scores),
        "kendall_tau": kendall_tau(pred_scores, true_scores),
        "pairwise_accuracy": mean_pairwise_accuracy(true_scores, pred_scores),
    }
    metrics.update(score_separation_metrics(pred_scores, true_scores))
    return metrics
