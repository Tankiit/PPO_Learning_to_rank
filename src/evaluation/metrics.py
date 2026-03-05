"""
Ranking Evaluation Metrics.

Computes all metrics reported in the paper:
  - NDCG@k (k=1,3,5)
  - MAP (Mean Average Precision)
  - MRR (Mean Reciprocal Rank)
  - Spearman ρ
  - Kendall τ
  - Score separation ratio (critical for PPO viability)

All functions accept:
  - pred_scores: [N, k] predicted scores
  - true_scores: [N, k] ground-truth quality scores
"""

import torch
import numpy as np
from scipy import stats
from typing import Dict, Optional


def ndcg_at_k(pred_scores: torch.Tensor, true_scores: torch.Tensor, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    
    For each query, sort candidates by predicted score,
    compute DCG of true scores in that order, normalize by ideal DCG.
    """
    N = pred_scores.shape[0]
    ndcg_sum = 0.0
    
    for i in range(N):
        # Sort by predicted scores (descending)
        _, pred_order = pred_scores[i].sort(descending=True)
        sorted_true = true_scores[i][pred_order][:k]
        
        # DCG
        positions = torch.arange(1, min(k, len(sorted_true)) + 1, dtype=torch.float32)
        discounts = torch.log2(positions + 1.0)
        gains = torch.pow(2.0, sorted_true.float()) - 1.0
        dcg = (gains / discounts).sum().item()
        
        # Ideal DCG
        ideal_sorted, _ = true_scores[i].sort(descending=True)
        ideal_sorted = ideal_sorted[:k]
        ideal_gains = torch.pow(2.0, ideal_sorted.float()) - 1.0
        ideal_dcg = (ideal_gains / discounts[:len(ideal_gains)]).sum().item()
        
        if ideal_dcg > 0:
            ndcg_sum += dcg / ideal_dcg
        else:
            ndcg_sum += 1.0  # Perfect score if no relevant items
    
    return ndcg_sum / N


def mean_average_precision(pred_scores: torch.Tensor, true_scores: torch.Tensor) -> float:
    """
    Mean Average Precision.
    Treat items with above-median true score as "relevant".
    """
    N = pred_scores.shape[0]
    ap_sum = 0.0
    
    for i in range(N):
        _, pred_order = pred_scores[i].sort(descending=True)
        sorted_true = true_scores[i][pred_order]
        
        # Binary relevance: above median = relevant
        threshold = true_scores[i].median()
        relevant = (sorted_true > threshold).float()
        
        if relevant.sum() == 0:
            ap_sum += 1.0
            continue
        
        # Average precision
        cum_relevant = relevant.cumsum(dim=0)
        precisions = cum_relevant / torch.arange(1, len(relevant) + 1, dtype=torch.float32)
        ap = (precisions * relevant).sum() / relevant.sum()
        ap_sum += ap.item()
    
    return ap_sum / N


def mean_reciprocal_rank(pred_scores: torch.Tensor, true_scores: torch.Tensor) -> float:
    """
    Mean Reciprocal Rank.
    Rank of the highest true-score item in the predicted ranking.
    """
    N = pred_scores.shape[0]
    rr_sum = 0.0
    
    for i in range(N):
        _, pred_order = pred_scores[i].sort(descending=True)
        sorted_true = true_scores[i][pred_order]
        
        # Find position of best true item
        best_true_idx = true_scores[i].argmax()
        rank = (pred_order == best_true_idx).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            rr_sum += 1.0 / (rank[0].item() + 1)
    
    return rr_sum / N


def spearman_correlation(pred_scores: torch.Tensor, true_scores: torch.Tensor) -> float:
    """Average Spearman ρ across all queries."""
    N = pred_scores.shape[0]
    rho_sum = 0.0
    valid = 0
    
    for i in range(N):
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
    """Average Kendall τ across all queries."""
    N = pred_scores.shape[0]
    tau_sum = 0.0
    valid = 0
    
    for i in range(N):
        p = pred_scores[i].cpu().numpy()
        t = true_scores[i].cpu().numpy()
        
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            continue
        
        tau, _ = stats.kendalltau(p, t)
        if not np.isnan(tau):
            tau_sum += tau
            valid += 1
    
    return tau_sum / max(valid, 1)


def score_separation_metrics(pred_scores: torch.Tensor, true_scores: torch.Tensor) -> Dict[str, float]:
    """
    Score separation analysis — THE critical metric for PPO viability.
    
    Returns:
        separation_ratio: std(pred) / std(true), averaged across queries
        score_range: mean of (max_pred - min_pred) per query
        score_std: mean std of predicted scores per query
    """
    pred_std_per_query = pred_scores.std(dim=-1)  # [N]
    true_std_per_query = true_scores.std(dim=-1)  # [N]
    
    ranges = pred_scores.max(dim=-1).values - pred_scores.min(dim=-1).values
    
    # Separation ratio: how much of the true variance is preserved
    sep_ratios = pred_std_per_query / true_std_per_query.clamp(min=1e-8)
    
    return {
        "separation_ratio": sep_ratios.mean().item(),
        "score_range": ranges.mean().item(),
        "score_std": pred_std_per_query.mean().item(),
    }


def compute_ranking_metrics(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all ranking metrics from the paper.
    
    Args:
        pred_scores: [N, k] predicted scores
        true_scores: [N, k] ground-truth quality scores
    
    Returns:
        dict with all metrics
    """
    metrics = {
        "ndcg@1": ndcg_at_k(pred_scores, true_scores, k=1),
        "ndcg@3": ndcg_at_k(pred_scores, true_scores, k=3),
        "ndcg@5": ndcg_at_k(pred_scores, true_scores, k=5),
        "map": mean_average_precision(pred_scores, true_scores),
        "mrr": mean_reciprocal_rank(pred_scores, true_scores),
        "spearman": spearman_correlation(pred_scores, true_scores),
        "kendall_tau": kendall_tau(pred_scores, true_scores),
    }
    
    sep = score_separation_metrics(pred_scores, true_scores)
    metrics.update(sep)
    
    return metrics


# =============================================================================
# Bootstrap confidence intervals (for 5-seed runs)
# =============================================================================

def bootstrap_ci(values: list, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    boot_means = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def paired_bootstrap_test(
    scores_a: list,
    scores_b: list,
    n_bootstrap: int = 10000,
) -> tuple:
    """
    Paired bootstrap significance test.
    
    Returns:
        observed_diff: mean(A) - mean(B)
        p_value: probability that A <= B
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b
    observed = np.mean(diff)
    
    count = sum(
        np.mean(np.random.choice(diff, size=len(diff), replace=True)) <= 0
        for _ in range(n_bootstrap)
    )
    
    return float(observed), count / n_bootstrap


def aggregate_seed_results(results_per_seed: list) -> Dict:
    """
    Aggregate results across multiple seeds.
    
    Args:
        results_per_seed: list of metric dicts, one per seed
    
    Returns:
        dict with mean, std, 95% CI for each metric
    """
    all_metrics = {}
    for r in results_per_seed:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, []).append(v)
    
    aggregated = {}
    for k, values in all_metrics.items():
        values_arr = np.array(values)
        ci_low, ci_high = bootstrap_ci(values)
        aggregated[k] = {
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high,
            "values": values,
        }
    
    return aggregated
