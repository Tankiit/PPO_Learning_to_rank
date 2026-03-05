"""
Ranking Loss Functions for Explanation Quality Assessment.

Implements all 5 losses from the paper:
  - MSE (pointwise baseline)
  - Binary/BradleyTerry (pairwise preference — DPO foundation)
  - RankNet (pairwise ranking)
  - ApproxNDCG (listwise metric-approximation)
  - ListNet (listwise distributional — Plackett-Luce)

Each loss takes:
  - pred_scores: [batch, k] predicted scores for k candidates per query
  - true_scores: [batch, k] ground-truth quality scores

Usage:
    loss_fn = get_loss_function("listnet")
    loss = loss_fn(pred_scores, true_scores)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Pointwise
# =============================================================================

class MSELoss(nn.Module):
    """Pointwise MSE — baseline that causes score compression."""
    def forward(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_scores, true_scores)


# =============================================================================
# Pairwise
# =============================================================================

class BradleyTerryLoss(nn.Module):
    """
    Binary preference loss (Bradley-Terry model).
    Mathematical foundation of DPO (Rafailov et al., 2023).
    
    For each pair (i,j) where true_i > true_j:
        loss = -log(sigmoid(pred_i - pred_j))
    """
    def forward(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        batch_size, k = pred_scores.shape
        total_loss = 0.0
        n_pairs = 0
        
        for b in range(batch_size):
            for i in range(k):
                for j in range(i + 1, k):
                    if true_scores[b, i] > true_scores[b, j]:
                        diff = pred_scores[b, i] - pred_scores[b, j]
                        total_loss += -F.logsigmoid(diff)
                        n_pairs += 1
                    elif true_scores[b, j] > true_scores[b, i]:
                        diff = pred_scores[b, j] - pred_scores[b, i]
                        total_loss += -F.logsigmoid(diff)
                        n_pairs += 1
        
        return total_loss / max(n_pairs, 1)


class RankNetLoss(nn.Module):
    """
    RankNet (Burges et al., 2005).
    Pairwise: P(i > j) = sigmoid(s_i - s_j)
    
    Identical formula to BradleyTerry but typically used with
    continuous scores rather than binary preferences.
    """
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        # Vectorized pairwise computation
        # pred_diff[b, i, j] = pred[b,i] - pred[b,j]
        pred_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)  # [B, k, k]
        true_diff = true_scores.unsqueeze(2) - true_scores.unsqueeze(1)  # [B, k, k]
        
        # S_ij = 1 if true_i > true_j, -1 if true_i < true_j, 0 if equal
        S_ij = torch.sign(true_diff)
        
        # RankNet loss: -S_ij * sigma * pred_diff + log(1 + exp(sigma * pred_diff))
        loss = (0.5 * (1.0 - S_ij) * self.sigma * pred_diff
                + torch.log1p(torch.exp(-self.sigma * pred_diff)))
        
        # Mask diagonal and equal-score pairs
        mask = (S_ij != 0).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        
        return loss


# =============================================================================
# Listwise
# =============================================================================

class ListNetLoss(nn.Module):
    """
    ListNet (Cao et al., 2007) — Plackett-Luce probability model.
    
    KL divergence between ground-truth and predicted top-1 probability:
        P_y(i) = exp(y_i) / sum_j exp(y_j)
        P_f(i) = exp(f(i)) / sum_j exp(f(j))
        L = -sum_i P_y(i) * log(P_f(i))
    
    This is the loss that achieves 0.920 separation ratio (Table 2).
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        # Top-1 probability distributions
        P_true = F.softmax(true_scores / self.temperature, dim=-1)
        P_pred = F.log_softmax(pred_scores / self.temperature, dim=-1)
        
        # Cross-entropy (equivalent to KL up to constant)
        loss = -(P_true * P_pred).sum(dim=-1).mean()
        
        return loss


class ApproxNDCGLoss(nn.Module):
    """
    ApproxNDCG (Bruch et al., 2019).
    Differentiable approximation of NDCG using sigmoid smoothing.
    
    Approximates position via:
        approx_rank(i) = 1 + sum_{j!=i} sigmoid((s_j - s_i) / sigma)
    """
    def __init__(self, sigma: float = 1.0, k: Optional[int] = None):
        super().__init__()
        self.sigma = sigma
        self.k = k
    
    def forward(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        batch_size, n = pred_scores.shape
        
        # Compute approximate ranks using sigmoid
        # diff[b, i, j] = pred[b,j] - pred[b,i]
        diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(2)  # [B, n, n]
        approx_ranks = 1.0 + torch.sigmoid(diff / self.sigma).sum(dim=-1) - 0.5  # [B, n]
        
        # Compute gains: 2^relevance - 1
        gains = torch.pow(2.0, true_scores) - 1.0
        
        # Compute discounts: 1 / log2(rank + 1)
        discounts = 1.0 / torch.log2(approx_ranks + 1.0)
        
        # Approximate DCG
        approx_dcg = (gains * discounts).sum(dim=-1)  # [B]
        
        # Ideal DCG (for normalization)
        sorted_gains, _ = true_scores.sort(dim=-1, descending=True)
        ideal_gains = torch.pow(2.0, sorted_gains) - 1.0
        ideal_positions = torch.arange(1, n + 1, dtype=torch.float32, device=pred_scores.device)
        ideal_discounts = 1.0 / torch.log2(ideal_positions + 1.0)
        ideal_dcg = (ideal_gains * ideal_discounts).sum(dim=-1)  # [B]
        
        # Approximate NDCG (negate for minimization)
        approx_ndcg = approx_dcg / ideal_dcg.clamp(min=1e-8)
        
        return 1.0 - approx_ndcg.mean()


class LambdaRankLoss(nn.Module):
    """
    LambdaRank gradient weighting.
    
    Weight pairwise gradients by |delta_NDCG| — the change in NDCG
    from swapping positions of items i and j.
    """
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        batch_size, n = pred_scores.shape
        
        # Sort by predicted scores to get current ranking
        _, pred_order = pred_scores.sort(dim=-1, descending=True)
        
        # Compute gains
        gains = torch.pow(2.0, true_scores) - 1.0
        
        # Pairwise score differences
        pred_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)  # [B, n, n]
        true_diff = true_scores.unsqueeze(2) - true_scores.unsqueeze(1)
        
        # Only consider pairs where true_i > true_j
        S_ij = (true_diff > 0).float()
        
        # Compute |delta_NDCG| for each pair
        # Approximate: use gain differences as proxy
        gain_diff = torch.abs(gains.unsqueeze(2) - gains.unsqueeze(1))
        
        # Positions (approximate from predicted scores)
        ranks = pred_scores.argsort(dim=-1, descending=True).argsort(dim=-1).float() + 1.0
        discount_i = 1.0 / torch.log2(ranks.unsqueeze(2) + 1.0)
        discount_j = 1.0 / torch.log2(ranks.unsqueeze(1) + 1.0)
        delta_ndcg = torch.abs(gain_diff * (discount_i - discount_j))
        
        # Lambda = |delta_NDCG| / (1 + exp(s_i - s_j))
        lambdas = delta_ndcg / (1.0 + torch.exp(self.sigma * pred_diff))
        
        loss = (S_ij * lambdas).sum() / S_ij.sum().clamp(min=1)
        
        return loss


# =============================================================================
# Factory
# =============================================================================

LOSS_REGISTRY = {
    "mse": MSELoss,
    "binary": BradleyTerryLoss,
    "bradley_terry": BradleyTerryLoss,
    "ranknet": RankNetLoss,
    "listnet": ListNetLoss,
    "approxndcg": ApproxNDCGLoss,
    "lambdarank": LambdaRankLoss,
}


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Factory to get loss by name.
    
    Args:
        name: one of 'mse', 'binary', 'ranknet', 'listnet', 'approxndcg', 'lambdarank'
        **kwargs: passed to loss constructor (e.g., sigma, temperature)
    
    Returns:
        Loss module
    """
    name = name.lower().strip()
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)
