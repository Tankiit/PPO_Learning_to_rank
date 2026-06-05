"""Ranking losses + MSE/Bradley-Terry baselines.

RECONSTRUCTED — verify ranknet/lambdarank/approxndcg against your src/losses.py.
All take (pred_scores, gold_scores), both shape (K,) for one query's K candidates.

The MSE and bradley_terry entries are the CONTROLS that show compression is a
loss-function property, not a model-family property (your status-doc decision).
"""

import torch
import torch.nn.functional as F


def listnet_loss(pred_scores, gold_scores):
    """Top-1 ListNet: cross-entropy between softmax(gold) and softmax(pred)."""
    return -torch.sum(F.softmax(gold_scores.float(), dim=0) * F.log_softmax(pred_scores, dim=0))


def ranknet_loss(pred_scores, gold_scores):
    """Pairwise logistic over all i<j with gold_i != gold_j.
    RECONSTRUCTED — verify margin/reduction against your version."""
    K = pred_scores.shape[0]
    loss, n = pred_scores.new_zeros(()), 0
    for i in range(K):
        for j in range(K):
            if gold_scores[i] > gold_scores[j]:
                loss = loss - F.logsigmoid(pred_scores[i] - pred_scores[j])
                n += 1
    return loss / max(n, 1)


def lambdarank_loss(pred_scores, gold_scores):
    """RankNet gradient weighted by |delta-NDCG|. RECONSTRUCTED — placeholder
    weights NDCG swap by gain difference; verify against your version."""
    K = pred_scores.shape[0]
    loss, n = pred_scores.new_zeros(()), 0
    for i in range(K):
        for j in range(K):
            if gold_scores[i] > gold_scores[j]:
                delta = torch.abs(gold_scores[i] - gold_scores[j]).float()
                loss = loss - delta * F.logsigmoid(pred_scores[i] - pred_scores[j])
                n += 1
    return loss / max(n, 1)


def approxndcg_loss(pred_scores, gold_scores, alpha=10.0):
    """Smooth NDCG via soft ranks. RECONSTRUCTED — verify against your version."""
    diff = pred_scores.unsqueeze(0) - pred_scores.unsqueeze(1)
    soft_rank = 1.0 + torch.sigmoid(-alpha * diff).sum(dim=1)
    gains = torch.pow(2.0, gold_scores.float()) - 1.0
    discounts = torch.log2(soft_rank + 1.0)
    dcg = (gains / discounts).sum()
    ideal_disc = torch.log2(torch.arange(2, len(gold_scores) + 2, dtype=torch.float32,
                                         device=pred_scores.device))
    idcg = (torch.sort(gains, descending=True).values / ideal_disc).sum()
    return -(dcg / idcg.clamp(min=1e-8))


def mse_loss(pred_scores, gold_scores):
    """Pointwise regression baseline. The compression control."""
    return F.mse_loss(pred_scores, gold_scores.float())


def bradley_terry_loss(pred_scores, gold_scores):
    """Binary preference (BT) baseline: same as RankNet pairwise but the
    canonical 'pairwise preference' control your doc names explicitly."""
    return ranknet_loss(pred_scores, gold_scores)


LOSS_REGISTRY = {
    "listnet": listnet_loss,
    "ranknet": ranknet_loss,
    "lambdarank": lambdarank_loss,
    "approxndcg": approxndcg_loss,
    "mse": mse_loss,
    "bradley_terry": bradley_terry_loss,
}


def get_loss_function(name):
    """Compatibility helper used by trainer/tests."""
    key = str(name).lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[key]
