from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch import Tensor


def _validate(scores: Tensor, targets: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if scores.shape != targets.shape or scores.shape != mask.shape:
        raise ValueError(
            f"scores, targets and mask must share a shape; got "
            f"{scores.shape}, {targets.shape}, {mask.shape}"
        )
    if scores.ndim != 2:
        raise ValueError(f"ranking losses expect [batch, candidates], got {scores.shape}")
    mask = mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("every group needs at least one unmasked candidate")
    return scores, targets.to(scores.dtype), mask


def masked_mse(scores: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    scores, targets, mask = _validate(scores, targets, mask)
    errors = (scores - targets).square().masked_select(mask)
    return errors.mean()


def listnet_loss(
    scores: Tensor,
    targets: Tensor,
    mask: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    scores, targets, mask = _validate(scores, targets, mask)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    minimum = torch.finfo(scores.dtype).min
    predicted = (scores / temperature).masked_fill(~mask, minimum)
    reference = (targets / temperature).masked_fill(~mask, minimum)
    reference_probability = functional.softmax(reference, dim=-1)
    log_probability = functional.log_softmax(predicted, dim=-1)
    return -(reference_probability * log_probability).masked_fill(~mask, 0.0).sum(dim=-1).mean()


def ranknet_loss(
    scores: Tensor,
    targets: Tensor,
    mask: Tensor,
    tie_epsilon: float = 1e-8,
) -> Tensor:
    scores, targets, mask = _validate(scores, targets, mask)
    group_losses: list[Tensor] = []
    for row_scores, row_targets, row_mask in zip(scores, targets, mask):
        valid_scores = row_scores[row_mask]
        valid_targets = row_targets[row_mask]
        if valid_scores.numel() < 2:
            continue
        upper = torch.triu_indices(valid_scores.numel(), valid_scores.numel(), offset=1, device=scores.device)
        target_delta = valid_targets[upper[0]] - valid_targets[upper[1]]
        informative = target_delta.abs() > tie_epsilon
        if not informative.any():
            continue
        score_delta = valid_scores[upper[0]] - valid_scores[upper[1]]
        labels = (target_delta > 0).to(scores.dtype)
        group_losses.append(
            functional.binary_cross_entropy_with_logits(
                score_delta[informative], labels[informative], reduction="mean"
            )
        )
    if not group_losses:
        return scores.sum() * 0.0
    return torch.stack(group_losses).mean()


def lambdarank_loss(
    scores: Tensor,
    targets: Tensor,
    mask: Tensor,
    k: int = 5,
    tie_epsilon: float = 1e-8,
) -> Tensor:
    """Pairwise logistic loss weighted by the absolute delta-NDCG.

    Predicted ranks are detached.  Gradients only flow through pairwise score
    differences, which avoids the broken in-place/rank gradient used by the legacy
    implementation.
    """

    scores, targets, mask = _validate(scores, targets, mask)
    group_losses: list[Tensor] = []
    for row_scores, row_targets, row_mask in zip(scores, targets, mask):
        valid_scores = row_scores[row_mask]
        valid_targets = row_targets[row_mask]
        count = valid_scores.numel()
        if count < 2:
            continue
        upper = torch.triu_indices(count, count, offset=1, device=scores.device)
        target_delta = valid_targets[upper[0]] - valid_targets[upper[1]]
        informative = target_delta.abs() > tie_epsilon
        if not informative.any():
            continue

        gains = torch.pow(2.0, valid_targets) - 1.0
        ideal_order = torch.argsort(valid_targets, descending=True)
        cutoff = min(k, count)
        discounts = 1.0 / torch.log2(
            torch.arange(2, count + 2, device=scores.device, dtype=scores.dtype)
        )
        ideal_dcg = (gains[ideal_order[:cutoff]] * discounts[:cutoff]).sum().clamp_min(1e-12)

        predicted_order = torch.argsort(valid_scores.detach(), descending=True)
        predicted_ranks = torch.empty_like(predicted_order)
        predicted_ranks[predicted_order] = torch.arange(count, device=scores.device)
        item_discount = torch.where(
            predicted_ranks < cutoff,
            discounts[predicted_ranks],
            torch.zeros_like(discounts[predicted_ranks]),
        )
        delta_ndcg = (
            (gains[upper[0]] - gains[upper[1]]).abs()
            * (item_discount[upper[0]] - item_discount[upper[1]]).abs()
            / ideal_dcg
        ).detach()

        score_delta = valid_scores[upper[0]] - valid_scores[upper[1]]
        labels = (target_delta > 0).to(scores.dtype)
        pair_loss = functional.binary_cross_entropy_with_logits(score_delta, labels, reduction="none")
        weights = delta_ndcg[informative]
        group_losses.append((pair_loss[informative] * weights).sum() / weights.sum().clamp_min(1e-12))
    if not group_losses:
        return scores.sum() * 0.0
    return torch.stack(group_losses).mean()


LOSSES: dict[str, Callable[[Tensor, Tensor, Tensor], Tensor]] = {
    "mse": masked_mse,
    "listnet": listnet_loss,
    "ranknet": ranknet_loss,
    "lambdarank": lambdarank_loss,
}


def get_loss(name: str) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    try:
        return LOSSES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown loss {name!r}; choose one of {sorted(LOSSES)}") from exc
