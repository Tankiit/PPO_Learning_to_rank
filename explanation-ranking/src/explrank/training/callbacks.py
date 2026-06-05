"""Training callbacks: consistency checks and optional W&B."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def check_score_consistency(
    pred: torch.Tensor,
    gold: torch.Tensor,
    *,
    warn_sep_below: float = 0.05,
) -> None:
    if pred.shape != gold.shape:
        raise ValueError(f"pred {pred.shape} != gold {gold.shape}")
    if torch.isnan(pred).any() or torch.isnan(gold).any():
        raise ValueError("NaN in pred or gold scores")
    pred_std = pred.std().item()
    if pred_std < warn_sep_below:
        logger.warning("Very low pred std (%.4f) — possible score collapse", pred_std)


class WandbCallback:
    def __init__(self, project: str, config: Dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        self.run = None
        if not enabled:
            return
        try:
            import wandb

            self.run = wandb.init(project=project, config=config)
        except ImportError:
            logger.warning("wandb not installed; disabling logging")
            self.enabled = False

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self.enabled and self.run:
            import wandb

            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self.run:
            import wandb

            wandb.finish()
