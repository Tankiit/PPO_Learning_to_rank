"""Epoch training loop with early stopping on separation ratio."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from explrank.data.collate import collate_by_query
from explrank.losses.ranking import get_loss_function
from explrank.metrics.ranking import compute_ranking_metrics
from explrank.models.reward_model import EncoderRewardModel
from explrank.training.callbacks import WandbCallback, check_score_consistency


class RankingTrainer:
    def __init__(
        self,
        model: EncoderRewardModel,
        loss_name: str,
        device: torch.device,
        learning_rate: float = 2e-5,
        loss_kwargs: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = get_loss_function(loss_name, **(loss_kwargs or {}))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _forward_query(
        self, query: str, explanations: List[str], max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.model.encode_pairs(
            [query] * len(explanations), explanations, max_length=max_length
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        pred = self.model(**enc)
        return pred, enc

    def train_epoch(
        self,
        loader: DataLoader,
        max_length: int = 256,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in tqdm(loader, desc="train", leave=False):
            for query, exps, scores in zip(
                batch["queries"], batch["explanations"], batch["scores"]
            ):
                pred, _ = self._forward_query(query, exps, max_length)
                gold = scores.to(self.device)
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                    gold = gold.unsqueeze(0)
                loss = self.loss_fn(pred.unsqueeze(0), gold.unsqueeze(0))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        max_length: int = 256,
    ) -> Dict[str, float]:
        self.model.eval()
        all_pred, all_gold = [], []
        for batch in loader:
            for query, exps, scores in zip(
                batch["queries"], batch["explanations"], batch["scores"]
            ):
                pred, _ = self._forward_query(query, exps, max_length)
                k = len(exps)
                if pred.numel() != k:
                    pred = pred.view(-1)[:k]
                all_pred.append(pred.cpu())
                all_gold.append(scores)
        if not all_pred:
            return {}
        max_k = max(p.numel() for p in all_pred)
        pred_pad = torch.zeros(len(all_pred), max_k)
        gold_pad = torch.zeros(len(all_gold), max_k)
        for i, (p, g) in enumerate(zip(all_pred, all_gold)):
            pred_pad[i, : p.numel()] = p
            gold_pad[i, : g.numel()] = g
        check_score_consistency(pred_pad, gold_pad)
        return compute_ranking_metrics(pred_pad, gold_pad)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        output_dir: str,
        *,
        max_length: int = 256,
        early_stopping_metric: str = "separation_ratio_std",
        patience: int = 3,
        wandb_cb: Optional[WandbCallback] = None,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        best_val = -float("inf")
        wait = 0
        history: List[Dict] = []

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, max_length=max_length)
            val_metrics = self.evaluate(val_loader, max_length=max_length)
            val_metrics["train_loss"] = train_loss
            history.append(val_metrics)
            if wandb_cb:
                wandb_cb.log({**val_metrics, "epoch": epoch}, step=epoch)

            score = val_metrics.get(early_stopping_metric, 0.0)
            if score > best_val:
                best_val = score
                wait = 0
                torch.save(self.model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            else:
                wait += 1
                if wait >= patience:
                    break

        with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
            json.dump(history, f, indent=2)
        return {"history": history, "best": best_val}


def build_dataloader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_by_query,
    )
