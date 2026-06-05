#!/usr/bin/env python3
"""Train encoder reward model (Hydra entry point)."""

from __future__ import annotations

import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Repo root on path for `scripts` when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from explrank.data.ds_critique import load_ds_critique_ranking
from explrank.data.graded_dataset import GradedExplanationDataset
from explrank.models.reward_model import EncoderRewardModel
from explrank.training.callbacks import WandbCallback
from explrank.training.trainer import RankingTrainer, build_dataloader
from explrank.utils.logging import setup_logging
from explrank.utils.seed import set_seed


def load_data(cfg: DictConfig):
    name = cfg.data.name
    if name == "ds_critique":
        train, val = load_ds_critique_ranking(
            cache_dir=cfg.data.cache_dir,
            min_candidates=cfg.data.get("min_candidates_per_query", 3),
            seed=cfg.seed,
        )
    elif name == "esnli":
        from datasets import load_from_disk

        ds = load_from_disk(cfg.data.data_dir)
        train, val = [], []
        for split, out in [("train", train), ("validation", val)]:
            groups = {}
            for row in ds[split]:
                qid = row.get("query_id", row.get("premise", ""))
                if qid not in groups:
                    groups[qid] = {"query": row.get("premise", ""), "explanations": [], "scores": []}
                label_map = {"entailment": 3, "neutral": 2, "contradiction": 1}
                groups[qid]["explanations"].append(row.get("gold_explanation", ""))
                groups[qid]["scores"].append(float(label_map.get(row.get("label", ""), 0)))
            out.extend(
                [
                    {"query": g["query"], "explanations": g["explanations"], "scores": g["scores"]}
                    for g in groups.values()
                    if len(g["explanations"]) >= cfg.data.get("min_candidates", 1)
                ]
            )
    else:
        raise ValueError(f"Unknown dataset: {name}")
    norm = cfg.data.get("normalize_scores", True)
    scale = cfg.data.get("score_scale", 5.0)
    train_ds = GradedExplanationDataset(train, normalize=norm, score_scale=scale)
    val_ds = GradedExplanationDataset(val, normalize=norm, score_scale=scale)
    return train_ds, val_ds


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    train_ds, val_ds = load_data(cfg)
    train_loader = build_dataloader(train_ds, cfg.batch_size, shuffle=True)
    val_loader = build_dataloader(val_ds, cfg.batch_size, shuffle=False)

    model = EncoderRewardModel(
        base_model=cfg.model.encoder,
        dropout=cfg.model.dropout,
        use_quantization=cfg.model.get("use_quantization", False),
    )
    loss_kwargs = {k: v for k, v in cfg.loss.items() if k != "name"}
    trainer = RankingTrainer(
        model, cfg.loss.name, device, learning_rate=cfg.learning_rate, loss_kwargs=loss_kwargs
    )
    wandb_cb = WandbCallback(
        cfg.get("wandb_project", "explrank"),
        OmegaConf.to_container(cfg, resolve=True),
        enabled=cfg.get("use_wandb", False),
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    trainer.fit(
        train_loader,
        val_loader,
        num_epochs=cfg.num_epochs,
        output_dir=cfg.output_dir,
        max_length=cfg.data.get("max_length", 256),
        early_stopping_metric=cfg.get("early_stopping_metric", "separation_ratio_std"),
        patience=cfg.get("early_stopping_patience", 3),
        wandb_cb=wandb_cb,
    )
    wandb_cb.finish()


if __name__ == "__main__":
    main()
