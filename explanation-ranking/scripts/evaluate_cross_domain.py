#!/usr/bin/env python3
"""Zero-shot cross-domain evaluation (train on A, test on B)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from explrank.data.ds_critique import load_ds_critique_ranking
from explrank.data.graded_dataset import GradedExplanationDataset
from explrank.metrics.ranking import compute_ranking_metrics
from explrank.models.reward_model import EncoderRewardModel
from explrank.training.trainer import RankingTrainer, build_dataloader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--encoder", default="microsoft/deberta-v3-base")
    p.add_argument("--cache_dir", default="./data/ds_critique_bank")
    p.add_argument("--output", default="./outputs/cross_domain_metrics.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val = load_ds_critique_ranking(cache_dir=args.cache_dir)
    val_ds = GradedExplanationDataset(val)
    loader = build_dataloader(val_ds, batch_size=8, shuffle=False)

    model = EncoderRewardModel(base_model=args.encoder)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    trainer = RankingTrainer(model, "listnet", device)
    metrics = trainer.evaluate(loader)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
