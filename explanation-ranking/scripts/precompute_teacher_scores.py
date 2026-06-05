#!/usr/bin/env python3
"""Precompute teacher (encoder) scores for distillation."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from explrank.data.ds_critique import load_ds_critique_ranking
from explrank.data.graded_dataset import GradedExplanationDataset
from explrank.models.reward_model import EncoderRewardModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--encoder", default="microsoft/deberta-v3-base")
    p.add_argument("--cache_dir", default="./data/ds_critique_bank")
    p.add_argument("--output", default="./outputs/teacher_scores.jsonl")
    p.add_argument("--split", choices=["train", "val"], default="train")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = load_ds_critique_ranking(cache_dir=args.cache_dir)
    examples = train if args.split == "train" else val
    ds = GradedExplanationDataset(examples)

    model = EncoderRewardModel(base_model=args.encoder)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fout:
        for ex in tqdm(ds.examples):
            row = {
                "query_id": ex.get("query_id", ""),
                "query": ex["query"],
                "explanations": list(ex["explanations"]),
                "teacher_scores": [],
            }
            for expl in ex["explanations"]:
                sc = model.score_batch([ex["query"]], [expl], device)
                row["teacher_scores"].append(float(sc[0].cpu()))
            fout.write(json.dumps(row) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
