#!/usr/bin/env python3
"""Fine-tune student model with teacher score distillation."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from explrank.data.collate import collate_by_query
from explrank.data.graded_dataset import GradedExplanationDataset
from explrank.losses.distillation import score_distillation_loss
from explrank.models.llm_judge import LLMJudgeRewardModel
from explrank.models.reward_model import EncoderRewardModel
from explrank.utils.seed import set_seed


def load_teacher_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_scores", required=True)
    p.add_argument("--train_json", help="optional ranking JSON; uses teacher file queries if omitted")
    p.add_argument("--encoder", default="distilroberta-base", help="student backbone")
    p.add_argument("--teacher_checkpoint", default=None, help="optional teacher .pt for from_teacher init")
    p.add_argument("--teacher_encoder", default="microsoft/deberta-v3-base")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--output_dir", default="./outputs/distill")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_teacher_jsonl(args.teacher_scores)
    examples = []
    for row in rows:
        exps = row.get("explanations")
        if not exps:
            raise ValueError(
                f"query_id={row.get('query_id')}: missing 'explanations' in teacher JSONL. "
                "Re-run precompute_teacher_scores.py with the updated script."
            )
        if len(exps) != len(row["teacher_scores"]):
            raise ValueError(f"query_id={row.get('query_id')}: explanations vs teacher_scores length mismatch")
        examples.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "explanations": exps,
                "scores": row["teacher_scores"],
            }
        )

    ds = GradedExplanationDataset(examples, normalize=False, min_candidates=1)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_by_query)

    if args.teacher_checkpoint:
        teacher = EncoderRewardModel(base_model=args.teacher_encoder)
        teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
        student = LLMJudgeRewardModel.from_teacher(
            teacher,
            student_encoder=args.encoder,
            freeze_encoder=args.freeze_encoder,
        ).to(device)
    else:
        student = LLMJudgeRewardModel(
            base_model=args.encoder, freeze_encoder=args.freeze_encoder
        ).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        for batch in loader:
            for query, exps, t_list in zip(
                batch["queries"], batch["explanations"], batch["scores"]
            ):
                t_scores = t_list.to(device)
                if len(exps) != t_scores.numel():
                    continue
                pred = student.score_batch([query] * len(exps), exps, device)
                loss = score_distillation_loss(pred.unsqueeze(0), t_scores.unsqueeze(0))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
        print(f"epoch {epoch} loss {total / max(n, 1):.4f}")

    student.save_pretrained(os.path.join(args.output_dir, "student.pt"))


if __name__ == "__main__":
    main()
