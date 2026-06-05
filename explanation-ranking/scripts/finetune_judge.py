"""Judge fine-tuning entry point — the main ACML experiment.

RECONSTRUCTED from status-doc spec — verify against your
scripts/finetune_with_distillation.py. Handles all ablations via flags.

Pipeline (your four RQs):
  RQ1  baseline: measure judge sep ratio BEFORE fine-tuning (viability already
       gave 0.164 absolute / 0.409 CoT for Llama-3-8B; this re-confirms on the
       same split the fine-tune uses).
  RQ3  fine-tune with --loss {mse,bradley_terry,ranknet,approxndcg,listnet}
       on HUMAN-ANNOTATED e-SNLI ONLY (anti-circularity), optionally + distill.
  Then sep ratio AFTER -> does ranking lift it past 0.8?

Anti-circularity constraints baked in:
  * --train_data must be human-annotated e-SNLI (no GPT-4 scores). Asserted.
  * Distillation teacher is the encoder RM, not GPT-4.

Ablation flags:
  --loss            which objective (the 5-way control)
  --lambda_distill  0 = ranking only; >0 = Signal 2 active
  --seed            42 / 1 / 2

Local smoke (Mac, mps):
  python scripts/finetune_judge.py --loss listnet --max_queries 64 \
      --epochs 1 --device auto --judge_model meta-llama/Llama-3.1-8B \
      --no_4bit   # mps can't do bitsandbytes 4-bit; use fp16/bf16 small slice

Cluster (IDRIS, after smoke passes):
  sbatch slurm/judge_finetune_array.sh
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Repo root on path for `scripts` when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from explrank.losses.distillation import CombinedRankingDistillLoss
from explrank.losses.ranking import LOSS_REGISTRY
from explrank.metrics import compute_ranking_metrics
from explrank.utils.device import resolve_device
from explrank.utils.seed import set_seed


def load_judge(model_name, device, load_in_4bit):
    # RECONSTRUCTED: build LLMJudgeRewardModel; on mps force load_in_4bit=False
    from explrank.models.llm_judge import LLMJudgeRewardModel
    return LLMJudgeRewardModel(model_name, load_in_4bit=load_in_4bit).to(device)


def load_human_esnli(max_queries=None):
    return _load_human_esnli_from_source("esnli_human", max_queries=max_queries)


def _label_to_score(label):
    if isinstance(label, (int, np.integer)):
        return float({0: 3.0, 1: 2.0, 2: 1.0}.get(int(label), float(label)))
    if isinstance(label, str):
        norm = label.strip().lower()
        label_map = {"entailment": 3.0, "neutral": 2.0, "contradiction": 1.0}
        if norm in label_map:
            return label_map[norm]
        try:
            return float(norm)
        except ValueError:
            return 0.0
    return 0.0


def _normalize_grouped_row(row):
    exps = row.get("explanations")
    scores = row.get("scores")
    if not isinstance(exps, list) or not isinstance(scores, list):
        return None
    if len(exps) != len(scores) or len(exps) < 2:
        return None
    query = row.get("query", row.get("premise", ""))
    query_id = row.get("query_id", query)
    clean = [(str(e).strip(), float(s)) for e, s in zip(exps, scores) if str(e).strip() != ""]
    if len(clean) < 2:
        return None
    return {
        "query_id": query_id,
        "query": query,
        "explanations": [e for e, _ in clean],
        "scores": [s for _, s in clean],
    }


def _normalize_esnli_row(row):
    premise = str(row.get("premise", "")).strip()
    hypothesis = str(row.get("hypothesis", "")).strip()
    if not premise and not hypothesis:
        return None
    query = f"Premise: {premise}\nHypothesis: {hypothesis}".strip()
    query_id = row.get("query_id", f"{premise} || {hypothesis}")

    explanations = []
    for k in ("explanation_1", "explanation_2", "explanation_3", "gold_explanation", "explanation"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            explanations.append(v.strip())
        elif isinstance(v, list):
            explanations.extend([str(x).strip() for x in v if str(x).strip()])
    # Deduplicate while preserving order
    explanations = list(dict.fromkeys(explanations))
    if len(explanations) < 2:
        return None

    score = _label_to_score(row.get("label"))
    return {
        "query_id": query_id,
        "query": query,
        "explanations": explanations,
        "scores": [score] * len(explanations),
    }


def _load_human_esnli_from_source(source, max_queries=None):
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    if source in {"esnli_human", "esnli"}:
        ds = load_dataset("esnli", split="train", trust_remote_code=True)
    else:
        p = Path(source)
        if p.exists() and p.is_dir():
            loaded = load_from_disk(str(p))
            if isinstance(loaded, DatasetDict):
                split = "train" if "train" in loaded else list(loaded.keys())[0]
                ds = loaded[split]
            else:
                ds = loaded
        elif p.exists() and p.suffix.lower() in {".json", ".jsonl", ".parquet", ".csv"}:
            if p.suffix.lower() == ".jsonl":
                ds = load_dataset("json", data_files=str(p), split="train")
            elif p.suffix.lower() == ".json":
                ds = load_dataset("json", data_files=str(p), split="train")
            elif p.suffix.lower() == ".parquet":
                ds = load_dataset("parquet", data_files=str(p), split="train")
            else:
                ds = load_dataset("csv", data_files=str(p), split="train")
        else:
            ds = load_dataset(source, split="train")

    if not isinstance(ds, Dataset):
        raise ValueError("Failed to load human e-SNLI data as a Dataset")

    rows = []
    for row in ds:
        normalized = _normalize_grouped_row(row) or _normalize_esnli_row(row)
        if normalized is None:
            continue
        rows.append(normalized)
        if max_queries is not None and len(rows) >= max_queries:
            break
    if not rows:
        raise ValueError("No valid human e-SNLI groups found in train_data")
    return rows


def load_teacher_scores(path):
    # MIGRATE: precomputed encoder-RM scores keyed by (query_id, cand_idx).
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--loss", default="listnet", choices=list(LOSS_REGISTRY))
    ap.add_argument("--lambda_distill", type=float, default=0.0)
    ap.add_argument("--teacher_scores", default=None,
                    help="precomputed encoder-RM scores (enables Signal 2)")
    ap.add_argument("--train_data", default="esnli_human",
                    help="MUST be human-annotated (anti-circularity)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_4bit", action="store_true", help="set on mps/local")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Anti-circularity guard
    assert "gpt" not in args.train_data.lower(), (
        "train_data must be human-annotated, not GPT-4 scored (circularity)"
    )

    set_seed(args.seed)
    device = resolve_device(args.device)
    judge = load_judge(args.judge_model, device, load_in_4bit=not args.no_4bit)

    data = _load_human_esnli_from_source(args.train_data, max_queries=args.max_queries)
    teacher = load_teacher_scores(args.teacher_scores)

    # RQ1: sep ratio BEFORE fine-tuning -------------------------------------
    before = measure(judge, data, device)
    print(f"[RQ1] before fine-tune: sep={before['separation_ratio_std']:.3f} "
          f"spearman={before['spearman']:.3f} pairwise={before['pairwise_accuracy']:.3f}")

    # RQ3: fine-tune --------------------------------------------------------
    rank_fn = LOSS_REGISTRY[args.loss]
    objective = CombinedRankingDistillLoss(
        rank_loss_fn=rank_fn, lambda_distill=args.lambda_distill
    )
    opt = torch.optim.AdamW(judge.parameters(), lr=args.lr)
    judge.train()
    for epoch in range(args.epochs):
        for ex in data:  # one query group at a time (listwise)
            scores = score_group(judge, ex, device)         # (K,)
            gold = torch.tensor(ex["scores"], device=device)
            ts = teacher_for(ex, teacher, device)            # (K,) or None
            out = objective(scores, gold, ts)
            opt.zero_grad(); out["loss"].backward(); opt.step()

    after = measure(judge, data, device)
    lifted = after["separation_ratio_std"] > 0.8
    print(f"[RQ3] after  fine-tune: sep={after['separation_ratio_std']:.3f} "
          f"spearman={after['spearman']:.3f} pairwise={after['pairwise_accuracy']:.3f}")
    print(f"[RQ3] sep lifted past 0.8? {lifted}")

    result = {"loss": args.loss, "lambda_distill": args.lambda_distill,
              "seed": args.seed, "before": before, "after": after,
              "lifted_past_0.8": bool(lifted)}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")


# --- helpers (MIGRATE: align with your data/model exact interfaces) --------

def score_group(judge, ex, device):
    texts = [f"{ex['query']} {judge.tokenizer.sep_token} {e}"
             if judge.tokenizer.sep_token else f"{ex['query']} {e}"
             for e in ex["explanations"]]
    enc = judge.tokenizer(texts, padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to(device)
    return judge(enc["input_ids"], enc["attention_mask"])


def teacher_for(ex, teacher, device):
    if teacher is None:
        return None
    key = ex.get("query_id", ex["query"])
    vals = [teacher.get(f"{key}:{i}") for i in range(len(ex["explanations"]))]
    if any(v is None for v in vals):
        return None
    return torch.tensor(vals, device=device)


@torch.no_grad()
def measure(judge, data, device):
    judge.eval()
    preds, golds = [], []
    for ex in data:
        if len(ex["explanations"]) < 2:
            continue
        preds.append(score_group(judge, ex, device).cpu())
        golds.append(torch.tensor(ex["scores"]))
    judge.train()
    # pad to [N,k] by truncating to min k for the vectorized metrics
    k = min(p.numel() for p in preds)
    P = torch.stack([p[:k] for p in preds])
    G = torch.stack([g[:k] for g in golds])
    return compute_ranking_metrics(P, G)


if __name__ == "__main__":
    main()
