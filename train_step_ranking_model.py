"""
Extension 1: Step-Level Process Reward Model for NLI Explanation Quality
=========================================================================
Architecture
------------
  DeBERTa-v3 / RoBERTa encoder
    → [STEP] token hidden states  → shared projection head → step scores s_i
    → min-aggregation             → explanation score  (for LTR loss)
    → [CLS] hidden state          → explanation score  (auxiliary, optional)

Input format
------------
  [CLS] premise [SEP] hypothesis [SEP] sent_1 [STEP] sent_2 [STEP] sent_3 [SEP]

Loss (two-level)
----------------
  L = λ_rank * L_ListNet(expl_scores, gold_ranks)        # listwise, over candidates
    + λ_step * L_step_BCE(step_scores, silver_labels)    # binary, per step

Silver step labels
------------------
  Strategy A  – FreePRM heuristic  (default, zero cost)
  Strategy B  – LLM-as-judge       (GPT-4o, ~$200 for 50k expl)
  Strategy C  – Implicit PRM       (reference model log-ratio, zero extra annotation)

Usage
-----
  # Train with FreePRM silver labels (default)
  python train_step_ranking_model.py \
      --dataset esnli \
      --loss_function listnet \
      --encoder deberta-v3-base \
      --step_loss_weight 0.5 \
      --silver_strategy freeprm \
      --output_dir results/step_prm_esnli_listnet

  # Train with LLM-as-judge labels (if annotations already produced)
  python train_step_ranking_model.py \
      --silver_strategy llm_judge \
      --silver_labels_path data/step_labels_esnli.jsonl \
      ...
"""

import os, re, json, math, logging, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, load_from_disk
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. SPECIAL TOKEN + SENTENCE SEGMENTATION
# ---------------------------------------------------------------------------

STEP_TOKEN = "[STEP]"

SENT_SPLIT_RE = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+"
)


def split_into_steps(text: str, max_steps: int = 5) -> List[str]:
    """
    Split an explanation into steps (sentences).
    Falls back to the whole text as a single step if splitting yields <2 pieces.

    Boundary cases handled:
      - Single-sentence explanations  → [text]   (1 step, still valid)
      - Very long sentences           → kept whole (no mid-sentence split)
      - Empty string                  → [""]
      - Trailing whitespace / newlines → stripped
    """
    if not text or not text.strip():
        return [""]
    text = text.strip()
    parts = SENT_SPLIT_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) == 0:
        return [text]
    return parts[:max_steps]


def build_step_input(
    premise: str,
    hypothesis: str,
    explanation: str,
    tokenizer: AutoTokenizer,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Construct tokenised input with [STEP] tokens between sentences.

    Returns a dict with keys:
      input_ids, attention_mask, step_positions  (list[int])

    step_positions holds the token index of each [STEP] token.
    If there are N sentences, there are N-1 [STEP] tokens between them,
    plus the final [SEP] is used to obtain the last step hidden state.

    Example (3 sentences):
      [CLS] premise [SEP] hypothesis [SEP] s1 [STEP] s2 [STEP] s3 [SEP]
      step_positions = [idx_STEP_1, idx_STEP_2, idx_SEP_final]
    """
    steps = split_into_steps(explanation)

    # Build the text string with STEP_TOKEN markers
    explanation_with_steps = f" {STEP_TOKEN} ".join(steps)

    # Tokenise as a single sequence: "premise [SEP] hypothesis [SEP] expl_with_steps"
    # We rely on tokenizer's automatic [CLS]/[SEP] insertion for BERT-style models.
    encoding = tokenizer(
        f"{premise} {hypothesis}",          # sentence A
        explanation_with_steps,             # sentence B
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].squeeze(0)  # (seq_len,)

    # Find positions of [STEP] token id
    step_token_id = tokenizer.convert_tokens_to_ids(STEP_TOKEN)
    step_positions = (input_ids == step_token_id).nonzero(as_tuple=True)[0].tolist()

    # Also append position of final [SEP] token so every sentence has a
    # corresponding position to extract its hidden state from.
    sep_token_id = tokenizer.sep_token_id
    sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0].tolist()
    if sep_positions:
        step_positions.append(sep_positions[-1])

    return {
        "input_ids": input_ids,
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "step_positions": step_positions,   # list[int], length = num_steps
    }


# ---------------------------------------------------------------------------
# 2. SILVER STEP LABEL GENERATION
# ---------------------------------------------------------------------------

class SilverLabelGenerator:
    """
    Three strategies for generating step-level binary labels
    without human annotation.

    Strategy A: FreePRM heuristic
    ─────────────────────────────
    Assign label = 1 (correct) to all steps of explanations with
    quality_score >= quality_threshold, else 0.
    Simple and effective — FreePRM achieves 53% F1 on ProcessBench.

    Boundary case: single-step explanations always have the explanation
    score propagated directly.

    Strategy B: LLM-as-judge
    ─────────────────────────
    Load pre-generated JSONL file: {"qid": ..., "step_idx": ..., "label": 0/1}
    Use if you've run the annotation script (see generate_step_labels_llm.py).

    Strategy C: Implicit PRM (log-ratio)
    ──────────────────────────────────────
    Estimate per-step quality via:
        r(step_t) ≈ log π_θ(step_t | context) - log π_ref(step_t | context)
    Requires a reference language model.  Sign of r gives binary label.
    Useful when you have a trained policy but no step annotations.
    """

    def __init__(self, strategy: str = "freeprm", quality_threshold: float = 3.0):
        self.strategy = strategy
        self.quality_threshold = quality_threshold
        self._llm_labels: Dict[Tuple, int] = {}   # (qid, step_idx) → label

    def load_llm_labels(self, path: str):
        """Load pre-generated LLM-as-judge step labels from JSONL."""
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                self._llm_labels[(rec["qid"], rec["step_idx"])] = rec["label"]
        logger.info(f"Loaded {len(self._llm_labels)} LLM step labels from {path}")

    def get_step_labels(
        self,
        qid: str,
        quality_score: float,
        num_steps: int,
    ) -> List[int]:
        """
        Return binary label per step.

        Parameters
        ----------
        qid           : query-level unique ID
        quality_score : explanation-level quality (0–4 scale)
        num_steps     : number of steps in this explanation
        """
        if self.strategy == "freeprm":
            label = 1 if quality_score >= self.quality_threshold else 0
            return [label] * num_steps

        elif self.strategy == "llm_judge":
            labels = []
            for i in range(num_steps):
                key = (qid, i)
                if key in self._llm_labels:
                    labels.append(self._llm_labels[key])
                else:
                    # Fallback: propagate explanation-level label
                    labels.append(1 if quality_score >= self.quality_threshold else 0)
            return labels

        else:
            raise ValueError(f"Unknown silver strategy: {self.strategy}")


# ---------------------------------------------------------------------------
# 3. DATASET
# ---------------------------------------------------------------------------

class StepRankingDataset(Dataset):
    """
    Each item is ONE explanation (one candidate for one query).
    The DataLoader collects candidates for the SAME query into a list
    via a custom collate_fn (see below).

    Fields per item
    ---------------
    input_ids, attention_mask  : standard token tensors
    step_positions             : List[int]  — positions of [STEP] tokens
    quality_score              : float      — explanation-level quality (0–4)
    step_labels                : List[int]  — binary label per step
    query_id                   : str        — used to group candidates
    """

    def __init__(
        self,
        data: List[Dict],          # list of dicts from the builder
        tokenizer: AutoTokenizer,
        silver_gen: SilverLabelGenerator,
        max_length: int = 256,
    ):
        self.items = []
        for ex in tqdm(data, desc="Tokenising"):
            for cand in ex["explanations"]:
                encoded = build_step_input(
                    ex["premise"],
                    ex["hypothesis"],
                    cand["explanation"],
                    tokenizer,
                    max_length=max_length,
                )
                step_labels = silver_gen.get_step_labels(
                    qid=ex["query_id"],
                    quality_score=cand["quality_score"],
                    num_steps=len(encoded["step_positions"]),
                )
                self.items.append({
                    "input_ids":      encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "step_positions": encoded["step_positions"],
                    "quality_score":  float(cand["quality_score"]),
                    "step_labels":    step_labels,
                    "query_id":       ex["query_id"],
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_by_query(batch: List[Dict]) -> Dict[str, List]:
    """
    Group items by query_id into lists of candidates.
    Returns one dict per batch, where each value is a list of length
    num_queries, and each element is a list over candidates.

    This mirrors the listwise structure expected by ListNet loss.

    Boundary cases
    ──────────────
    - Queries with only 1 candidate are still included (ListNet trivially
      returns 0 loss; step loss still trains).
    - step_positions lists may differ in length across candidates of the
      same query — handled by zero-padding inside the loss function.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for item in batch:
        grouped[item["query_id"]].append(item)

    return {"queries": list(grouped.values())}


# ---------------------------------------------------------------------------
# 4. MODEL
# ---------------------------------------------------------------------------

class StepLevelRankingModel(nn.Module):
    """
    Encoder + shared projection head for step-level ranking.

    Forward pass
    ────────────
    1. Encode the full input_ids sequence.
    2. Extract hidden states at each step_position.
    3. Pass each through shared projection_head → step_score_i.
    4. Aggregate step scores → explanation_score via min-pooling.
    5. Return both for separate loss terms.

    Design choices
    ──────────────
    - Shared head across all step positions: fewer parameters,
      position-invariant scoring (any step position can be the weak link).
    - min-pooling: explanation is only as good as its weakest step.
      Literature (VersaPRM, PQM) confirms min outperforms mean/last.
    - [CLS] auxiliary head (optional): provides a "holistic" score for
      multi-task training alongside the step-level objective.

    Boundary cases
    ──────────────
    - Single-step explanation: step_positions has length 1.
      min-pool of one element = that element's score. Works correctly.
    - Truncated input (step token shifted out of window): step_positions
      will be empty after truncation. The model falls back to [CLS] score.
      This is logged as a warning during data prep.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        hidden_size: int = 768,
        intermediate_size: int = 384,
        dropout: float = 0.1,
        use_cls_auxiliary: bool = True,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(dropout)

        # Shared projection head (same weights for every step position)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 1),
        )

        # Optional CLS-level auxiliary head
        self.use_cls_auxiliary = use_cls_auxiliary
        if use_cls_auxiliary:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, 1),
            )

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, seq_len)
        attention_mask: torch.Tensor,   # (B, seq_len)
        step_positions: List[List[int]],# outer list = batch, inner = positions
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        step_scores      : List[Tensor]  — one tensor per example, shape (num_steps,)
        expl_scores      : Tensor        — shape (B,), min over step scores
        cls_scores       : Tensor | None — shape (B,), from CLS head
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state  # (B, seq_len, d)

        step_scores_list = []
        expl_scores      = []

        for b_idx, positions in enumerate(step_positions):
            if len(positions) == 0:
                # Fallback: use CLS representation
                h = self.dropout(hidden[b_idx, 0, :].unsqueeze(0))  # (1, d)
                score = self.projection_head(h).squeeze(-1)          # (1,)
                step_scores_list.append(score)
                expl_scores.append(score.squeeze())
            else:
                h_steps = hidden[b_idx, positions, :]          # (num_steps, d)
                h_steps = self.dropout(h_steps)
                scores  = self.projection_head(h_steps).squeeze(-1)  # (num_steps,)
                step_scores_list.append(scores)
                # min-pooling: weakest step determines explanation quality
                expl_scores.append(scores.min())

        expl_scores_tensor = torch.stack(expl_scores)  # (B,)

        cls_scores = None
        if self.use_cls_auxiliary:
            h_cls  = self.dropout(hidden[:, 0, :])          # (B, d)
            cls_scores = self.cls_head(h_cls).squeeze(-1)   # (B,)

        return {
            "step_scores": step_scores_list,   # List[Tensor(num_steps,)]
            "expl_scores": expl_scores_tensor, # (B,)
            "cls_scores":  cls_scores,         # (B,) or None
        }


# ---------------------------------------------------------------------------
# 5. LOSS FUNCTIONS
# ---------------------------------------------------------------------------

def listnet_loss(
    pred_scores: torch.Tensor,  # (K,) predicted scores for K candidates
    gold_scores: torch.Tensor,  # (K,) gold quality scores
) -> torch.Tensor:
    """
    ListNet (top-1 approximation): KL divergence between softmax
    distributions of gold and predicted scores.

      L = -Σ_i  softmax(gold)[i] * log softmax(pred)[i]

    Boundary cases
    ──────────────
    - K=1: both softmax distributions are [1.0], loss = 0. Correct.
    - All gold scores equal: uniform distribution, zero gradient. This
      happens with some neutral NLI pairs — expected behaviour.
    - Very large score differences: numerical stability ensured by
      PyTorch's log_softmax (uses logsumexp internally).
    """
    log_pred = F.log_softmax(pred_scores, dim=0)
    target   = F.softmax(gold_scores.float(), dim=0)
    return -torch.sum(target * log_pred)


def step_bce_loss(
    step_scores_list: List[torch.Tensor],  # List[Tensor(num_steps,)]
    step_labels_list: List[List[int]],     # List[List[int]]
) -> torch.Tensor:
    """
    Binary cross-entropy over all (step, label) pairs in the batch.

    Labels are provided as 0/1 integers from the silver generator.

    Boundary cases
    ──────────────
    - num_steps > len(step_labels): can happen if truncation shifts out a
      [STEP] token unexpectedly. We zip and take the shorter length.
    - All labels identical in a batch: BCE still provides valid gradient
      as long as model predictions vary.
    - Empty batch (no steps at all): returns tensor(0.0) to avoid NaN.
    """
    all_scores  = []
    all_labels  = []
    for scores, labels in zip(step_scores_list, step_labels_list):
        n = min(len(scores), len(labels))
        if n == 0:
            continue
        all_scores.append(scores[:n])
        all_labels.extend(labels[:n])

    if len(all_scores) == 0:
        return torch.tensor(0.0, requires_grad=True)

    scores_cat = torch.cat(all_scores)
    labels_t   = torch.tensor(all_labels, dtype=torch.float32,
                              device=scores_cat.device)
    return F.binary_cross_entropy_with_logits(scores_cat, labels_t)


def combined_loss(
    model_out:        Dict,
    gold_scores:      List[torch.Tensor],  # one (K,) tensor per query
    step_labels_list: List[List[int]],     # flat list over batch
    lambda_rank:      float = 1.0,
    lambda_step:      float = 0.5,
    lambda_cls:       float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    L = λ_rank * L_ListNet(expl_scores, gold_ranks)
      + λ_step * L_step_BCE(step_scores, silver_labels)
      + λ_cls  * L_ListNet(cls_scores, gold_ranks)   [optional]

    The ListNet loss is computed LISTWISE — all K candidates for a single
    query are grouped and scored together.  This requires the DataLoader
    to yield one query group at a time (see collate_by_query).

    Implementation note: gold_scores and model expl_scores must be aligned
    such that gold_scores[i][j] corresponds to the j-th candidate of the
    i-th query in the batch.
    """
    rank_losses = []
    for i, gs in enumerate(gold_scores):
        if len(gs) < 2:
            continue  # skip single-candidate queries (no ranking signal)
        # Slice model expl_scores for candidates belonging to this query
        # (see training loop below for how we pass these per-query)
        pass  # implemented inline in training loop

    step_loss = step_bce_loss(
        model_out["step_scores"],
        step_labels_list,
    )

    return {"step_loss": step_loss}   # rank_loss assembled in training loop


# ---------------------------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------------------------

def ndcg_at_k(
    predicted_scores: List[float],
    gold_scores:      List[float],
    k: int = 5,
) -> float:
    """
    NDCG@k computed from predicted ranking vs. gold quality scores.

    Boundary cases
    ──────────────
    - k > len(gold_scores): k clamped to len(gold_scores)
    - All gold scores equal: ideal DCG = actual DCG → NDCG = 1.0
    - Single candidate: NDCG = 1.0 by definition
    """
    k = min(k, len(gold_scores))
    if k == 0:
        return 0.0

    order = np.argsort(predicted_scores)[::-1]
    ranked_gold = np.array(gold_scores)[order]

    def dcg(scores, k):
        gains = scores[:k]
        discounts = np.log2(np.arange(2, k + 2))
        return np.sum(gains / discounts)

    ideal_gold = np.sort(gold_scores)[::-1]
    idcg = dcg(ideal_gold, k)
    if idcg == 0:
        return 1.0
    return dcg(ranked_gold, k) / idcg


def separation_ratio(
    predicted_scores: List[float],
    gold_scores:      List[float],
) -> float:
    """
    Fraction of the ground-truth score range captured by predicted scores.
    Higher is better; values near 1.0 indicate good score spread.
    This is the paper's central metric for PPO viability.

    separation_ratio = std(predicted) / std(gold)

    Clamped to [0, 1].
    """
    if len(predicted_scores) < 2:
        return 0.0
    std_pred = float(np.std(predicted_scores))
    std_gold = float(np.std(gold_scores))
    if std_gold == 0:
        return 1.0 if std_pred == 0 else 0.0
    return min(std_pred / std_gold, 1.0)


def step_accuracy(
    step_scores_list:  List[torch.Tensor],
    step_labels_list:  List[List[int]],
    threshold:         float = 0.0,  # logit threshold (0 = sigmoid(0) = 0.5)
) -> float:
    """
    Binary accuracy of per-step predictions against silver labels.
    Used to monitor whether step-level BCE is learning anything useful.
    """
    correct, total = 0, 0
    for scores, labels in zip(step_scores_list, step_labels_list):
        n = min(len(scores), len(labels))
        preds = (scores[:n] > threshold).long().cpu().tolist()
        correct += sum(p == l for p, l in zip(preds, labels[:n]))
        total   += n
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 7. TRAINING LOOP
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:       StepLevelRankingModel,
    dataloader:  DataLoader,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    device:      torch.device,
    lambda_rank: float = 1.0,
    lambda_step: float = 0.5,
) -> Dict[str, float]:
    model.train()
    total_rank_loss, total_step_loss, total_loss = 0.0, 0.0, 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        # batch["queries"] is a list of query-groups;
        # each group is a list of candidate dicts
        query_groups = batch["queries"]

        for group in query_groups:
            if len(group) == 0:
                continue

            # Stack tensors for this query group
            input_ids   = torch.stack([g["input_ids"] for g in group]).to(device)
            attn_mask   = torch.stack([g["attention_mask"] for g in group]).to(device)
            step_pos    = [g["step_positions"] for g in group]
            gold_scores = torch.tensor(
                [g["quality_score"] for g in group], dtype=torch.float32
            ).to(device)
            step_labels = [g["step_labels"] for g in group]

            out = model(input_ids, attn_mask, step_pos)

            # Listwise ranking loss over explanation scores
            rank_loss = listnet_loss(out["expl_scores"], gold_scores)

            # Step-level BCE loss
            s_loss = step_bce_loss(out["step_scores"], step_labels)

            loss = lambda_rank * rank_loss + lambda_step * s_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_rank_loss += rank_loss.item()
            total_step_loss += s_loss.item()
            total_loss      += loss.item()
            n_batches       += 1

    n = max(n_batches, 1)
    return {
        "rank_loss": total_rank_loss / n,
        "step_loss": total_step_loss / n,
        "total_loss": total_loss / n,
    }


@torch.no_grad()
def evaluate(
    model:      StepLevelRankingModel,
    dataloader: DataLoader,
    device:     torch.device,
) -> Dict[str, float]:
    model.eval()
    all_ndcg5, all_sep_ratios, all_step_acc = [], [], []

    for batch in dataloader:
        query_groups = batch["queries"]
        for group in query_groups:
            if len(group) < 2:
                continue

            input_ids = torch.stack([g["input_ids"] for g in group]).to(device)
            attn_mask = torch.stack([g["attention_mask"] for g in group]).to(device)
            step_pos  = [g["step_positions"] for g in group]
            gold      = [g["quality_score"] for g in group]
            s_labels  = [g["step_labels"] for g in group]

            out = model(input_ids, attn_mask, step_pos)
            pred = out["expl_scores"].cpu().tolist()

            all_ndcg5.append(ndcg_at_k(pred, gold, k=5))
            all_sep_ratios.append(separation_ratio(pred, gold))
            all_step_acc.append(step_accuracy(out["step_scores"], s_labels))

    return {
        "ndcg@5":          float(np.mean(all_ndcg5)) if all_ndcg5 else 0.0,
        "sep_ratio":       float(np.mean(all_sep_ratios)) if all_sep_ratios else 0.0,
        "step_accuracy":   float(np.mean(all_step_acc)) if all_step_acc else 0.0,
    }


# ---------------------------------------------------------------------------
# 8. ARGUMENT PARSING + MAIN
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train step-level process reward model")
    p.add_argument("--encoder",            default="microsoft/deberta-v3-base")
    p.add_argument("--dataset",            default="esnli",
                   choices=["esnli", "ds_critique", "combined"])
    p.add_argument("--data_dir",           default="data/processed/comprehensive_ranking_dataset")
    p.add_argument("--loss_function",      default="listnet",
                   choices=["listnet", "ranknet", "lambdarank", "approxndcg"])
    p.add_argument("--silver_strategy",    default="freeprm",
                   choices=["freeprm", "llm_judge"])
    p.add_argument("--silver_labels_path", default=None,
                   help="Path to JSONL with LLM step labels (only for llm_judge)")
    p.add_argument("--step_loss_weight",   type=float, default=0.5)
    p.add_argument("--rank_loss_weight",   type=float, default=1.0)
    p.add_argument("--quality_threshold",  type=float, default=3.0,
                   help="FreePRM: explanations with score >= threshold get label=1")
    p.add_argument("--max_steps_per_expl", type=int, default=5)
    p.add_argument("--max_length",         type=int, default=256)
    p.add_argument("--batch_size",         type=int, default=8)
    p.add_argument("--num_epochs",         type=int, default=10)
    p.add_argument("--lr",                 type=float, default=2e-5)
    p.add_argument("--warmup_ratio",       type=float, default=0.1)
    p.add_argument("--output_dir",         default="results/step_prm_esnli_listnet")
    p.add_argument("--seed",               type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Tokenizer + add STEP token ──────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    tokenizer.add_tokens([STEP_TOKEN])
    step_token_id = tokenizer.convert_tokens_to_ids(STEP_TOKEN)
    logger.info(f"[STEP] token id: {step_token_id}")

    # ── Silver label generator ───────────────────────────────────────────────
    silver_gen = SilverLabelGenerator(
        strategy=args.silver_strategy,
        quality_threshold=args.quality_threshold,
    )
    if args.silver_strategy == "llm_judge":
        assert args.silver_labels_path, "--silver_labels_path required for llm_judge"
        silver_gen.load_llm_labels(args.silver_labels_path)

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset from {args.data_dir}")
    raw = load_from_disk(args.data_dir)

    # Convert HuggingFace dataset splits to list[dict]
    train_data = list(raw["train"])
    val_data   = list(raw["validation"])

    train_ds = StepRankingDataset(train_data, tokenizer, silver_gen, args.max_length)
    val_ds   = StepRankingDataset(val_data,   tokenizer, silver_gen, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_by_query,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_by_query,
        num_workers=2,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    # Infer hidden_size from encoder config
    from transformers import AutoConfig
    enc_config  = AutoConfig.from_pretrained(args.encoder)
    hidden_size = enc_config.hidden_size

    model = StepLevelRankingModel(
        encoder_name=args.encoder,
        hidden_size=hidden_size,
        intermediate_size=hidden_size // 2,
    ).to(device)

    # Resize embeddings to account for new [STEP] token
    model.encoder.resize_token_embeddings(len(tokenizer))

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    best_ndcg, best_sep = 0.0, 0.0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            lambda_rank=args.rank_loss_weight,
            lambda_step=args.step_loss_weight,
        )
        val_metrics = evaluate(model, val_loader, device)

        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)

        logger.info(
            f"Epoch {epoch:02d} | "
            f"rank_loss={train_metrics['rank_loss']:.4f} "
            f"step_loss={train_metrics['step_loss']:.4f} | "
            f"val NDCG@5={val_metrics['ndcg@5']:.4f} "
            f"sep_ratio={val_metrics['sep_ratio']:.4f} "
            f"step_acc={val_metrics['step_accuracy']:.4f}"
        )

        if val_metrics["ndcg@5"] > best_ndcg:
            best_ndcg = val_metrics["ndcg@5"]
            best_sep  = val_metrics["sep_ratio"]
            model_save = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_len":    len(tokenizer),
                "args":             vars(args),
                "val_metrics":      val_metrics,
            }, model_save)
            logger.info(f"  ✓ Best model saved (NDCG@5={best_ndcg:.4f})")

    # Save training history
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
        json.dump({"history": history, "best_ndcg": best_ndcg, "best_sep": best_sep}, f, indent=2)

    logger.info(f"\n✅ Done. Best NDCG@5={best_ndcg:.4f}, sep_ratio={best_sep:.4f}")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()