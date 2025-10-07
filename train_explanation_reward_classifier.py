"""
Train a binary reward classifier for explanations.

This script supports three modes:
  - esnli : train only on the locally built eSNLI 9x3 dataset
  - all   : train only on the merged "all_datasets" corpus
  - both  : concatenate (all_datasets + eSNLI 9x3) for training

The original code mixed both flows in one file. This refactor makes the
choice explicit via --mode and provides a consistent preprocessing pipeline.

Data expectations
-----------------
eSNLI path (default: datasets/my_dataset_eSNLI_9x3_20K) should contain fields:
  - premise, hypothesis
  - Question_{i}_{j}, Answer_{i}_{j}, label_{i}_{j}, gold_label_{i}_{j}
and we flatten them into per-row examples with:
  - text: either (question + explanation) or (explanation)   [configurable]
  - labels: from gold_label (1 good / 0 bad)

all_datasets root (default: all_datasets) should contain subfolders as saved by
`build_multi_reasoning_datasets.py`, plus a file `dataset_names.txt`.
Each sub-dataset is expected to expose a "Question" field and "labels" ∈ {0,1}.

Training
--------
- Any HF sequence classifier (default: bert-base-uncased)
- Tokenization on "text" for eSNLI and on "Question" for all_datasets
- Class balancing (downsample majority)
- Early stopping on validation
- Evaluation on test split at the end

Examples
--------
Train on ALL datasets only (last modifications behavior):
    python train_explanation_reward_classifier.py --mode all \
        --all-root all_datasets --model-name bert-base-uncased

Train on eSNLI only:
    python train_explanation_reward_classifier.py --mode esnli \
        --esnli-path datasets/my_dataset_eSNLI_9x3_20K

Train on BOTH (concat) with custom lengths:
    python train_explanation_reward_classifier.py --mode both \
        --max-length 192 --per-device-train-batch-size 128
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

LOG = logging.getLogger("train_reward_classifier")


# ---------------------------------------------------------------------------
# CLI / Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    # data sources & mode
    mode: str  # "esnli" | "all" | "both"
    esnli_path: Path
    all_root: Path

    # tokenizer/model
    model_name: str

    # text building for eSNLI ("explanation" | "q_plus_expl")
    esnli_text_mode: str

    # target sizes per-dataset contribution for "all" merge (beware of total!)
    target_train_per_dataset: int
    target_valid_per_dataset: int
    target_test_per_dataset: int

    # tokenization / padding
    max_length: int

    # training
    output_dir: Path
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    eval_steps: int
    save_steps: int
    seed: int


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Train a binary explanation reward classifier.")

    # data & mode
    p.add_argument("--mode", type=str, default="all", choices=["esnli", "all", "both"],
                   help="Training source: esnli | all | both.")
    p.add_argument("--esnli-path", type=Path, default=Path("datasets/my_dataset_eSNLI_9x3_20K"))
    p.add_argument("--all-root", type=Path, default=Path("all_datasets"))

    # model
    p.add_argument("--model-name", type=str, default="bert-base-uncased",
                   help="HF encoder model name for classification.")

    # eSNLI text building
    p.add_argument("--esnli-text-mode", type=str, default="q_plus_expl",
                   choices=["explanation", "q_plus_expl"],
                   help="Build eSNLI text from explanation only or question+explanation.")

    # all_datasets per-dataset target sizes (the final total = sum over datasets)
    p.add_argument("--target-train-per-dataset", type=int, default=100000)
    p.add_argument("--target-valid-per-dataset", type=int, default=10000)
    p.add_argument("--target-test-per-dataset", type=int, default=10000)

    # tokenization
    p.add_argument("--max-length", type=int, default=192)

    # training args
    p.add_argument("--output-dir", type=Path, default=Path("models/bert-base-uncased_CLS-2"))
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--per-device-train-batch-size", type=int, default=128)
    p.add_argument("--per-device-eval-batch-size", type=int, default=128)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    a = p.parse_args()
    return RunConfig(
        mode=a.mode,
        esnli_path=a.esnli_path,
        all_root=a.all_root,
        model_name=a.model_name,
        esnli_text_mode=a.esnli_text_mode,
        target_train_per_dataset=a.target_train_per_dataset,
        target_valid_per_dataset=a.target_valid_per_dataset,
        target_test_per_dataset=a.target_test_per_dataset,
        max_length=a.max_length,
        output_dir=a.output_dir,
        num_train_epochs=a.num_train_epochs,
        per_device_train_batch_size=a.per_device_train_batch_size,
        per_device_eval_batch_size=a.per_device_eval_batch_size,
        eval_steps=a.eval_steps,
        save_steps=a.save_steps,
        seed=a.seed,
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy/precision/recall/f1 (weighted, micro, macro)."""
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_w": float(precision_score(labels, preds, zero_division=0, average="weighted")),
        "recall_w": float(recall_score(labels, preds, zero_division=0, average="weighted")),
        "f1_w": float(f1_score(labels, preds, zero_division=0, average="weighted")),
        "precision_mi": float(precision_score(labels, preds, zero_division=0, average="micro")),
        "recall_mi": float(recall_score(labels, preds, zero_division=0, average="micro")),
        "f1_mi": float(f1_score(labels, preds, zero_division=0, average="micro")),
        "precision_ma": float(precision_score(labels, preds, zero_division=0, average="macro")),
        "recall_ma": float(recall_score(labels, preds, zero_division=0, average="macro")),
        "f1_ma": float(f1_score(labels, preds, zero_division=0, average="macro")),
    }
    return metrics


# ---------------------------------------------------------------------------
# eSNLI 9x3 processing
# ---------------------------------------------------------------------------

_Q_USER_SPLIT_RE = re.compile(
    r"<\|start_header_id\|>user<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>",
    flags=re.DOTALL,
)

def _extract_question_from_chat(query: str) -> str:
    """Extract the 'user' message content from a Llama-3 style chat template."""
    m = _Q_USER_SPLIT_RE.search(query or "")
    return m.group(1).strip() if m else (query or "").strip()


def _extract_explanation_from_answer(answer_text: str) -> str:
    """Best-effort extraction of the explanation from 'Answer_{i}_{j}'.

    In the refactored dataset builder we decode only the generated continuation,
    so 'answer_text' should already be the completion. This function keeps a
    light cleanup just in case.
    """
    if not answer_text:
        return ""
    # Drop special tokens if present
    cleaned = re.sub(r"<\|.*?\|>", " ", str(answer_text)).strip()
    return cleaned


def _flatten_esnli_9x3_row(example: Dict, text_mode: str) -> List[Dict]:
    """Produce 27 new rows from a single eSNLI example."""
    rows: List[Dict] = []
    for i in range(1, 10):
        for j in range(1, 4):
            q_key = f"Question_{i}_{j}"
            a_key = f"Answer_{i}_{j}"
            l_key = f"label_{i}_{j}"
            g_key = f"gold_label_{i}_{j}"

            question = _extract_question_from_chat(example.get(q_key, ""))
            explanation = _extract_explanation_from_answer(example.get(a_key, ""))

            if text_mode == "explanation":
                text = explanation
            else:  # "q_plus_expl"
                text = f"{question} {explanation}".strip()

            rows.append(
                {
                    "text": text,
                    "labels": int(example.get(g_key, 0)),
                    # keep for potential debugging
                    "premise": example.get("premise", ""),
                    "hypothesis": example.get("hypothesis", ""),
                }
            )
    return rows


def load_and_prepare_esnli_9x3(esnli_path: Path, text_mode: str, seed: int) -> DatasetDict:
    """Load eSNLI 9x3 dataset and flatten to simple (text, labels)."""
    if not esnli_path.exists():
        raise FileNotFoundError(f"eSNLI path not found: {esnli_path}")
    ds: DatasetDict = load_from_disk(str(esnli_path))

    LOG.info("Loaded eSNLI 9x3: %s", {k: len(v) for k, v in ds.items()})

    for split in list(ds.keys()):
        flat_rows: List[Dict] = []
        for ex in ds[split]:
            flat_rows.extend(_flatten_esnli_9x3_row(ex, text_mode=text_mode))
        ds[split] = Dataset.from_list(flat_rows)

    # Balance classes per split (downsample majority)
    for split in list(ds.keys()):
        ds[split] = balance_binary(ds[split], label_field="labels", seed=seed)

    LOG.info("eSNLI 9x3 flattened & balanced: %s", {k: len(v) for k, v in ds.items()})
    return ds


# ---------------------------------------------------------------------------
# all_datasets processing
# ---------------------------------------------------------------------------

def _load_all_datasets(root: Path) -> Dict[str, DatasetDict]:
    names_file = root / "dataset_names.txt"
    if not names_file.exists():
        raise FileNotFoundError(f"Missing dataset_names.txt in {root}")
    names = [ln.strip() for ln in names_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    result: Dict[str, DatasetDict] = {}
    for name in names:
        dpath = root / name
        if not dpath.exists():
            LOG.warning("Dataset listed but not found on disk: %s", dpath)
            continue
        result[name] = DatasetDict.load_from_disk(str(dpath))
    LOG.info("Loaded %d datasets from %s", len(result), root)
    return result


def _sample_to_size(ds: Dataset, target_size: int, seed: int, with_replacement: bool) -> Dataset:
    """Return a dataset of exactly target_size."""
    n = len(ds)
    if n == target_size:
        return ds
    rng = np.random.RandomState(seed)
    idxs = rng.choice(n, size=target_size, replace=with_replacement)
    return ds.select(list(map(int, idxs)))


def build_all_merged(
    root: Path,
    target_train_per_dataset: int,
    target_valid_per_dataset: int,
    target_test_per_dataset: int,
    seed: int,
) -> DatasetDict:
    """Merge all datasets by sampling each split to a fixed size per dataset.

    WARNING: total sizes = (#datasets × per-dataset target). Adjust targets to fit GPU.
    """
    all_sets = _load_all_datasets(root)
    merged: Dict[str, List[Dataset]] = {"train": [], "validation": [], "test": []}

    for name, ds in all_sets.items():
        for split, target in [
            ("train", target_train_per_dataset),
            ("validation", target_valid_per_dataset),
            ("test", target_test_per_dataset),
        ]:
            if split not in ds:
                continue
            part = ds[split]
            # remember the source for debugging/analysis
            part = part.add_column("source", [name] * len(part))

            with_replacement = len(part) < target
            sampled = _sample_to_size(part, target, seed=seed, with_replacement=with_replacement)
            merged[split].append(sampled)

    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        if merged[split]:
            out[split] = concatenate_datasets(merged[split])
            # Balance classes per split
            out[split] = balance_binary(out[split], label_field="labels", seed=seed)
    LOG.info("Merged all_datasets: %s", {k: len(v) for k, v in out.items()})
    return out


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def tokenize_text_column(ds: DatasetDict, tokenizer: AutoTokenizer, col: str, max_length: int) -> DatasetDict:
    """Tokenize a text column in all splits."""
    def _tok(batch):
        return tokenizer(batch[col], truncation=True, max_length=max_length)
    result = DatasetDict()
    for split in ds.keys():
        result[split] = ds[split].map(_tok, batched=True, desc=f"tokenize[{split}]")
    return result


def drop_all_but_model_inputs(ds: DatasetDict) -> DatasetDict:
    """Remove non-model columns, keep only input_ids/attention_mask/labels (token_type_ids kept if present)."""
    result = DatasetDict()
    for split in ds.keys():
        cols = set(ds[split].column_names)
        keep = {"input_ids", "attention_mask", "labels", "token_type_ids"}
        drop = sorted(list(cols - keep))
        result[split] = ds[split].remove_columns(drop)
    return result


def balance_binary(ds: Dataset, label_field: str = "labels", seed: int = 42) -> Dataset:
    """Downsample the majority class to the minority size."""
    labels = np.array(ds[label_field], dtype=int)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return ds  # nothing to balance

    target = min(len(idx_pos), len(idx_neg))
    rng = np.random.RandomState(seed)
    pick_pos = rng.choice(idx_pos, size=target, replace=False)
    pick_neg = rng.choice(idx_neg, size=target, replace=False)
    idx = np.concatenate([pick_pos, pick_neg])
    rng.shuffle(idx)
    return ds.select(list(map(int, idx)))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_esnli_pipeline(cfg: RunConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """eSNLI branch: flatten, build text, tokenize, keep model inputs."""
    ds = load_and_prepare_esnli_9x3(cfg.esnli_path, cfg.esnli_text_mode, cfg.seed)
    ds = tokenize_text_column(ds, tokenizer, col="text", max_length=cfg.max_length)
    ds = drop_all_but_model_inputs(ds)
    return ds


def build_all_pipeline(cfg: RunConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """all_datasets branch: merge, tokenize on Question, keep model inputs."""
    ds = build_all_merged(
        root=cfg.all_root,
        target_train_per_dataset=cfg.target_train_per_dataset,
        target_valid_per_dataset=cfg.target_valid_per_dataset,
        target_test_per_dataset=cfg.target_test_per_dataset,
        seed=cfg.seed,
    )
    # Ensure 'Question' exists; builder guarantees it.
    ds = tokenize_text_column(ds, tokenizer, col="Question", max_length=cfg.max_length)
    ds = drop_all_but_model_inputs(ds)
    return ds


def maybe_concat(a: DatasetDict | None, b: DatasetDict | None) -> DatasetDict:
    """Concatenate DatasetDicts split-wise (handles None)."""
    if a is None and b is None:
        raise ValueError("Nothing to train on (both branches are None).")
    if a is None:
        return b  # type: ignore
    if b is None:
        return a
    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        if split in a and split in b:
            out[split] = concatenate_datasets([a[split], b[split]])
        elif split in a:
            out[split] = a[split]
        elif split in b:
            out[split] = b[split]
    # Shuffle train/validation for safety
    if "train" in out:
        out["train"] = out["train"].shuffle(seed=42)
    if "validation" in out:
        out["validation"] = out["validation"].shuffle(seed=42)
    return out


def run(cfg: RunConfig) -> None:
    hf_set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    LOG.info("Loading tokenizer & model: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    esnli_ds: DatasetDict | None = None
    all_ds: DatasetDict | None = None

    if cfg.mode in ("esnli", "both"):
        LOG.info("Building eSNLI pipeline (mode=%s, text_mode=%s)", cfg.mode, cfg.esnli_text_mode)
        esnli_ds = build_esnli_pipeline(cfg, tokenizer)

    if cfg.mode in ("all", "both"):
        LOG.info("Building all_datasets pipeline (mode=%s)", cfg.mode)
        all_ds = build_all_pipeline(cfg, tokenizer)

    final_ds = maybe_concat(esnli_ds, all_ds)
    LOG.info("Final dataset sizes: %s", {k: len(v) for k, v in final_ds.items()})

    # Collator pads to a fixed max length
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=cfg.max_length,
        return_tensors="pt",
    )

    # TrainingArguments
    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        seed=cfg.seed,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_w",
        greater_is_better=True,
        logging_steps=max(50, cfg.eval_steps // 5),
        report_to=[],  # set to ["wandb"] if you want W&B
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=final_ds.get("train"),
        eval_dataset=final_ds.get("validation"),
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    LOG.info("# Train")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    LOG.info("# Save model to %s", cfg.output_dir)
    trainer.save_model(str(cfg.output_dir))

    LOG.info("# Evaluate on test split")
    test_metrics = trainer.evaluate(eval_dataset=final_ds.get("test"))
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
