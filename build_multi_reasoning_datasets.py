"""
Build and normalize multiple reasoning datasets into a unified format.

Datasets covered:
- DeltaNLI (defeasible-nli: atomic & snli)
- Sense-Making (tasksource/sen-making)
- AlphaNLI (local JSONL + labels)
- WinoWhy (local JSON)

For each dataset we produce examples with fields:
    - "Premise" (str)
    - "Hypothesis" (str)                # when applicable
    - "Explanation" (str)               # when applicable
    - "Question" (str)                  # instruction-like prompt
    - "labels" (int)                    # 1 = good/valid, 0 = bad/invalid

Output:
- Saves each processed DatasetDict to <output_root>/<dataset_name>
- Writes a dataset_names.txt listing all saved names

Example:
    python build_multi_reasoning_datasets.py \
        --output-root all_datasets \
        --alphanli-dir datasets/alphanli-train-dev \
        --winowhy-json datasets/WinoWhy/winowhy.json \
        --seed 42

Notes:
- Keep the label semantics consistent: 1 (good/strengthening/valid), 0 (bad/weakening/invalid).
- Splits:
    * Sense-Making: 80/10/10 (train/val/test) from its train split
    * AlphaNLI: 80/20 from train => (train/test), dev => validation
    * WinoWhy: 80/20 then split test into 50/50 => val/test
    * DeltaNLI splits are those provided by the HF dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

LOG = logging.getLogger("build_multi_reasoning_datasets")


# -----------------------------------------------------------------------------
# CLI / Config / Utils
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    output_root: Path
    alphanli_dir: Path
    winowhy_json: Path
    seed: int


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Build multiple reasoning datasets in a unified format.")
    p.add_argument("--output-root", type=Path, default=Path("all_datasets"), help="Root directory to save datasets.")
    p.add_argument("--alphanli-dir", type=Path, default=Path("datasets/alphanli-train-dev"),
                   help="Directory containing AlphaNLI train/dev JSONL + labels.")
    p.add_argument("--winowhy-json", type=Path, default=Path("datasets/WinoWhy/winowhy.json"),
                   help="Path to WinoWhy JSON file.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    a = p.parse_args()
    return RunConfig(output_root=a.output_root, alphanli_dir=a.alphanli_dir, winowhy_json=a.winowhy_json, seed=a.seed)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)


# -----------------------------------------------------------------------------
# DeltaNLI (defeasible-nli: atomic & snli)
# -----------------------------------------------------------------------------

def _transform_delta_nli(example: Dict) -> Dict:
    """Map a defeasible-NLI example to (Question, labels)."""
    # 1 = "strengthener", 0 otherwise
    label = 1 if str(example.get("UpdateType", "")).lower() == "strengthener" else 0
    premise = example.get("Premise", "")
    hypo = example.get("Hypothesis", "")
    update = example.get("Update", "")

    example["Question"] = f"If: '{premise}' entails: '{hypo}', why is that true? {update}"
    example["labels"] = label
    return example


def build_delta_nli_atomic() -> DatasetDict:
    ds = load_dataset("tasksource/defeasible-nli", "atomic")
    ds = ds.map(_transform_delta_nli, desc="deltaNLI[atomic]: transform")
    LOG.info("DeltaNLI[atomic] splits: %s", {k: len(v) for k, v in ds.items()})
    return ds


def build_delta_nli_snli() -> DatasetDict:
    ds = load_dataset("tasksource/defeasible-nli", "snli")
    ds = ds.map(_transform_delta_nli, desc="deltaNLI[snli]: transform")
    LOG.info("DeltaNLI[snli] splits: %s", {k: len(v) for k, v in ds.items()})
    return ds


# -----------------------------------------------------------------------------
# Sense-Making (tasksource/sen-making)
# -----------------------------------------------------------------------------

_SEN_POSSIBLE_REASONS = ["A", "B", "C"]  # keys present in dataset
_SEN_QUESTION_SUFFIX = "Why is this statement against common sense?"

def _transform_sense_making_one(example: Dict, rng: random.Random) -> List[Dict]:
    """Create two examples per input: (good reason, bad reason)."""
    false_idx = example["false"]                    # which sentence is false (1..N); dataset uses 1..?
    sentence_key = f"sentence{false_idx}"          # e.g., sentence1/sentence2/...
    correct_reason_key = example["reason"]         # 'A'|'B'|'C'

    # GOOD example
    correct_reason_text = example.get(correct_reason_key, "")
    good = {
        "Premise": example[sentence_key],
        "Hypothesis": "",  # not applicable
        "Explanation": correct_reason_text,
        "labels": 1,
        "Question": f"{example[sentence_key]}, {_SEN_QUESTION_SUFFIX} {correct_reason_text}",
    }

    # BAD example: pick a different reason key
    wrong_candidates = [r for r in _SEN_POSSIBLE_REASONS if r != correct_reason_key]
    fake_key = rng.choice(wrong_candidates)
    fake_reason_text = example.get(fake_key, "")
    bad = {
        "Premise": example[sentence_key],
        "Hypothesis": "",
        "Explanation": fake_reason_text,
        "labels": 0,
        "Question": f"{example[sentence_key]}, {_SEN_QUESTION_SUFFIX} {fake_reason_text}",
    }
    return [good, bad]


def build_sense_making(seed: int) -> DatasetDict:
    """Load and expand Sense-Making train split into 2x examples, then 80/10/10."""
    base = load_dataset("tasksource/sen-making")
    rng = random.Random(seed)

    expanded: List[Dict] = []
    for ex in base["train"]:
        expanded.extend(_transform_sense_making_one(ex, rng))

    ds = Dataset.from_list(expanded)
    # 80/20 then 50/50 of the 20
    train_test = ds.train_test_split(test_size=0.2, seed=seed)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=seed)
    result = DatasetDict(
        train=train_test["train"],
        validation=val_test["train"],
        test=val_test["test"],
    )
    LOG.info("Sense-Making splits: %s", {k: len(v) for k, v in result.items()})
    return result


# -----------------------------------------------------------------------------
# AlphaNLI (local JSONL + labels)
# -----------------------------------------------------------------------------

def _load_alphanli_split(jsonl_path: Path, labels_path: Path) -> Tuple[List[Dict], List[int]]:
    with jsonl_path.open("r", encoding="utf-8") as f_jsonl:
        examples = [json.loads(line) for line in f_jsonl]
    with labels_path.open("r", encoding="utf-8") as f_labels:
        labels = [int(line.strip()) for line in f_labels]
    if len(examples) != len(labels):
        raise ValueError(f"AlphaNLI size mismatch: {len(examples)} examples vs {len(labels)} labels.")
    return examples, labels


def _transform_alphanli(examples: Sequence[Dict], labels: Sequence[int]) -> List[Dict]:
    """Produce two examples per item: (correct hypothesis, wrong hypothesis)."""
    out: List[Dict] = []
    for ex, lab in zip(examples, labels):
        # In AlphaNLI, label=1 means hyp1 is correct (common convention)
        correct_hyp = ex["hyp1"] if lab == 1 else ex["hyp2"]
        wrong_hyp = ex["hyp2"] if lab == 1 else ex["hyp1"]

        premise = ex["obs1"]
        hypothesis = ex["obs2"]

        # positive
        out.append({
            "Premise": premise,
            "Hypothesis": hypothesis,
            "Explanation": correct_hyp,
            "labels": 1,
            "Question": f"If: '{premise}' entails: '{hypothesis}', why is that true? {correct_hyp}",
        })
        # negative
        out.append({
            "Premise": premise,
            "Hypothesis": hypothesis,
            "Explanation": wrong_hyp,
            "labels": 0,
            "Question": f"If: '{premise}' entails: '{hypothesis}', why is that true? {wrong_hyp}",
        })
    return out


def build_alphanli(alphanli_dir: Path, seed: int) -> DatasetDict:
    """Load AlphaNLI train/dev, create train/test from train (80/20) and use dev as validation."""
    train_jsonl = alphanli_dir / "train.jsonl"
    train_labels = alphanli_dir / "train-labels.lst"
    dev_jsonl = alphanli_dir / "dev.jsonl"
    dev_labels = alphanli_dir / "dev-labels.lst"

    train_ex, train_lab = _load_alphanli_split(train_jsonl, train_labels)
    dev_ex, dev_lab = _load_alphanli_split(dev_jsonl, dev_labels)

    combined = list(zip(train_ex, train_lab))
    rng = random.Random(seed)
    rng.shuffle(combined)

    split_idx = int(0.8 * len(combined))
    train_split = combined[:split_idx]
    test_split = combined[split_idx:]

    t_ex, t_lab = zip(*train_split) if train_split else ([], [])
    te_ex, te_lab = zip(*test_split) if test_split else ([], [])

    train_data = _transform_alphanli(t_ex, t_lab) if t_ex else []
    test_data = _transform_alphanli(te_ex, te_lab) if te_ex else []
    val_data = _transform_alphanli(dev_ex, dev_lab)

    ds = DatasetDict(
        train=Dataset.from_list(train_data),
        validation=Dataset.from_list(val_data),
        test=Dataset.from_list(test_data),
    )
    LOG.info("AlphaNLI splits: %s", {k: len(v) for k, v in ds.items()})
    return ds


# -----------------------------------------------------------------------------
# WinoWhy (local JSON)
# -----------------------------------------------------------------------------

def _extract_winowhy_examples(items: Sequence[Dict]) -> List[Dict]:
    """Extract positive/negative examples with minimal fields from WinoWhy JSON."""
    out: List[Dict] = []
    for item in items:
        text = item["text"]
        premise = f"{text['txt1']} {text['pron']} {text['txt2']}"

        answers = item["answers"]
        # correctAnswer can be 'A', 'A.', 'B', 'B.'
        corr_is_a = str(item["correctAnswer"]).strip().upper().startswith("A")
        correct_idx = 0 if corr_is_a else 1
        hyp_correct = f"It refers to {answers[correct_idx]}."
        hyp_wrong = f"It refers to {answers[1 - correct_idx]}."

        # reasons is a list of [reason, source, score, label]
        reasons = item["reasons"]
        valid_reason = None
        invalid_reason = None
        for reason, source, score, label in reasons:
            if valid_reason is None and str(label).lower().startswith("valid"):
                valid_reason = reason
            if invalid_reason is None and str(label).lower().startswith("invalid"):
                invalid_reason = reason
            if valid_reason and invalid_reason:
                break

        if not valid_reason or not invalid_reason:
            continue  # skip if we can't find both

        # positive
        out.append({
            "Premise": premise,
            "Hypothesis": hyp_correct,
            "Explanation": valid_reason,
            "labels": 1,
            "Question": f"If: '{premise}' entails: '{hyp_correct}', why is that true? {valid_reason}",
        })
        # negative
        out.append({
            "Premise": premise,
            "Hypothesis": hyp_wrong,
            "Explanation": invalid_reason,
            "labels": 0,
            "Question": f"If: '{premise}' entails: '{hyp_wrong}', why is that true? {invalid_reason}",
        })
    return out


def build_winowhy(json_path: Path, seed: int) -> DatasetDict:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = _extract_winowhy_examples(data)
    base = Dataset.from_list(rows)

    # 80/20 then 50/50
    train_test = base.train_test_split(test_size=0.2, seed=seed)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=seed)
    ds = DatasetDict(
        train=train_test["train"],
        validation=val_test["train"],
        test=val_test["test"],
    )
    LOG.info("WinoWhy splits: %s", {k: len(v) for k, v in ds.items()})
    return ds


# -----------------------------------------------------------------------------
# Save utilities
# -----------------------------------------------------------------------------

def save_datasets(all_sets: Dict[str, DatasetDict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name, ds in all_sets.items():
        out_dir = output_root / name
        ds.save_to_disk(str(out_dir))
        LOG.info("Saved %s to %s", name, out_dir)

    names_path = output_root / "dataset_names.txt"
    with names_path.open("w", encoding="utf-8") as f:
        for name in all_sets:
            f.write(name + "\n")
    LOG.info("Wrote dataset_names.txt with %d names.", len(all_sets))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run(cfg: RunConfig) -> None:
    set_seed(cfg.seed)

    LOG.info("Building DeltaNLI (atomic)...")
    delta_atomic = build_delta_nli_atomic()

    LOG.info("Building DeltaNLI (snli)...")
    delta_snli = build_delta_nli_snli()

    LOG.info("Building Sense-Making...")
    sen_making = build_sense_making(cfg.seed)

    LOG.info("Building AlphaNLI from: %s", cfg.alphanli_dir)
    alphanli = build_alphanli(cfg.alphanli_dir, cfg.seed)

    LOG.info("Building WinoWhy from: %s", cfg.winowhy_json)
    winowhy = build_winowhy(cfg.winowhy_json, cfg.seed)

    LOG.info("All datasets ready. Saving...")
    all_datasets = {
        "deltaNLI_atomic": delta_atomic,
        "deltaNLI_snli": delta_snli,
        "sen_making": sen_making,
        "alphanli": alphanli,
        "winowhy": winowhy,
    }
    save_datasets(all_datasets, cfg.output_root)


def main() -> None:
    setup_logging()
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
