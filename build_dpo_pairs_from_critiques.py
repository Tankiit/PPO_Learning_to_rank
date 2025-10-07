"""
Build DPO preference pairs from a 'Digital Socrates' critiques dataset.

The script:
1) Loads a DatasetDict from disk (e.g., "train" and optionally "validation").
2) Groups student explanations by question id (qid).
3) Extracts a numeric critique score for each explanation.
4) Forms (chosen, rejected) pairs for DPO whenever the chosen score > rejected score.
5) Writes JSONL with fields: prompt, chosen, rejected, source_model{chosen,rejected}, scores{chosen,rejected}.

Assumptions about input dataset fields (configurable via CLI):
- qid field: "qid"
- question field: "question"
- explanation field: "student_explanation"
- model field: "student_model"
- critiques field: "critiques" (list of dicts)
- score path inside each critique item (dot path): "critique_elements.explanation_score"

Score aggregation across multiple critiques per item is configurable:
- first (default): use the first critique that contains the score
- max / mean: aggregate across all critiques that provide the score

Examples
--------
Default (train split only):
    python build_dpo_pairs_from_critiques.py \
        --dataset-path datasets/Digital_Socrate_Critique_Bank \
        --output-file dpo_preferences_from_critiques.jsonl

Train + validation, min score gap and cap pairs:
    python build_dpo_pairs_from_critiques.py \
        --dataset-path datasets/Digital_Socrate_Critique_Bank \
        --splits train,validation \
        --min-score-gap 0.05 \
        --max-pairs-total 200000

Change score aggregation and prompt template:
    python build_dpo_pairs_from_critiques.py \
        --score-aggregate mean \
        --prompt-template "Question: {question}\nExplain why the selected answer is best.\n"
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from collections import defaultdict
from datasets import Dataset, DatasetDict, load_from_disk

LOG = logging.getLogger("build_dpo_pairs")


# ---------------------------------------------------------------------------
# CLI / Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    dataset_path: Path
    output_file: Path
    splits: Tuple[str, ...]
    qid_field: str
    question_field: str
    explanation_field: str
    model_field: str
    critiques_field: str
    score_path: str            # dotted path inside a single critique item
    score_aggregate: str       # "first" | "max" | "mean"
    min_expl_len: int          # minimal explanation length to keep
    min_score_gap: float       # minimal difference between scores to create a pair
    max_pairs_per_qid: Optional[int]
    max_pairs_total: Optional[int]
    shuffle_pairs: bool


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Build DPO pairs from critiques.")
    p.add_argument("--dataset-path", type=Path, required=True, help="Path to a saved DatasetDict (load_from_disk).")
    p.add_argument("--output-file", type=Path, default=Path("dpo_preferences_from_critiques.jsonl"))

    p.add_argument("--splits", type=str, default="train", help="Comma-separated splits to include, e.g. 'train,validation'.")

    # Field names
    p.add_argument("--qid-field", type=str, default="qid")
    p.add_argument("--question-field", type=str, default="question")
    p.add_argument("--explanation-field", type=str, default="student_explanation")
    p.add_argument("--model-field", type=str, default="student_model")
    p.add_argument("--critiques-field", type=str, default="critiques")

    # Score extraction
    p.add_argument("--score-path", type=str, default="critique_elements.explanation_score",
                   help="Dotted path inside EACH critique item to reach the numeric score.")
    p.add_argument("--score-aggregate", type=str, default="first", choices=["first", "max", "mean"],
                   help="How to aggregate when multiple critiques are present.")

    # Filtering / pairing controls
    p.add_argument("--min-expl-len", type=int, default=1, help="Drop explanations shorter than this length.")
    p.add_argument("--min-score-gap", type=float, default=0.0, help="Require at least this absolute score difference.")
    p.add_argument("--max-pairs-per-qid", type=int, default=None, help="Cap number of pairs generated per qid.")
    p.add_argument("--max-pairs-total", type=int, default=None, help="Cap total number of pairs written.")
    p.add_argument("--shuffle-pairs", type=lambda x: str(x).lower() == "true", default=True)

    a = p.parse_args()
    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    return RunConfig(
        dataset_path=a.dataset_path,
        output_file=a.output_file,
        splits=splits or ("train",),
        qid_field=a.qid_field,
        question_field=a.question_field,
        explanation_field=a.explanation_field,
        model_field=a.model_field,
        critiques_field=a.critiques_field,
        score_path=a.score_path,
        score_aggregate=a.score_aggregate,
        min_expl_len=a.min_expl_len,
        min_score_gap=a.min_score_gap,
        max_pairs_per_qid=a.max_pairs_per_qid,
        max_pairs_total=a.max_pairs_total,
        shuffle_pairs=a.shuffle_pairs,
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_nested(d: Dict[str, Any], dotted: str) -> Optional[Any]:
    """Safely get a nested value using a dotted path (e.g., 'a.b.c')."""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _extract_score_from_critiques(
    critiques: Any,
    score_path: str,
    aggregate: str = "first",
) -> Optional[float]:
    """Extract a numeric score from a critiques list.

    Args:
        critiques: usually a list[dict]
        score_path: dotted path inside EACH critique item
        aggregate: 'first' | 'max' | 'mean'

    Returns:
        float score if found, else None.
    """
    if not isinstance(critiques, (list, tuple)) or not critiques:
        return None

    scores: List[float] = []
    for c in critiques:
        if not isinstance(c, dict):
            continue
        v = _get_nested(c, score_path)
        if v is None:
            continue
        try:
            scores.append(float(v))
        except (TypeError, ValueError):
            continue

    if not scores:
        return None

    if aggregate == "first":
        return scores[0]
    if aggregate == "max":
        return max(scores)
    if aggregate == "mean":
        return sum(scores) / len(scores)
    return scores[0]


def _normalize_text(x: Optional[str]) -> str:
    return (x or "").strip()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_entries(cfg: RunConfig) -> List[Dict[str, Any]]:
    """Load and combine selected splits into a flat list of dictionaries."""
    ds: DatasetDict = load_from_disk(str(cfg.dataset_path))
    parts: List[Dataset] = []
    for sp in cfg.splits:
        if sp not in ds:
            LOG.warning("Split '%s' not found. Available: %s", sp, list(ds.keys()))
            continue
        parts.append(ds[sp])

    if not parts:
        raise ValueError("No valid splits found to load.")

    combined = Dataset.from_dict({}) if len(parts) == 1 else None
    # Concatenate without importing concatenate_datasets by re-materializing rows
    rows: List[Dict[str, Any]] = []
    for part in parts:
        rows.extend(part)
    LOG.info("Loaded %d rows across splits: %s", len(rows), cfg.splits)
    return rows


def group_by_qid(rows: List[Dict[str, Any]], cfg: RunConfig) -> Dict[str, List[Dict[str, Any]]]:
    """Group usable explanations by qid, each enriched with score and source_model."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    kept, skipped = 0, 0

    for item in rows:
        qid = str(item.get(cfg.qid_field, "")).strip()
        if not qid:
            skipped += 1
            continue

        question = _normalize_text(item.get(cfg.question_field))
        explanation = _normalize_text(item.get(cfg.explanation_field))
        if len(explanation) < cfg.min_expl_len:
            skipped += 1
            continue

        score = _extract_score_from_critiques(
            item.get(cfg.critiques_field, []),
            score_path=cfg.score_path,
            aggregate=cfg.score_aggregate,
        )
        if score is None:
            skipped += 1
            continue

        source_model = _normalize_text(item.get(cfg.model_field)) or "unknown"

        grouped[qid].append(
            {
                "question": question,
                "explanation": explanation,
                "score": float(score),
                "source_model": source_model,
            }
        )
        kept += 1

    LOG.info("Grouped entries: kept=%d, skipped=%d, qids=%d", kept, skipped, len(grouped))
    return grouped


def build_pairs_for_qid(
    qid: str,
    entries: List[Dict[str, Any]],
    min_gap: float,
    max_pairs: Optional[int],
) -> List[Dict[str, Any]]:
    """Form all ordered pairs (chosen>rejected) for a single qid with optional cap."""
    pairs: List[Dict[str, Any]] = []

    # Generate combinations without replacement
    for a, b in itertools.combinations(entries, 2):
        if a["score"] == b["score"]:
            continue
        # Determine chosen/rejected and check margin
        if a["score"] > b["score"]:
            chosen, rejected = a, b
        else:
            chosen, rejected = b, a

        if abs(chosen["score"] - rejected["score"]) < min_gap:
            continue

        prompt = f"Question: {chosen['question']}\nWhy is the selected answer correct?\n"
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen["explanation"],
                "rejected": rejected["explanation"],
                "source_model": {
                    "chosen": chosen["source_model"],
                    "rejected": rejected["source_model"],
                },
                "scores": {
                    "chosen": chosen["score"],
                    "rejected": rejected["score"],
                },
                "qid": qid,
            }
        )

        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    return pairs


def build_all_pairs(
    grouped: Dict[str, List[Dict[str, Any]]],
    cfg: RunConfig,
) -> List[Dict[str, Any]]:
    """Build DPO pairs across all qids, optionally shuffling and capping total."""
    all_pairs: List[Dict[str, Any]] = []

    for qid, entries in grouped.items():
        if len(entries) < 2:
            continue
        q_pairs = build_pairs_for_qid(qid, entries, cfg.min_score_gap, cfg.max_pairs_per_qid)
        all_pairs.extend(q_pairs)

    LOG.info("Built %d raw pairs (before optional shuffle & cap).", len(all_pairs))

    if cfg.shuffle_pairs:
        import random
        random.shuffle(all_pairs)

    if cfg.max_pairs_total is not None and len(all_pairs) > cfg.max_pairs_total:
        all_pairs = all_pairs[: cfg.max_pairs_total]
        LOG.info("Capped pairs to max_pairs_total=%d.", cfg.max_pairs_total)

    return all_pairs


def write_jsonl(pairs: Iterable[Dict[str, Any]], output_file: Path) -> int:
    """Write pairs to JSONL; returns number written."""
    n = 0
    with output_file.open("w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(cfg: RunConfig) -> None:
    rows = load_entries(cfg)
    grouped = group_by_qid(rows, cfg)
    pairs = build_all_pairs(grouped, cfg)
    n = write_jsonl(pairs, cfg.output_file)
    LOG.info("%d DPO pairs written to %s", n, cfg.output_file)


def main() -> None:
    setup_logging()
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
