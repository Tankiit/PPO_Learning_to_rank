from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Candidate, RankingGroup
from .utils import (
    deterministic_uniform,
    fingerprint_records,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
ACL_SCORE_RANGES = {
    "gold": (0.70, 1.00),
    "good": (0.50, 0.85),
    "fair": (0.30, 0.70),
    "poor": (0.10, 0.50),
    "nonsense": (0.00, 0.30),
}


def load_groups(path: str | Path) -> list[RankingGroup]:
    return [RankingGroup.from_dict(value) for value in read_jsonl(path)]


def _as_rows(dataset: Any) -> list[dict[str, Any]]:
    return [dict(dataset[index]) for index in range(len(dataset))]


def _find_cached_esnli() -> dict[str, Path] | None:
    roots = [
        Path.home() / ".cache" / "huggingface" / "datasets" / "esnli",
        Path.home() / ".cache" / "huggingface" / "datasets" / "esnli" / "plain_text",
    ]
    candidates: list[dict[str, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for train in root.rglob("esnli-train.arrow"):
            directory = train.parent
            validation = directory / "esnli-validation.arrow"
            test = directory / "esnli-test.arrow"
            if validation.exists() and test.exists():
                candidates.append({"train": train, "validation": validation, "test": test})
    if not candidates:
        return None
    # Prefer the newest complete cache entry, not transient ``cache-*.arrow`` files.
    return max(candidates, key=lambda item: item["train"].stat().st_mtime)


def load_esnli_source(source: str | Path | None = None) -> dict[str, Any]:
    """Load official e-SNLI splits from disk/cache, then Hugging Face as a last resort."""

    try:
        from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required for e-SNLI preparation") from exc

    if source is not None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path.suffix != ".arrow":
                raise ValueError("an e-SNLI source file must be an Arrow file")
            return {"train": Dataset.from_file(str(path))}
        if (path / "dataset_dict.json").exists():
            loaded = load_from_disk(str(path))
            if not isinstance(loaded, DatasetDict):
                raise ValueError(f"expected a DatasetDict at {path}")
            return dict(loaded)
        files = {
            split: path / f"esnli-{split}.arrow"
            for split in ("train", "validation", "test")
        }
        if all(item.exists() for item in files.values()):
            return {split: Dataset.from_file(str(item)) for split, item in files.items()}
        raise ValueError(f"cannot identify an official e-SNLI dataset under {path}")

    cached = _find_cached_esnli()
    if cached is not None:
        return {split: Dataset.from_file(str(path)) for split, path in cached.items()}
    try:
        return dict(load_dataset("esnli", "plain_text"))
    except Exception as exc:
        raise RuntimeError(
            "official e-SNLI is neither supplied nor cached; provide --source or allow a HF download"
        ) from exc


def _normalise_label(value: Any) -> int:
    if isinstance(value, str):
        lowered = value.lower().strip()
        reverse = {name: label for label, name in LABEL_NAMES.items()}
        if lowered in reverse:
            return reverse[lowered]
        if lowered.isdigit():
            value = int(lowered)
    label = int(value)
    if label not in LABEL_NAMES:
        raise ValueError(f"invalid NLI label: {value!r}")
    return label


def _valid_esnli_row(row: Mapping[str, Any]) -> bool:
    return all(str(row.get(key, "")).strip() for key in ("premise", "hypothesis", "explanation_1"))


def _stratified_hash_sample(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    by_label: dict[int, list[tuple[str, int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not _valid_esnli_row(row):
            continue
        label = _normalise_label(row["label"])
        identity = stable_hash(
            "esnli-source-row",
            row["premise"],
            row["hypothesis"],
            row["explanation_1"],
        )
        by_label[label].append((identity, index, row))
    labels = sorted(LABEL_NAMES)
    base, remainder = divmod(count, len(labels))
    selected: list[dict[str, Any]] = []
    for position, label in enumerate(labels):
        quota = base + int(position < remainder)
        available = sorted(by_label[label], key=lambda item: (item[0], item[1]))
        if len(available) < quota:
            raise ValueError(f"e-SNLI label {label} has only {len(available)} valid rows, need {quota}")
        for identity, source_index, row in available[:quota]:
            selected.append({**dict(row), "_source_index": source_index, "_source_hash": identity})
    return sorted(selected, key=lambda row: row["_source_hash"])


def _all_valid_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        if not _valid_esnli_row(row):
            continue
        identity = stable_hash(
            "esnli-source-row",
            row["premise"],
            row["hypothesis"],
            row["explanation_1"],
        )
        output.append({**dict(row), "_source_index": source_index, "_source_hash": identity})
    return output


def _tier_texts(premise: str, hypothesis: str, label_name: str, gold: str) -> dict[str, str]:
    relation = {
        "entailment": "directly supports",
        "neutral": "does not provide enough information to establish",
        "contradiction": "conflicts with",
    }[label_name]
    short_relation = {
        "entailment": "supports",
        "neutral": "does not decide",
        "contradiction": "opposes",
    }[label_name]
    return {
        "gold": gold.strip(),
        "good": (
            f'The premise, "{premise}", {relation} the hypothesis, '
            f'"{hypothesis}"; therefore the relation is {label_name}.'
        ),
        "fair": f"The premise {short_relation} the hypothesis, so the label is {label_name}.",
        "poor": f'The hypothesis says "{hypothesis}", which is labelled {label_name}.',
        "nonsense": "A calendar can be folded twice, and blue triangles usually prefer quiet libraries.",
    }


def _esnli_candidates(row: Mapping[str, Any], group_id: str) -> tuple[Candidate, ...]:
    label = _normalise_label(row["label"])
    label_name = LABEL_NAMES[label]
    texts = _tier_texts(
        str(row["premise"]).strip(),
        str(row["hypothesis"]).strip(),
        label_name,
        str(row["explanation_1"]).strip(),
    )
    candidates: list[Candidate] = []
    for tier in ("gold", "good", "fair", "poor", "nonsense"):
        low, high = ACL_SCORE_RANGES[tier]
        score = deterministic_uniform(low, high, "arr-esnli-score-v1", group_id, tier)
        candidates.append(
            Candidate(
                candidate_id=stable_hash(group_id, tier, length=20),
                text=texts[tier],
                score=round(score, 8),
                score_provenance=(
                    "human_esnli_explanation+deterministic_acl26_range"
                    if tier == "gold"
                    else "deterministic_acl26_degradation_range"
                ),
                metadata={
                    "quality_tier": tier,
                    "range": [low, high],
                    "human_text": tier == "gold",
                },
            )
        )
    return tuple(candidates)


def build_esnli_groups(
    source: str | Path | None = None,
    train_groups: int = 10_000,
    limit_per_split: int | None = None,
) -> dict[str, list[RankingGroup]]:
    datasets = load_esnli_source(source)
    required = {"train", "validation", "test"}
    missing = required - set(datasets)
    if missing:
        raise ValueError(f"official e-SNLI source lacks split(s): {sorted(missing)}")
    result: dict[str, list[RankingGroup]] = {}
    for official_split in ("train", "validation", "test"):
        rows = _as_rows(datasets[official_split])
        if official_split == "train":
            requested = min(train_groups, limit_per_split) if limit_per_split is not None else train_groups
            selected = _stratified_hash_sample(rows, requested)
        elif limit_per_split is not None:
            selected = _stratified_hash_sample(rows, limit_per_split)
        else:
            selected = _all_valid_rows(rows)
        split = "val" if official_split == "validation" else official_split
        fingerprint_payload = [
            {
                "source_hash": row["_source_hash"],
                "label": _normalise_label(row["label"]),
                "split": split,
            }
            for row in selected
        ]
        fingerprint = fingerprint_records(fingerprint_payload)
        groups: list[RankingGroup] = []
        for row in selected:
            label = _normalise_label(row["label"])
            group_id = stable_hash(
                "arr-esnli-v1", split, row["_source_hash"], row["_source_index"], length=24
            )
            question = (
                f"Premise: {str(row['premise']).strip()}\n"
                f"Hypothesis: {str(row['hypothesis']).strip()}\n"
                f"Relation: {LABEL_NAMES[label]}"
            )
            groups.append(
                RankingGroup(
                    group_id=group_id,
                    split=split,
                    domain="nli",
                    question=question,
                    candidates=_esnli_candidates(row, group_id),
                    data_fingerprint=fingerprint,
                    metadata={
                        "dataset": "e-SNLI",
                        "source_split": official_split,
                        "source_index": row["_source_index"],
                        "source_hash": row["_source_hash"],
                        "label": label,
                        "label_name": LABEL_NAMES[label],
                        "score_construction": "arr-esnli-v1",
                    },
                )
            )
        result[split] = groups
    audit_split_overlap(result)
    return result


def write_esnli_dataset(groups: Mapping[str, Sequence[RankingGroup]], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    split_manifest: dict[str, Any] = {}
    for split, split_groups in groups.items():
        path = output / f"esnli_{split}.jsonl"
        write_jsonl(path, split_groups)
        split_manifest[split] = {
            "path": str(path),
            "groups": len(split_groups),
            "candidates": sum(len(group.candidates) for group in split_groups),
            "fingerprint": split_groups[0].data_fingerprint if split_groups else None,
            "labels": dict(Counter(group.metadata["label_name"] for group in split_groups)),
        }
    manifest = {
        "pipeline": "arr",
        "dataset": "e-SNLI",
        "schema_version": 1,
        "score_ranges": ACL_SCORE_RANGES,
        "splits": split_manifest,
        "audit": audit_groups([group for values in groups.values() for group in values]),
    }
    write_json(output / "esnli_manifest.json", manifest)
    return manifest


def _human_scores(row: Mapping[str, Any], source_path: Path) -> tuple[list[float], list[str]]:
    annotations = row.get("explanation_annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(f"{source_path}: row {row.get('id')} has no human explanation_annotations")
    scores: list[float] = []
    workers: list[str] = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping) or "explanation_score" not in annotation:
            raise ValueError(f"{source_path}: malformed human annotation in row {row.get('id')}")
        score = float(annotation["explanation_score"])
        if not 0.0 <= score <= 5.0:
            raise ValueError(f"{source_path}: human explanation score outside 0..5: {score}")
        scores.append(score)
        if annotation.get("worker") is not None:
            workers.append(str(annotation["worker"]))
    return scores, workers


def build_ds_critique_groups(
    source: str | Path,
    split: str,
    expected_groups: int | None = None,
    expected_candidates: int | None = None,
) -> list[RankingGroup]:
    """Build DS-Critique groups exclusively from crowd explanation annotations."""

    path = Path(source)
    if "crowd-anno" not in path.name:
        raise ValueError(
            "DS-Critique evaluation must use a *crowd-anno.jsonl file; automatic critique scores are forbidden"
        )
    raw_rows = list(read_jsonl(path))
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    human_payload: list[dict[str, Any]] = []
    for row in raw_rows:
        qid = str(row.get("qid", "")).strip()
        text = str(row.get("student_explanation", "")).strip()
        if not qid or not text:
            raise ValueError(f"{path}: missing qid or student_explanation in row {row.get('id')}")
        scores, workers = _human_scores(row, path)
        prepared = dict(row)
        prepared["_human_scores"] = scores
        prepared["_human_workers"] = workers
        by_qid[qid].append(prepared)
        human_payload.append(
            {
                "id": row.get("id"),
                "qid": qid,
                "scores": scores,
                "workers": workers,
                "text_hash": stable_hash(text),
            }
        )
    if expected_groups is not None and len(by_qid) != expected_groups:
        raise ValueError(f"{path}: expected {expected_groups} groups, found {len(by_qid)}")
    if expected_candidates is not None and len(raw_rows) != expected_candidates:
        raise ValueError(f"{path}: expected {expected_candidates} candidates, found {len(raw_rows)}")

    fingerprint = fingerprint_records(sorted(human_payload, key=lambda item: str(item["id"])))
    groups: list[RankingGroup] = []
    for qid in sorted(by_qid):
        rows = sorted(by_qid[qid], key=lambda row: str(row.get("id", "")))
        candidates: list[Candidate] = []
        for row in rows:
            scores = row["_human_scores"]
            raw_id = str(row.get("id") or stable_hash(qid, row["student_explanation"]))
            candidates.append(
                Candidate(
                    candidate_id=stable_hash("arr-dscb", raw_id, length=20),
                    text=str(row["student_explanation"]).strip(),
                    score=float(np.mean(scores) / 5.0),
                    score_provenance="DS_Critique_Bank.explanation_annotations.human_crowd_mean",
                    metadata={
                        "source_id": raw_id,
                        "raw_human_scores": scores,
                        "annotation_count": len(scores),
                        "workers": row["_human_workers"],
                        "student_model": row.get("student_model"),
                        "student_answer": row.get("student_answer"),
                        "student_accuracy": row.get("student_accuracy"),
                    },
                )
            )
        first = rows[0]
        domain = str(first.get("dataset", "unknown"))
        claim_role = (
            "confirmatory_subdomain"
            if domain == "WinoGrande"
            else "exploratory_small_human_sample"
            if domain == "CommonsenseQA"
            else "external_transfer"
        )
        groups.append(
            RankingGroup(
                group_id=stable_hash("arr-dscb-v1", split, qid, length=24),
                split=split,
                domain=domain,
                question=str(first["question"]).strip(),
                candidates=tuple(candidates),
                data_fingerprint=fingerprint,
                metadata={
                    "dataset": "DS_Critique_Bank",
                    "source_file": path.name,
                    "qid": qid,
                    "gold_answer": first.get("gold_answer"),
                    "human_only": True,
                    "claim_role": claim_role,
                },
            )
        )
    return groups


def write_ds_critique_dataset(
    groups: Sequence[RankingGroup], output_dir: str | Path, split: str
) -> dict[str, Any]:
    output = Path(output_dir)
    path = output / f"ds_critique_{split}.jsonl"
    write_jsonl(path, groups)
    audit = audit_groups(groups)
    manifest = {
        "pipeline": "arr",
        "dataset": "DS_Critique_Bank",
        "split": split,
        "path": str(path),
        "groups": len(groups),
        "candidates": sum(len(group.candidates) for group in groups),
        "fingerprint": groups[0].data_fingerprint if groups else None,
        "score_provenance": "DS_Critique_Bank.explanation_annotations.human_crowd_mean",
        "human_only": True,
        "domains": dict(Counter(group.domain for group in groups)),
        "claim_roles": dict(Counter(group.metadata["claim_role"] for group in groups)),
        "audit": audit,
    }
    write_json(output / f"ds_critique_{split}_manifest.json", manifest)
    return manifest


def audit_split_overlap(split_groups: Mapping[str, Sequence[RankingGroup]]) -> None:
    source_to_split: dict[str, str] = {}
    for split, groups in split_groups.items():
        for group in groups:
            source_hash = str(group.metadata.get("source_hash", group.group_id))
            previous = source_to_split.get(source_hash)
            if previous is not None and previous != split:
                raise ValueError(f"source example {source_hash} overlaps {previous} and {split}")
            source_to_split[source_hash] = split


def audit_groups(groups: Sequence[RankingGroup]) -> dict[str, Any]:
    ids = [group.group_id for group in groups]
    duplicate_group_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    constant_groups = [
        group.group_id for group in groups if np.std([candidate.score for candidate in group.candidates]) <= 1e-12
    ]
    empty_candidates = [
        f"{group.group_id}/{candidate.candidate_id}"
        for group in groups
        for candidate in group.candidates
        if not candidate.text.strip()
    ]
    provenance = Counter(
        candidate.score_provenance for group in groups for candidate in group.candidates
    )
    return {
        "group_count": len(groups),
        "candidate_count": sum(len(group.candidates) for group in groups),
        "duplicate_group_ids": duplicate_group_ids,
        "constant_group_count": len(constant_groups),
        "constant_group_ids": constant_groups,
        "empty_candidates": empty_candidates,
        "score_provenance": dict(provenance),
        "passed": not duplicate_group_ids and not empty_candidates,
    }


def stratified_group_sample(
    groups: Sequence[RankingGroup], count: int, seed: int, fields: Sequence[str] = ("domain",)
) -> list[RankingGroup]:
    if count >= len(groups):
        return list(groups)
    strata: dict[tuple[str, ...], list[RankingGroup]] = defaultdict(list)
    for group in groups:
        key = tuple(str(group.metadata.get(field, group.domain if field == "domain" else "")) for field in fields)
        strata[key].append(group)
    ordered_strata = sorted(strata)
    allocations = {key: int(round(count * len(strata[key]) / len(groups))) for key in ordered_strata}
    while sum(allocations.values()) < count:
        key = max(ordered_strata, key=lambda item: len(strata[item]) - allocations[item])
        allocations[key] += 1
    while sum(allocations.values()) > count:
        viable = [key for key in ordered_strata if allocations[key] > 0]
        key = max(viable, key=lambda item: allocations[item] - count * len(strata[item]) / len(groups))
        allocations[key] -= 1
    selected: list[RankingGroup] = []
    for key in ordered_strata:
        candidates = sorted(strata[key], key=lambda group: stable_hash(seed, key, group.group_id))
        selected.extend(candidates[: allocations[key]])
    return sorted(selected, key=lambda group: stable_hash(seed, group.group_id))
