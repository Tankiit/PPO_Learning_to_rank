"""Prepare a fixed-candidate-count GPT-4 provenance exposure experiment.

This is a *new* intervention, not the existing full-data QID campaign. Every
training group has nine candidates at all exposure levels. Exposed QIDs replace
one candidate from each of the three known student models with the three GPT-4
candidates. Thus QID count, group size, and optimizer steps can be matched;
student composition and target quality still change and must be reported.

Evaluation is fixed on 54 disjoint held-out QIDs, both as full twelve-candidate
groups and as the GPT-4-only three-candidate subset. Neither evaluation file is
used for training. The partial exposure level is selected by a stable hash and
stratified by domain. The same files are reused across all training seeds.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.arr.data import load_groups
from src.arr.schema import RankingGroup
from src.arr.utils import stable_hash, write_json, write_jsonl


GPT4 = "gpt-4-0613"
TRAIN_FRACTIONS = (0.25, 0.5, 1.0)
EXPOSURE_FRACTIONS = (0.0, 0.5, 1.0)
PROVENANCE = "DS_Critique_Bank.explanation_annotations.human_crowd_mean"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_qids(
    groups: Sequence[RankingGroup], fraction: float, *, salt: str
) -> set[str]:
    """Choose a nested, domain-stratified QID prefix with SHA-256 ordering."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    by_domain: dict[str, list[RankingGroup]] = defaultdict(list)
    for group in groups:
        by_domain[group.domain].append(group)
    selected = set()
    for domain, members in sorted(by_domain.items()):
        ordered = sorted(members, key=lambda group: stable_hash(salt, domain, group.metadata["qid"]))
        selected.update(str(group.metadata["qid"]) for group in ordered[:round(fraction * len(ordered))])
    return selected


def _candidate_selection(group: RankingGroup, exposed: bool) -> list[dict[str, Any]]:
    known = [c for c in group.candidates if c.metadata["student_model"] != GPT4]
    gpt4 = [c for c in group.candidates if c.metadata["student_model"] == GPT4]
    models = sorted({c.metadata["student_model"] for c in known})
    if len(known) != 9 or len(gpt4) != 3 or len(models) != 3:
        raise ValueError(f"expected 9 known and 3 GPT-4 candidates: {group.group_id}")
    if not exposed:
        return [candidate.to_dict() for candidate in known]
    dropped = set()
    for model in models:
        choices = [candidate for candidate in known if candidate.metadata["student_model"] == model]
        if len(choices) != 3:
            raise ValueError(f"expected three candidates for {model}")
        dropped.add(min(choices, key=lambda candidate: stable_hash(
            "arr-r4-drop-v1", group.metadata["qid"], candidate.candidate_id
        )).candidate_id)
    retained = {candidate.candidate_id for candidate in known if candidate.candidate_id not in dropped}
    retained.update(candidate.candidate_id for candidate in gpt4)
    return [candidate.to_dict() for candidate in group.candidates if candidate.candidate_id in retained]


def materialize_cell(
    groups: Sequence[RankingGroup], train_fraction: float, exposure_fraction: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_qids = select_qids(groups, train_fraction, salt="arr-r4-train-qid-v1")
    selected_groups = [group for group in groups if group.metadata["qid"] in train_qids]
    exposed_qids = select_qids(
        selected_groups, exposure_fraction, salt="arr-r4-exposure-qid-v1"
    )
    rows = []
    for group in selected_groups:
        exposed = group.metadata["qid"] in exposed_qids
        clone = group.to_dict()
        clone["candidates"] = _candidate_selection(group, exposed)
        clone["candidate_mask"] = [True] * 9
        clone["split"] = "r4_exposure_train"
        clone["metadata"] = {
            **group.metadata,
            "r4_exposed_to_gpt4": exposed,
            "r4_train_fraction": train_fraction,
            "r4_exposure_fraction": exposure_fraction,
        }
        rows.append(clone)
    fingerprint = stable_hash(
        "arr-r4-exposure-v1", train_fraction, exposure_fraction,
        [(row["group_id"], [c["candidate_id"] for c in row["candidates"]]) for row in rows],
    )
    for row in rows:
        row["data_fingerprint"] = fingerprint
    if not rows or any(len(row["candidates"]) != 9 for row in rows):
        raise ValueError("R4 training cell has missing or variable candidates")
    info = {
        "train_fraction": train_fraction,
        "exposure_fraction": exposure_fraction,
        "groups": len(rows), "candidates": len(rows) * 9,
        "exposed_qids": len(exposed_qids),
        "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "exposed_domains": dict(sorted(Counter(
            row["domain"] for row in rows if row["metadata"]["r4_exposed_to_gpt4"]
        ).items())),
        "fingerprint": fingerprint,
        "qid_set_hash": stable_hash(sorted(train_qids)),
        "exposed_qid_set_hash": stable_hash(sorted(exposed_qids)),
        "training_candidate_count_per_group": 9,
    }
    return rows, info


def gpt4_heldout(groups: Sequence[RankingGroup]) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for group in groups:
        clone = group.to_dict()
        clone["candidates"] = [candidate.to_dict() for candidate in group.candidates
                               if candidate.metadata["student_model"] == GPT4]
        if len(clone["candidates"]) != 3:
            raise ValueError(f"held-out QID lacks three GPT-4 explanations: {group.group_id}")
        clone["candidate_mask"] = [True] * 3
        clone["split"] = "r4_gpt4_heldout"
        rows.append(clone)
    fingerprint = stable_hash(
        "arr-r4-gpt4-heldout-v1",
        [(row["group_id"], [c["candidate_id"] for c in row["candidates"]]) for row in rows],
    )
    for row in rows:
        row["data_fingerprint"] = fingerprint
    return rows, fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/arr/ds_critique_qidsplit_train.jsonl"))
    parser.add_argument("--heldout", type=Path, default=Path("data/arr/ds_critique_qidsplit_dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/arr/r4_exposure"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"R4 output already exists; refusing to overwrite: {args.output}")
    train = load_groups(args.train)
    heldout = load_groups(args.heldout)
    if len(train) != 216 or len(heldout) != 54:
        raise ValueError("expected 216 training and 54 held-out QIDs")
    train_qids = {group.metadata["qid"] for group in train}
    heldout_qids = {group.metadata["qid"] for group in heldout}
    if train_qids & heldout_qids:
        raise ValueError("R4 training and held-out QIDs overlap")
    if {candidate.score_provenance for group in (*train, *heldout)
        for candidate in group.candidates} != {PROVENANCE}:
        raise ValueError("R4 requires human crowd targets only")
    args.output.mkdir(parents=True, exist_ok=True)
    cells = []
    for fraction in TRAIN_FRACTIONS:
        for exposure in EXPOSURE_FRACTIONS:
            rows, info = materialize_cell(train, fraction, exposure)
            name = f"train_qid{round(fraction * 100):03d}_exposure{round(exposure * 100):03d}.jsonl"
            write_jsonl(args.output / name, rows)
            cells.append({"path": name, **info})
    heldout_rows, fingerprint = gpt4_heldout(heldout)
    write_jsonl(args.output / "gpt4_qid_holdout.jsonl", heldout_rows)
    full_cells = [cell for cell in cells if cell["train_fraction"] == 1.0]
    if [cell["exposed_qids"] for cell in full_cells] != [0, 108, 216]:
        raise ValueError("incorrect full-data exposure dose")
    write_json(args.output / "manifest.json", {
        "protocol": "arr-r4-exposure-v1",
        "target_provenance": PROVENANCE,
        "train_source_sha256": _file_sha256(args.train),
        "heldout_source_sha256": _file_sha256(args.heldout),
        "training_group_size": 9,
        "heldout_full_group_size": 12,
        "heldout_gpt4_group_size": 3,
        "heldout_gpt4_fingerprint": fingerprint,
        "train_qid_overlap_with_heldout": 0,
        "selection": "domain-stratified SHA-256; fixed across training seeds",
        "limitation": (
            "Replacing known-model explanations with GPT-4 keeps group size and "
            "optimizer budget matched but can change the target-quality distribution."
        ),
        "cells": cells,
    })
    print(f"Prepared {len(cells)} R4 train cells and one GPT-4 QID holdout in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
