"""Build a qid-disjoint train/held-out split from the DS-Critique train pool.

The published in-domain validation file cannot support a powered evaluation: it
holds 270 student explanations over 197 questions, 141 of them singletons, and
only 52 groups are rankable. The train pool by contrast is 270 questions with
twelve scored explanations each, every group rankable. Holding out a fifth of
its questions gives 54 groups of twelve - 648 candidates with no singletons -
at the cost of training on 216 questions instead of 270.

Two properties matter and are enforced here.

**The split is disjoint by question, not by explanation.** Every candidate for a
held-out qid leaves the training set together, so a model cannot have seen a
sibling explanation of a question it is evaluated on.

**The split does not depend on the training seed.** Selection is a stable hash
of the qid, so seeds 42, 123 and 777 are trained and evaluated on exactly the
same partition and their results are comparable. A seeded RNG would have made
each replicate a different experiment.

The holdout is stratified by domain, because the pool is imbalanced
(ARC-Challenge 100 questions, most others 20) and an unstratified fifth could
miss a domain entirely.

    python scripts/build_qid_split.py --holdout-fraction 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arr.utils import read_jsonl, stable_hash, write_json, write_jsonl


def select_holdout(
    groups: list[dict], fraction: float, salt: str
) -> tuple[list[dict], list[dict]]:
    """Deterministic, domain-stratified holdout keyed on a stable hash of qid."""

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        by_domain[group["domain"]].append(group)

    held_ids: set[str] = set()
    for domain in sorted(by_domain):
        members = sorted(
            by_domain[domain],
            key=lambda g: stable_hash(salt, (g.get("metadata") or {}).get("qid", g["group_id"])),
        )
        take = max(1, int(round(fraction * len(members))))
        held_ids.update(g["group_id"] for g in members[:take])

    held = [g for g in groups if g["group_id"] in held_ids]
    kept = [g for g in groups if g["group_id"] not in held_ids]
    return kept, held


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/arr/ds_critique_external_test.jsonl")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/arr"))
    parser.add_argument("--prefix", default="ds_critique_qidsplit")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--salt", default="arr-qid-split-v1")
    args = parser.parse_args(argv)

    groups = list(read_jsonl(args.source))
    qids = {(g.get("metadata") or {}).get("qid") for g in groups}
    if len(qids) != len(groups):
        raise ValueError(
            f"expected one group per qid, found {len(groups)} groups and {len(qids)} qids"
        )

    kept, held = select_holdout(groups, args.holdout_fraction, args.salt)
    kept_qids = {(g["metadata"] or {}).get("qid") for g in kept}
    held_qids = {(g["metadata"] or {}).get("qid") for g in held}
    overlap = kept_qids & held_qids
    if overlap:
        raise ValueError(f"qid overlap between partitions: {sorted(overlap)[:5]}")

    train_path = args.out_dir / f"{args.prefix}_train.jsonl"
    dev_path = args.out_dir / f"{args.prefix}_dev.jsonl"
    write_jsonl(train_path, kept)
    write_jsonl(dev_path, held)

    def profile(rows: list[dict]) -> dict:
        domains: dict[str, int] = defaultdict(int)
        for row in rows:
            domains[row["domain"]] += 1
        return {
            "groups": len(rows),
            "candidates": sum(len(r["candidates"]) for r in rows),
            "domains": dict(sorted(domains.items())),
        }

    manifest = {
        "source": str(args.source),
        "salt": args.salt,
        "holdout_fraction": args.holdout_fraction,
        "selection": "stable_hash(salt, qid), domain-stratified, seed-independent",
        "train": {"path": str(train_path), **profile(kept)},
        "held_out": {"path": str(dev_path), **profile(held)},
        "held_out_qids": sorted(q for q in held_qids if q),
        "qid_disjoint": True,
    }
    write_json(args.out_dir / f"{args.prefix}_manifest.json", manifest)

    print(f"train     {profile(kept)}")
    print(f"held out  {profile(held)}")
    print(f"qid-disjoint: {not overlap}")
    print(f"wrote {train_path}, {dev_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
