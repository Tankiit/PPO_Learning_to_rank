"""Materialise the perturbed training files for controls 2 and 3.

Both controls are interventions on the *training data*, so each cell is a real
file on disk. That is deliberate rather than incidental.

**Control 2 closes its trap structurally.** The aleatoric control requires that
every ensemble member see the same noisy targets: if each member drew its own
noise, the noise itself would inject member disagreement - synthetic epistemic
uncertainty - and the control would pass for the wrong reason. Writing one
perturbed file per (sigma, seed) and pointing all five members at it makes a
per-member draw impossible to express, which is stronger than a comment saying
not to do it.

**Control 3 keeps the evaluation fixed.** Training fractions are nested subsets
of the qid-split training questions, chosen by a stable hash of the qid, so the
25% questions are contained in the 50% and both in the full set. The held-out
questions are untouched at every fraction, so coverage varies and the
evaluation does not move with it.

Every cell is registered in a splits registry that the Modal scripts read, so
adding a cell never means editing a training script.

    python scripts/build_perturbations.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arr.utils import read_jsonl, stable_hash, write_json, write_jsonl

BASE_TRAIN = Path("data/arr/ds_critique_qidsplit_train.jsonl")
BASE_DEV = Path("data/arr/ds_critique_qidsplit_dev.jsonl")
REGISTRY = Path("data/arr/splits_registry.json")


def _tag(value: float) -> str:
    return f"{value}".replace(".", "p").rstrip("p")


def pair_flip_rate(original: list[dict], perturbed: list[dict]) -> float:
    """Fraction of within-group ordered pairs whose order the noise reversed."""

    flips = total = 0
    for before, after in zip(original, perturbed):
        a = np.asarray([c["score"] for c in before["candidates"]], dtype=float)
        b = np.asarray([c["score"] for c in after["candidates"]], dtype=float)
        for i, j in itertools.combinations(range(a.size), 2):
            if a[i] == a[j]:
                continue
            total += 1
            if np.sign(a[i] - a[j]) != np.sign(b[i] - b[j]):
                flips += 1
    return flips / total if total else float("nan")


def add_target_noise(groups: list[dict], sigma: float, seed: int) -> list[dict]:
    """Unbiased Gaussian jitter on the reference scores, clipped to [0, 1].

    One draw for the whole file, so every member that trains on it sees exactly
    the same targets. Clipping is preferred to reflection at the boundary: with
    30% of targets sitting at exactly 1.0, reflection roughly doubles the
    induced mean shift. A uniform shift is inert for ListNet anyway, whose loss
    depends on the within-group softmax of the targets.
    """

    generator = np.random.default_rng(seed)
    out = []
    for group in groups:
        clone = json.loads(json.dumps(group))
        for candidate in clone["candidates"]:
            value = float(candidate["score"]) + float(generator.normal(0.0, sigma))
            candidate["score"] = float(np.clip(value, 0.0, 1.0))
            candidate["metadata"] = {
                **(candidate.get("metadata") or {}),
                "target_noise_sigma": sigma,
                "target_noise_seed": seed,
            }
        out.append(clone)
    return out


def nested_subset(groups: list[dict], fraction: float, salt: str) -> list[dict]:
    """Nested qid subsets: the 25% set is contained in the 50% set.

    Ranking every question once by a stable hash and taking a prefix gives
    nesting for free, so the fractions differ only in how much coverage they
    add - not in which questions they happen to sample.
    """

    ordered = sorted(
        groups,
        key=lambda g: stable_hash(salt, (g.get("metadata") or {}).get("qid", g["group_id"])),
    )
    take = max(1, int(round(fraction * len(ordered))))
    keep = {g["group_id"] for g in ordered[:take]}
    return [g for g in groups if g["group_id"] in keep]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", default="0.05,0.1,0.2")
    parser.add_argument("--fractions", default="0.25,0.5")
    parser.add_argument("--noise-seed", type=int, default=20260903)
    parser.add_argument("--salt", default="arr-coverage-v1")
    parser.add_argument("--out-dir", type=Path, default=Path("data/arr/perturbations"))
    args = parser.parse_args(argv)

    base = list(read_jsonl(BASE_TRAIN))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    registry = {
        "published": [
            "data/arr/ds_critique_external_test.jsonl",
            "data/arr/ds_critique_external_dev.jsonl",
        ],
        "qid": [str(BASE_TRAIN), str(BASE_DEV)],
    }
    cells = []

    for sigma in [float(x) for x in args.sigmas.replace(",", " ").split()]:
        # One draw per sigma, shared by every member that reads this file.
        perturbed = add_target_noise(base, sigma, args.noise_seed)
        name = f"qid_sigma{_tag(sigma)}"
        path = args.out_dir / f"{name}_train.jsonl"
        write_jsonl(path, perturbed)
        flip = pair_flip_rate(base, perturbed)
        before = np.mean([c["score"] for g in base for c in g["candidates"]])
        after = np.mean([c["score"] for g in perturbed for c in g["candidates"]])
        registry[name] = [str(path), str(BASE_DEV)]
        cells.append({
            "cell": name, "control": "aleatoric", "sigma": sigma,
            "train_groups": len(perturbed),
            "pair_flip_rate": flip,
            "target_mean_shift": float(after - before),
            "noise_seed": args.noise_seed,
            "shared_across_members": True,
            "advisory": (
                "flip rate above 0.05: partly a change of target concept, not "
                "purely aleatoric" if flip > 0.05 else "within advisory bound"
            ),
        })

    for fraction in [float(x) for x in args.fractions.replace(",", " ").split()]:
        subset = nested_subset(base, fraction, args.salt)
        name = f"qid_frac{_tag(fraction)}"
        path = args.out_dir / f"{name}_train.jsonl"
        write_jsonl(path, subset)
        registry[name] = [str(path), str(BASE_DEV)]
        cells.append({
            "cell": name, "control": "epistemic", "fraction": fraction,
            "train_groups": len(subset),
            "train_candidates": sum(len(g["candidates"]) for g in subset),
            "evaluation_unchanged": str(BASE_DEV),
        })

    # Verify nesting rather than trusting it.
    fracs = sorted(
        [c for c in cells if c["control"] == "epistemic"], key=lambda c: c["fraction"]
    )
    sets = []
    for cell in fracs:
        rows = list(read_jsonl(args.out_dir / f"{cell['cell']}_train.jsonl"))
        sets.append({r["group_id"] for r in rows})
    nested = all(a <= b for a, b in zip(sets, sets[1:])) and all(
        s <= {g["group_id"] for g in base} for s in sets
    )
    if not nested:
        raise ValueError("training fractions are not nested subsets")

    write_json(
        args.out_dir / "perturbation_manifest.json",
        {
            "base_train": str(BASE_TRAIN), "evaluation": str(BASE_DEV),
            "noise_seed": args.noise_seed, "salt": args.salt,
            "nested_verified": nested, "cells": cells,
        },
    )
    write_json(REGISTRY, registry)

    print(f"{'cell':22s} {'control':11s} {'groups':>7s} {'flip%':>7s} {'mean shift':>11s}")
    for cell in cells:
        print(
            f"{cell['cell']:22s} {cell['control']:11s} {cell['train_groups']:7d} "
            f"{cell.get('pair_flip_rate', float('nan')):7.3f} "
            f"{cell.get('target_mean_shift', float('nan')):11.4f}"
        )
    print(f"\nnested subsets verified: {nested}")
    print(f"wrote {REGISTRY} with {len(registry)} splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
