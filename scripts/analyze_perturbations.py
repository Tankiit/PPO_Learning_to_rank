"""Score the perturbation controls against their pre-committed predictions.

Each control is an intervention with a direction declared in advance in
`configs/arr/perturbation.yaml`. This script measures the direction actually
observed and prints PASS or FAIL per prediction. A FAIL is a result, not a bug:
the aleatoric control failing means the estimator conflates aleatoric and
epistemic uncertainty, and saying so is strictly better than having a reviewer
discover it.

Five statistics, all on the fixed qid held-out set:

``consensus_entropy``   flatness of the ensemble consensus - the aleatoric proxy.
``js_to_consensus``     Jensen-Shannon divergence among members - the epistemic
                        magnitude statistic.
``pr_deviation``        participation ratio of row-centred member disagreement.
``pr_residual``         participation ratio of the RAW residual covariance.
``ndcg_heldout``        tie-aware NDCG@5 on the held-out questions.

The two participation ratios are both reported because they answer different
questions and only one of them can move under Proposition 2. Subtracting the
target is a per-example constant, so it vanishes under row-centring: the centred
statistic is mathematically identical on residuals and on raw member scores.
Only the uncentred form can register a growing shared error term.

Verdicts use the relative change from the axis baseline to its extreme, with a
threshold stated here rather than chosen after seeing the data: a statistic
counts as moving if it changes by more than 20% of its baseline, and as flat
otherwise. The monotone trend across all levels is reported alongside.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr

from src.arr.data import load_groups
from src.arr.metrics import evaluate_predictions
from src.arr.schema import ScoreRecord
from src.arr.tier1 import (
    consensus_entropy,
    js_divergence_to_consensus,
    participation_ratio_conventions,
    prepare_listwise_arrays,
    uncentred_participation_ratio,
)
from src.arr.utils import read_jsonl, write_json

VOLUME = "ppo-ltr-epistemic-runs"
RELATIVE_THRESHOLD = 0.20
ARMS = {
    "independent_listnet": (
        "independent_backbones_{cell}_50ep_seed{seed}"
        "/runs/independent/ensemble/validation_predictions_epoch_49.jsonl"
    ),
    "shared_listnet": (
        "shared_ablation_{cell}_50ep_seed{seed}"
        "/listnet/baseline/validation_predictions_epoch_49.jsonl"
    ),
}


def _parses(path: Path) -> bool:
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                json.loads(line)
    except Exception:
        return False
    return True


def fetch(mirror: Path, remote: str) -> Path | None:
    local = mirror / remote
    if local.exists() and local.stat().st_size > 0 and _parses(local):
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["modal", "volume", "get", VOLUME, remote, str(local), "--force"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not local.exists() or not _parses(local):
        local.unlink(missing_ok=True)
        return None
    return local


def statistics(groups, records) -> dict[str, float]:
    arrays = prepare_listwise_arrays(groups, records)
    residuals = arrays.member_probabilities - arrays.target_probabilities[:, None]
    return {
        "consensus_entropy": consensus_entropy(arrays)["consensus_entropy_normalised"],
        "js_to_consensus": js_divergence_to_consensus(arrays)[
            "js_divergence_normalised_non_singleton"
        ],
        "pr_deviation": participation_ratio_conventions(residuals)["participation_fraction"],
        "pr_residual": uncentred_participation_ratio(residuals)["participation_fraction"],
        "ndcg_heldout": evaluate_predictions(groups, records)["aggregate"][
            "tie_aware_ndcg_at_5"
        ],
    }


def verdict(levels: list[float], values: list[float], predicted: str) -> dict:
    """Compare the observed direction with the one declared in advance."""

    finite = [(x, v) for x, v in zip(levels, values) if np.isfinite(v)]
    if len(finite) < 2:
        return {"predicted": predicted, "observed": "insufficient data", "pass": None}
    xs = [x for x, _ in finite]
    ys = [v for _, v in finite]
    baseline = ys[0]
    relative = (ys[-1] - baseline) / abs(baseline) if abs(baseline) > 1e-12 else float("nan")
    trend = float(spearmanr(xs, ys).correlation) if len(finite) > 2 else float("nan")
    if not np.isfinite(relative):
        observed = "undefined"
    elif relative > RELATIVE_THRESHOLD:
        observed = "increases"
    elif relative < -RELATIVE_THRESHOLD:
        observed = "decreases"
    else:
        observed = "flat"
    if predicted == "collapses at every fraction, no dose-response":
        passed = observed == "flat"
    else:
        passed = observed == predicted
    return {
        "predicted": predicted,
        "observed": observed,
        "relative_change": relative,
        "trend_spearman": trend,
        "levels": xs,
        "values": ys,
        "pass": bool(passed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/arr/perturbation.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/perturbations"))
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    seeds = config["seeds"]
    groups = load_groups(config["evaluation"])
    print(
        f"held-out: {len(groups)} groups / "
        f"{sum(len(g.candidates) for g in groups)} candidates\n"
    )

    axes = {
        "aleatoric": {
            "levels": config["aleatoric"]["target_noise_sigma"],
            "cell": lambda v: "qid" if v == 0.0 else f"qid_sigma{str(v).replace('.', 'p')}",
            "predictions": config["aleatoric"]["predictions"],
        },
        "epistemic": {
            "levels": config["epistemic"]["train_qid_fraction"],
            "cell": lambda v: "qid" if v == 1.0 else f"qid_frac{str(v).replace('.', 'p')}",
            "predictions": config["epistemic"]["predictions"],
        },
    }

    results: dict = {}
    missing: list[str] = []
    for axis, spec in axes.items():
        results[axis] = {}
        for arm, template in ARMS.items():
            per_level: dict[str, list[float]] = {}
            used_levels = []
            for level in spec["levels"]:
                cell = spec["cell"](level)
                per_seed: list[dict[str, float]] = []
                for seed in seeds:
                    remote = template.format(cell=cell, seed=seed)
                    path = fetch(args.mirror, remote)
                    if path is None:
                        missing.append(f"{axis}/{arm}/{cell}/seed{seed}")
                        continue
                    records = [ScoreRecord.from_dict(r) for r in read_jsonl(path)]
                    per_seed.append(statistics(groups, records))
                if not per_seed:
                    continue
                used_levels.append(level)
                for key in per_seed[0]:
                    per_level.setdefault(key, []).append(
                        float(np.mean([s[key] for s in per_seed]))
                    )
            if not used_levels:
                continue
            predictions = dict(spec["predictions"])
            if arm == "shared_listnet" and axis == "epistemic":
                predictions.update(config["epistemic"]["shared_listnet_prediction"])
            checks = {
                name: verdict(used_levels, per_level.get(name, []), predicted)
                for name, predicted in predictions.items()
            }
            results[axis][arm] = {"levels": used_levels, "statistics": per_level, "checks": checks}

            print(f"=== {axis} / {arm}   levels={used_levels}")
            for name, values in per_level.items():
                print(f"   {name:22s} " + "  ".join(f"{v:9.4f}" for v in values))
            for name, check in checks.items():
                mark = "PASS" if check["pass"] else ("n/a " if check["pass"] is None else "FAIL")
                print(
                    f"   [{mark}] {name:22s} predicted {check['predicted']:>10s}, "
                    f"observed {check['observed']:>10s} "
                    f"(relative {check.get('relative_change', float('nan')):+.2%})"
                )
            print()

    if missing:
        print(f"not yet available: {len(missing)} cells")
        for item in missing[:12]:
            print(f"   {item}")
        if len(missing) > 12:
            print(f"   ... and {len(missing) - 12} more")

    if any(results[a] for a in results):
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "perturbation_results.json",
            {"missing": missing, "threshold": RELATIVE_THRESHOLD, "axes": results},
        )
        print(f"\nwrote {args.output / 'perturbation_results.json'}")
        return 0
    print("\nno perturbation cells available yet")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
