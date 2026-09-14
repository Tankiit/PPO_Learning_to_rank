"""Recompute the Tier 1 claims from saved predictions. No GPU, no retraining.

Every arm is read from prediction JSONL files produced by the 50-epoch Modal
runs. The script answers four questions the previous report left open:

1. What is ranking quality once the 141 singleton in-domain groups cannot
   inflate it, and what is it on e-SNLI where every group has five candidates?
2. Is the OOD width/error association larger than the group structure alone
   produces, under a permutation null as well as a bootstrap interval?
3. Which participation-ratio convention is a value like 4.91 approaching, and
   is the diversity statistic a KL or a Jensen--Shannon divergence?
4. Does abstaining on the widest candidates actually reduce error?

Usage::

    python scripts/tier1_recompute.py --volume <downloaded-volume-root> \
        --dev data/arr/ds_critique_external_dev.jsonl \
        --output runs/tier1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arr.data import load_groups
from src.arr.schema import ScoreRecord
from src.arr.tier1 import (
    group_bootstrap_partial_spearman,
    group_risk_coverage_curve,
    js_divergence_to_consensus,
    ndcg_by_group_size,
    permutation_null_partial_spearman,
    prepare_listwise_arrays,
    participation_ratio_conventions,
    risk_coverage_curve,
)
from src.arr.utils import read_jsonl, write_json


SHARED_ARMS = (
    ("shared_baseline", "baseline"),
    ("shared_bootstrap", "bootstrap"),
    ("shared_features", "features"),
    ("shared_bootstrap_features", "bootstrap_features"),
    ("shared_lambda_0.01", "lambda_0p01"),
    ("shared_lambda_0.1", "lambda_0p1"),
    ("shared_lambda_1.0", "lambda_1"),
)
INDEPENDENT_ARMS = (
    ("independent_listnet", "independent"),
    ("independent_listnet_bootstrap", "independent_bootstrap"),
)
# The MSE ensembles are scored in the same within-group probability space as the
# ListNet ones so the two are comparable. NDCG is unaffected by that transform
# (it is monotone within a group), but note that pointwise MSE has no score
# gauge to fix, so its width statistics would also be admissible on raw scores.
INDEPENDENT_MSE_ARMS = (
    ("independent_mse", "independent"),
    ("independent_mse_bootstrap", "independent_bootstrap"),
)


def _arm_paths(volume: Path) -> dict[str, dict[str, Path]]:
    shared = volume / "shared_ablation_50ep_v1/listnet"
    independent = volume / "independent_backbones_50ep_v1/runs"
    paths: dict[str, dict[str, Path]] = {}
    for name, directory in INDEPENDENT_ARMS:
        base = independent / directory / "ensemble"
        paths[name] = {
            "in_domain": base / "validation_predictions_epoch_49.jsonl",
            "esnli": base / "evaluation_esnli/predictions.jsonl",
        }
    mse = volume / "independent_backbones_mse_50ep_v1/runs"
    for name, directory in INDEPENDENT_MSE_ARMS:
        base = mse / directory / "ensemble"
        paths[name] = {
            "in_domain": base / "validation_predictions_epoch_49.jsonl",
            "esnli": base / "evaluation_esnli/predictions.jsonl",
        }
    for name, directory in SHARED_ARMS:
        base = shared / directory
        paths[name] = {
            "in_domain": base / "validation_predictions_epoch_49.jsonl",
            "esnli": base / "evaluation_esnli/predictions.jsonl",
        }
    paths["mc_dropout_k8"] = {
        "in_domain": shared / "baseline/evaluation_mc8_in_domain/predictions.jsonl",
        "esnli": shared / "baseline/evaluation_mc8_esnli/predictions.jsonl",
    }
    return paths


def _load(path: Path) -> list[ScoreRecord]:
    return [ScoreRecord.from_dict(row) for row in read_jsonl(path)]


def _analyse_split(
    groups, records, *, bootstrap_samples: int, permutations: int, seed: int
) -> dict:
    arrays = prepare_listwise_arrays(groups, records)
    residuals = arrays.member_probabilities - arrays.target_probabilities[:, None]
    result = {
        "n_groups": arrays.group_count,
        "n_observations": arrays.candidate_count,
        "member_count": arrays.member_count,
        "ndcg": ndcg_by_group_size(groups, records),
        "participation_ratio": participation_ratio_conventions(residuals),
        "diversity": js_divergence_to_consensus(arrays),
        "risk_coverage_candidate": risk_coverage_curve(arrays, seed=seed),
        "risk_coverage_group": group_risk_coverage_curve(
            groups, records, arrays, seed=seed
        ),
        "mean_width": float(np.mean(arrays.width)),
        "mean_absolute_error": float(np.mean(arrays.absolute_error)),
    }
    for control in ("confidence", "entropy"):
        result[f"partial_spearman_given_{control}"] = {
            "bootstrap": group_bootstrap_partial_spearman(
                arrays, control=control, samples=bootstrap_samples, seed=seed
            ),
            "permutation_null_between_groups": permutation_null_partial_spearman(
                arrays,
                control=control,
                permutations=permutations,
                seed=seed,
                scheme="between_groups",
            ),
            "permutation_null_within_groups": permutation_null_partial_spearman(
                arrays,
                control=control,
                permutations=permutations,
                seed=seed,
                scheme="within_groups",
            ),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument(
        "--dev", type=Path, default=Path("data/arr/ds_critique_external_dev.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/tier1"))
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these arms")
    args = parser.parse_args(argv)

    dev_groups = load_groups(args.dev)
    esnli_groups = load_groups(
        args.volume / "independent_backbones_50ep_v1/data/esnli_test.jsonl"
    )
    print(
        f"in-domain: {len(dev_groups)} groups / "
        f"{sum(len(g.candidates) for g in dev_groups)} candidates; "
        f"e-SNLI: {len(esnli_groups)} groups / "
        f"{sum(len(g.candidates) for g in esnli_groups)} candidates"
    )

    paths = _arm_paths(args.volume)
    if args.only:
        paths = {name: value for name, value in paths.items() if name in args.only}

    # A singleton group has a one-candidate softmax, so its width and its error
    # are both identically zero. Left in, 141 such rows out of 270 dominate every
    # in-domain correlation and risk-coverage curve: sorting by width parks them
    # all at the front with zero risk. They are reported separately, never mixed.
    dev_non_singleton = [g for g in dev_groups if len(g.candidates) >= 2]
    splits = (
        ("in_domain", dev_groups),
        ("in_domain_non_singleton", dev_non_singleton),
        ("esnli", esnli_groups),
    )
    print(
        f"in-domain non-singleton subset: {len(dev_non_singleton)} groups / "
        f"{sum(len(g.candidates) for g in dev_non_singleton)} candidates"
    )

    results: dict[str, dict] = {}
    for name, split_paths in paths.items():
        results[name] = {}
        for split, groups in splits:
            path = split_paths["in_domain" if split.startswith("in_domain") else split]
            if not path.exists():
                print(f"  {name:32s} {split:10s} MISSING {path}")
                continue
            started = time.time()
            records = _load(path)
            wanted = {g.group_id for g in groups}
            records = [r for r in records if r.group_id in wanted]
            results[name][split] = _analyse_split(
                groups,
                records,
                bootstrap_samples=args.bootstrap_samples,
                permutations=args.permutations,
                seed=args.seed,
            )
            results[name][split]["source"] = str(
                path.relative_to(args.volume)
            )
            summary = results[name][split]
            partial = summary["partial_spearman_given_confidence"]
            ndcg = summary["ndcg"]["non_singleton"]
            print(
                f"  {name:30s} {split:23s} "
                f"NDCG={ndcg['tie_aware_ndcg_at_5']:.4f} "
                f"(rand {ndcg['random_ndcg_at_5']:.4f}, lift {ndcg['ndcg_lift_over_random']:+.3f}) "
                f"n={ndcg['evaluated_query_count']:4d} "
                f"rho={partial['bootstrap']['estimate']:+.4f} "
                f"[{partial['bootstrap']['low']:+.4f},{partial['bootstrap']['high']:+.4f}] "
                f"p_between={partial['permutation_null_between_groups']['two_sided_p_value']:.4f} "
                f"p_within={partial['permutation_null_within_groups']['two_sided_p_value']:.4f} "
                f"AURC={summary['risk_coverage_candidate']['normalised_aurc_gain']:+.3f} "
                f"({time.time() - started:.0f}s)"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": {
            "volume_root": str(args.volume),
            "dev_data": str(args.dev),
            "bootstrap_samples": args.bootstrap_samples,
            "permutations": args.permutations,
            "seed": args.seed,
            "in_domain_groups": len(dev_groups),
            "esnli_groups": len(esnli_groups),
        },
        "arms": results,
    }
    write_json(args.output / "tier1_results.json", payload)
    print(f"\nwrote {args.output / 'tier1_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
