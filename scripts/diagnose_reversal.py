"""Why does the ensemble rank e-SNLI at Spearman -0.703?

A degraded ranker under distribution shift sits near zero. A large negative
correlation on 1,000 groups is a competent ranker with its sign inverted, and
there are two very different explanations.

H1  Orientation bug. The DS-Critique and e-SNLI references are mapped in
    opposite directions somewhere between the loaders and the evaluator. Every
    OOD number computed so far would then be measured against an inverted
    error and is uninterpretable until recomputed.

H2  Concept mismatch. DS-Critique Bank ranks *critiques*: text that names errors
    and contradictions scores well. e-SNLI's degraded tiers are written in
    exactly that register ("the label is neutral because ..."), while its gold
    tier is a plain explanation. A competent critique-quality model would then
    genuinely prefer the degraded tiers, and e-SNLI is not a shifted version of
    this task at all - it is a different target concept.

Test B discriminates them on saved predictions alone, and the discriminator is
where the *nonsense* tier lands:

    H1 predicts a clean monotone reversal - nonsense ranked best of all, because
       its reference score is the lowest.
    H2 predicts nonsense ranked worst or near-worst - gibberish is not a
       critique - with poor/fair lifted above gold.

Usage:
    python scripts/diagnose_reversal.py --volume <mirror> [--arms ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr

from src.arr.utils import read_jsonl, write_json

TIER_ORDER = ("gold", "good", "fair", "poor", "nonsense")

ARMS = {
    "independent_listnet": "independent_backbones_50ep_v1/runs/independent/ensemble/evaluation_esnli/predictions.jsonl",
    "independent_listnet_bootstrap": "independent_backbones_50ep_v1/runs/independent_bootstrap/ensemble/evaluation_esnli/predictions.jsonl",
    "independent_mse": "independent_backbones_mse_50ep_v1/runs/independent/ensemble/evaluation_esnli/predictions.jsonl",
    "independent_mse_bootstrap": "independent_backbones_mse_50ep_v1/runs/independent_bootstrap/ensemble/evaluation_esnli/predictions.jsonl",
    "shared_baseline": "shared_ablation_50ep_v1/listnet/baseline/evaluation_esnli/predictions.jsonl",
    "shared_bootstrap": "shared_ablation_50ep_v1/listnet/bootstrap/evaluation_esnli/predictions.jsonl",
    "shared_features": "shared_ablation_50ep_v1/listnet/features/evaluation_esnli/predictions.jsonl",
    "shared_bootstrap_features": "shared_ablation_50ep_v1/listnet/bootstrap_features/evaluation_esnli/predictions.jsonl",
    "shared_lambda_0.01": "shared_ablation_50ep_v1/listnet/lambda_0p01/evaluation_esnli/predictions.jsonl",
    "shared_lambda_0.1": "shared_ablation_50ep_v1/listnet/lambda_0p1/evaluation_esnli/predictions.jsonl",
    "shared_lambda_1.0": "shared_ablation_50ep_v1/listnet/lambda_1/evaluation_esnli/predictions.jsonl",
    "mc_dropout_k8": "shared_ablation_50ep_v1/listnet/baseline/evaluation_mc8_esnli/predictions.jsonl",
}



def load_groups_raw(path: Path) -> dict:
    return {row["group_id"]: row for row in read_jsonl(path)}


def test_b(groups: dict, predictions: dict) -> dict:
    """Per-tier predicted rank and score, plus the nonsense discriminator."""

    per_tier_score: dict[str, list[float]] = defaultdict(list)
    per_tier_rank: dict[str, list[float]] = defaultdict(list)
    per_tier_reference: dict[str, list[float]] = defaultdict(list)
    tier_rank_pairs: list[tuple[int, float]] = []
    nonsense_best = 0
    nonsense_worst = 0
    groups_seen = 0

    for group_id, group in groups.items():
        candidates = group["candidates"]
        tiers = [c["metadata"].get("quality_tier", "unknown") for c in candidates]
        scores = np.asarray(
            [predictions[(group_id, c["candidate_id"])] for c in candidates], dtype=float
        )
        reference = np.asarray([float(c["score"]) for c in candidates], dtype=float)
        # Rank 1 = highest predicted score inside this group.
        order = np.argsort(-scores, kind="stable")
        rank = np.empty(len(candidates), dtype=float)
        rank[order] = np.arange(1, len(candidates) + 1)
        for tier, score, position, ref in zip(tiers, scores, rank, reference):
            per_tier_score[tier].append(float(score))
            per_tier_rank[tier].append(float(position))
            per_tier_reference[tier].append(float(ref))
            if tier in TIER_ORDER:
                tier_rank_pairs.append((TIER_ORDER.index(tier), float(position)))
        if "nonsense" in tiers:
            groups_seen += 1
            nonsense_position = rank[tiers.index("nonsense")]
            nonsense_best += int(nonsense_position == 1)
            nonsense_worst += int(nonsense_position == len(candidates))

    tiers = [t for t in TIER_ORDER if t in per_tier_score]
    summary = {
        tier: {
            "n": len(per_tier_score[tier]),
            "mean_predicted_score": float(np.mean(per_tier_score[tier])),
            "mean_predicted_rank": float(np.mean(per_tier_rank[tier])),
            "mean_reference_score": float(np.mean(per_tier_reference[tier])),
        }
        for tier in tiers
    }
    index = np.asarray([p[0] for p in tier_rank_pairs], dtype=float)
    positions = np.asarray([p[1] for p in tier_rank_pairs], dtype=float)
    return {
        "per_tier": summary,
        "tier_index_vs_predicted_rank_spearman": float(
            spearmanr(index, positions).correlation
        ),
        "nonsense_groups": groups_seen,
        "nonsense_ranked_best_fraction": nonsense_best / max(groups_seen, 1),
        "nonsense_ranked_worst_fraction": nonsense_worst / max(groups_seen, 1),
        "predicted_rank_order": sorted(tiers, key=lambda t: summary[t]["mean_predicted_rank"]),
        "reference_rank_order": sorted(
            tiers, key=lambda t: -summary[t]["mean_reference_score"]
        ),
    }


def reference_orientation(groups: dict) -> dict:
    """H1's first prediction, checked on the reference file alone.

    If the e-SNLI reference were inverted anywhere upstream, the stored scores
    would run against the quality tiers. They do not have to be perfectly
    monotone - the tier sampling ranges overlap by construction - but the
    ordering of tier means is unambiguous.
    """

    per_tier: dict[str, list[float]] = defaultdict(list)
    for group in groups.values():
        for candidate in group["candidates"]:
            tier = candidate["metadata"].get("quality_tier", "unknown")
            per_tier[tier].append(float(candidate["score"]))
    means = {t: float(np.mean(per_tier[t])) for t in TIER_ORDER if t in per_tier}
    ordered = sorted(means, key=lambda t: -means[t])
    return {
        "tier_mean_reference": means,
        "reference_order": ordered,
        "correctly_oriented": ordered == [t for t in TIER_ORDER if t in means],
    }


def gold_exclusion(groups: dict, predictions: dict) -> dict:
    """H1's second prediction: an inverted reference cannot be positive anywhere.

    The gold tier is the only human-written candidate in each group; the other
    four are template-generated. Recomputing the per-group correlation with gold
    removed separates 'the model ranks quality backwards' from 'the model
    rejects the human-written candidate'.
    """

    full, without_gold, without_nonsense = [], [], []
    gold_last = 0
    for group_id, group in groups.items():
        candidates = group["candidates"]
        tiers = [c["metadata"].get("quality_tier") for c in candidates]
        if "gold" not in tiers:
            continue
        reference = np.asarray([float(c["score"]) for c in candidates], dtype=float)
        predicted = np.asarray(
            [predictions[(group_id, c["candidate_id"])] for c in candidates], dtype=float
        )
        full.append(spearmanr(predicted, reference).correlation)
        gold_index = tiers.index("gold")
        gold_last += int(predicted[gold_index] == predicted.min())
        for drop, sink in (("gold", without_gold), ("nonsense", without_nonsense)):
            if drop not in tiers:
                continue
            keep = np.array([i != tiers.index(drop) for i in range(len(candidates))])
            if np.std(reference[keep]) > 1e-12 and np.std(predicted[keep]) > 1e-12:
                sink.append(spearmanr(predicted[keep], reference[keep]).correlation)
    mean = lambda values: float(np.nanmean(values)) if values else float("nan")
    return {
        "spearman_all_candidates": mean(full),
        "spearman_excluding_gold": mean(without_gold),
        "spearman_excluding_nonsense": mean(without_nonsense),
        "gold_ranked_last_fraction": gold_last / max(len(full), 1),
        "n_groups": len(full),
    }


def text_profile(groups: dict) -> dict:
    """Length and provenance per tier, to test the obvious confound."""

    per_tier: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for group in groups.values():
        for candidate in group["candidates"]:
            tier = candidate["metadata"].get("quality_tier", "unknown")
            per_tier[tier].append(
                (len(candidate["text"]), bool(candidate["metadata"].get("human_text")))
            )
    return {
        tier: {
            "mean_characters": float(np.mean([x[0] for x in per_tier[tier]])),
            "human_text_fraction": float(np.mean([x[1] for x in per_tier[tier]])),
        }
        for tier in TIER_ORDER
        if tier in per_tier
    }


def verdict(result: dict, exclusion: dict) -> str:
    """Describe this arm's pattern. H1 is not an per-arm verdict.

    H1 is a claim about the shared reference file and the shared evaluator, so
    it is settled once by ``reference_orientation`` and by whether *any* arm
    shows a positive correlation on a subset. An earlier version of this
    function offered "H1 SUPPORTED" per arm whenever nonsense happened to lead,
    which contradicted the global check on the same data. What varies between
    arms is only how much of the reversal the gold rejection accounts for.
    """

    full = exclusion["spearman_all_candidates"]
    residual = exclusion["spearman_excluding_gold"]
    gold_last = exclusion["gold_ranked_last_fraction"]
    if gold_last > 0.5:
        rejection = f"rejects the human-written gold candidate in {gold_last:.0%} of groups"
    else:
        rejection = f"ranks gold last in only {gold_last:.0%} of groups"
    if residual > 0.05:
        residual_note = (
            f"and is POSITIVELY correlated ({residual:+.3f}) among the four "
            "template tiers, so the gold rejection accounts for the whole reversal"
        )
    elif residual > -0.05:
        residual_note = (
            f"and is uncorrelated ({residual:+.3f}) among the four template "
            "tiers, so the gold rejection accounts for the whole reversal"
        )
    else:
        residual_note = (
            f"and remains negative ({residual:+.3f}) among the four template "
            "tiers, so gold rejection explains only part of the reversal"
        )
    return f"overall rho {full:+.3f}; {rejection} {residual_note}."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--output", type=Path, default=Path("runs/tier1"))
    args = parser.parse_args(argv)

    groups = load_groups_raw(
        args.volume / "independent_backbones_50ep_v1/data/esnli_test.jsonl"
    )
    print(f"e-SNLI: {len(groups)} groups\n")

    orientation = reference_orientation(groups)
    print("TEST A (cheap form) - reference orientation in the file itself:")
    for tier, value in orientation["tier_mean_reference"].items():
        print(f"   {tier:10s} mean reference score {value:.4f}")
    print(f"   reference order: {' > '.join(orientation['reference_order'])}")
    print(
        "   correctly oriented: "
        f"{orientation['correctly_oriented']} "
        "(H1 requires this to be False)\n"
    )
    profile = text_profile(groups)
    print("Text profile by tier:")
    for tier, row in profile.items():
        print(
            f"   {tier:10s} {row['mean_characters']:7.1f} chars   "
            f"human_text {row['human_text_fraction']:.2f}"
        )
    print()

    results = {}
    for name in args.arms:
        path = args.volume / ARMS[name]
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        predictions = {
            (row["group_id"], row["candidate_id"]): float(row["score"])
            for row in read_jsonl(path)
        }
        result = test_b(groups, predictions)
        exclusion = gold_exclusion(groups, predictions)
        result["gold_exclusion"] = exclusion
        results[name] = result
        print(f"=== {name} ===")
        print(f"{'tier':10s} {'n':>6s} {'mean pred':>10s} {'mean rank':>10s} {'mean ref':>9s}")
        for tier, row in result["per_tier"].items():
            print(
                f"{tier:10s} {row['n']:6d} {row['mean_predicted_score']:10.4f} "
                f"{row['mean_predicted_rank']:10.3f} {row['mean_reference_score']:9.4f}"
            )
        print(f"predicted order (best first): {' > '.join(result['predicted_rank_order'])}")
        print(f"reference order (best first): {' > '.join(result['reference_rank_order'])}")
        print(
            f"nonsense ranked best in {result['nonsense_ranked_best_fraction']:.1%} "
            f"of groups, worst in {result['nonsense_ranked_worst_fraction']:.1%}"
        )
        print(
            f"rho all={exclusion['spearman_all_candidates']:+.3f}  "
            f"excl. gold={exclusion['spearman_excluding_gold']:+.3f}  "
            f"excl. nonsense={exclusion['spearman_excluding_nonsense']:+.3f}  "
            f"gold last in {exclusion['gold_ranked_last_fraction']:.1%}"
        )
        print(f"VERDICT: {verdict(result, exclusion)}\n")

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "reversal_diagnosis.json",
        {
            "reference_orientation": orientation,
            "text_profile": profile,
            "arms": results,
        },
    )
    print(f"wrote {args.output / 'reversal_diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
