"""Audit and analyse the completed zero-exposure GPT-4 LOSO experiment.

Source-only LOSO uses the 216 training QIDs with unseen GPT-4 explanations;
QID+source shift uses 54 disjoint QIDs. These are different estimands and are
never pooled. The primary uncertainty read-out is within-group softmax.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.arr.data import load_groups
from src.arr.metrics import evaluate_predictions, holm_correction
from src.arr.schema import RankingGroup, ScoreRecord
from src.arr.tier1 import (
    group_bootstrap_partial_spearman,
    group_risk_coverage_curve,
    js_divergence_to_consensus,
    participation_ratio_conventions,
    permutation_null_partial_spearman,
    prepare_listwise_arrays,
    risk_coverage_curve,
)
from src.arr.utils import read_jsonl, write_json


SEEDS = (42, 123, 777)
EVALUATIONS = {
    "source_only": ("gpt4_loso_train_qids.jsonl", "gpt4_loso", "gpt4_loso_ensemble", 216),
    "qid_and_source": ("gpt4_qid_holdout.jsonl", "gpt4_holdout", "gpt4_ensemble", 54),
}
PROVENANCE = "DS_Critique_Bank.explanation_annotations.human_crowd_mean"


def _read_final(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"incomplete manifest: {path}")
    return value


def _records(path: Path, groups: Sequence[RankingGroup]) -> list[ScoreRecord]:
    values = [ScoreRecord.from_dict(row) for row in read_jsonl(path)]
    expected = {(group.group_id, candidate.candidate_id)
                for group in groups for candidate in group.candidates}
    found = {(value.group_id, value.candidate_id) for value in values}
    if len(values) != len(expected) or found != expected:
        raise ValueError(f"missing, duplicate, or unexpected prediction in {path}")
    fingerprint = groups[0].data_fingerprint
    if any(value.data_fingerprint != fingerprint or value.parsing_status != "ok"
           for value in values):
        raise ValueError(f"fingerprint or parsing failure in {path}")
    by_key = {(value.group_id, value.candidate_id): value for value in values}
    return [by_key[(group.group_id, candidate.candidate_id)]
            for group in groups for candidate in group.candidates]


def audit_data(data_root: Path) -> tuple[list[RankingGroup], dict[str, list[RankingGroup]]]:
    train = load_groups(data_root / "train_qid100_exposure000.jsonl")
    evaluations = {name: load_groups(data_root / spec[0])
                   for name, spec in EVALUATIONS.items()}
    if len(train) != 216 or any(len(group.candidates) != 9 for group in train):
        raise ValueError("zero-exposure training split must be 216 QIDs x 9 candidates")
    if any(candidate.metadata["student_model"] == "gpt-4-0613"
           for group in train for candidate in group.candidates):
        raise ValueError("GPT-4 candidate leaked into training")
    if {candidate.score_provenance for group in train
        for candidate in group.candidates} != {PROVENANCE}:
        raise ValueError("non-human training target")
    qids = lambda groups: {group.metadata["qid"] for group in groups}
    candidates = lambda groups: {candidate.candidate_id for group in groups
                                 for candidate in group.candidates}
    source, shift = evaluations["source_only"], evaluations["qid_and_source"]
    if (qids(train) != qids(source) or qids(train) & qids(shift)
            or candidates(train) & candidates(source)
            or candidates(train) & candidates(shift)):
        raise ValueError("LOSO QID or candidate isolation failed")
    for name, groups in evaluations.items():
        if len(groups) != EVALUATIONS[name][3] or len(qids(groups)) != len(groups):
            raise ValueError(f"wrong QID count in {name}")
        if any(len(group.candidates) != 3 for group in groups):
            raise ValueError(f"evaluation groups in {name} must have three candidates")
        if any(candidate.metadata["student_model"] != "gpt-4-0613"
               for group in groups for candidate in group.candidates):
            raise ValueError(f"non-GPT-4 candidate in {name}")
        if {candidate.score_provenance for group in groups
            for candidate in group.candidates} != {PROVENANCE}:
            raise ValueError(f"non-human target in {name}")
        if len({group.data_fingerprint for group in groups}) != 1:
            raise ValueError(f"fingerprint mismatch within {name}")
    return train, evaluations


def _hierarchical_paired_interval(
    shared: dict[int, dict[str, float]], independent: dict[int, dict[str, float]],
    *, samples: int, seed: int,
) -> dict[str, float | int]:
    """Bootstrap paired QIDs and the three independent training seeds."""

    if set(shared) != set(independent) or set(shared) != set(SEEDS):
        raise ValueError("paired models must share all three seeds")
    common = sorted(set.intersection(*(set(shared[s]) & set(independent[s]) for s in SEEDS)))
    if not common or any(set(shared[s]) != set(common) or set(independent[s]) != set(common)
                         for s in SEEDS):
        raise ValueError("paired models must have identical rankable QIDs")
    differences = np.asarray([[independent[s][qid] - shared[s][qid] for qid in common]
                              for s in SEEDS], dtype=float)
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        seed_indices = generator.integers(0, len(SEEDS), size=len(SEEDS))
        qid_indices = generator.integers(0, len(common), size=len(common))
        draws[index] = differences[np.ix_(seed_indices, qid_indices)].mean()
    return {
        "independent_minus_shared": float(differences.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "two_sided_p": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
        "rankable_qids": len(common),
        "seeds": len(SEEDS),
        "samples": samples,
    }


def analyse(
    root: Path, data_root: Path, *, bootstrap: int = 3000,
    permutations: int = 2000, paired_samples: int = 10000,
) -> dict[str, Any]:
    train, evaluations = audit_data(data_root)
    runs: list[dict[str, Any]] = []
    per_query_lift: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for evaluation_name, groups in evaluations.items():
        _, shared_subdir, independent_subdir, _ = EVALUATIONS[evaluation_name]
        per_query_lift[evaluation_name] = {"shared": {}, "independent": {}}
        for construction in ("shared", "independent"):
            for seed in SEEDS:
                base = root / f"seed{seed}" / construction
                if construction == "shared":
                    train_final = _read_final(base / "_final.json")
                    if (train_final.get("epochs_completed") != 50
                            or train_final.get("loss") != "listnet"
                            or train_final.get("train_fingerprint") != train[0].data_fingerprint):
                        raise ValueError(f"wrong shared training manifest: {base}")
                    prediction_dir = base / shared_subdir
                else:
                    prediction_dir = base / independent_subdir
                    ensemble_final = _read_final(base / "ensemble" / "_final.json")
                    if len(ensemble_final.get("epochs", [])) != 50 or ensemble_final.get("loss") != "listnet":
                        raise ValueError(f"wrong independent training manifest: {base}")
                    member_runs = sorted(base.glob("member_*_seed*"))
                    if len(member_runs) != 5 or any(
                        _read_final(member / "_final.json").get("epochs_completed") != 50
                        or _read_final(member / "_final.json").get("train_fingerprint") != train[0].data_fingerprint
                        for member in member_runs
                    ):
                        raise ValueError(f"incomplete independent members: {base}")
                final = _read_final(prediction_dir / "_final.json")
                if (final.get("global_seed") != seed or final.get("loss") != "listnet"
                        or final.get("evaluation_fingerprint") != groups[0].data_fingerprint
                        or final.get("evaluation_groups") != len(groups)
                        or final.get("evaluation_candidates") != len(groups) * 3):
                    raise ValueError(f"scoring manifest mismatch: {prediction_dir}")
                records = _records(prediction_dir / final["prediction_file"], groups)
                arrays = prepare_listwise_arrays(groups, records)
                if arrays.member_count != 5 or not np.isfinite(arrays.member_probabilities).all():
                    raise ValueError(f"invalid ensemble probabilities: {prediction_dir}")
                saved = np.asarray([record.metadata["group_softmax_scores"]
                                    for record in records], dtype=float)
                # The scorer stores float32 softmax outputs; analysis recomputes
                # them in float64 from bounded member scores.
                if np.max(np.abs(saved - arrays.member_probabilities)) > 1e-5:
                    raise ValueError(f"saved probabilities disagree: {prediction_dir}")
                native_quality = evaluate_predictions(groups, records)
                consensus = arrays.member_probabilities.mean(axis=1)
                identifiable_records = [replace(record, score=float(consensus[index]))
                                        for index, record in enumerate(records)]
                quality = evaluate_predictions(groups, identifiable_records)
                per_query_lift[evaluation_name][construction][seed] = {
                    row["group_id"]: float(row["ndcg_lift_over_random"])
                    for row in quality["per_query"]
                }
                interval = group_bootstrap_partial_spearman(arrays, samples=bootstrap, seed=seed)
                between = permutation_null_partial_spearman(
                    arrays, permutations=permutations, seed=seed, scheme="between_groups"
                )
                within = permutation_null_partial_spearman(
                    arrays, permutations=permutations, seed=seed, scheme="within_groups"
                )
                candidate_aurc = risk_coverage_curve(arrays, seed=seed)
                group_aurc = group_risk_coverage_curve(groups, identifiable_records, arrays, seed=seed)
                pr = participation_ratio_conventions(
                    arrays.member_probabilities - arrays.target_probabilities[:, None]
                )
                js = js_divergence_to_consensus(arrays)
                numeric = [interval["estimate"], interval["low"], interval["high"],
                           between["two_sided_p_value"], within["two_sided_p_value"],
                           candidate_aurc["normalised_aurc_gain"],
                           group_aurc["normalised_aurc_gain"]]
                if not np.isfinite(numeric).all():
                    raise ValueError(f"nonfinite uncertainty statistic: {prediction_dir}")
                runs.append({
                    "evaluation": evaluation_name, "construction": construction, "seed": seed,
                    "groups": len(groups), "rankable_groups": quality["aggregate"]["evaluated_query_count"],
                    "quality_consensus": quality["aggregate"],
                    "quality_native_mean_sigmoid": native_quality["aggregate"],
                    "partial_rho": float(interval["estimate"]),
                    "partial_rho_95ci": [float(interval["low"]), float(interval["high"])],
                    "between_p": float(between["two_sided_p_value"]),
                    "within_p": float(within["two_sided_p_value"]),
                    "candidate_aurc_gain": float(candidate_aurc["normalised_aurc_gain"]),
                    "group_aurc_gain": float(group_aurc["normalised_aurc_gain"]),
                    "mean_probability_width": float(np.mean(arrays.width)),
                    "js_div_log_c": float(js["js_divergence_normalised_non_singleton"]),
                    "centred_participation_ratio": float(pr["disagreement_participation_ratio"]),
                    "prediction_path": str(prediction_dir / final["prediction_file"]),
                })
    summary: dict[str, Any] = {}
    for evaluation_name in EVALUATIONS:
        rows = [row for row in runs if row["evaluation"] == evaluation_name]
        conditions = {}
        family_p = {}
        for construction in ("shared", "independent"):
            subset = [row for row in rows if row["construction"] == construction]
            if len(subset) != len(SEEDS):
                raise ValueError(f"missing seed for {evaluation_name}/{construction}")
            fields = ("partial_rho", "candidate_aurc_gain", "group_aurc_gain",
                      "mean_probability_width", "js_div_log_c", "centred_participation_ratio")
            result = {field: float(np.mean([row[field] for row in subset])) for field in fields}
            result.update({
                "ndcg_at_5": float(np.mean([row["quality_consensus"]["ndcg_at_5"] for row in subset])),
                "random_ndcg_at_5": float(np.mean([row["quality_consensus"]["random_ndcg_at_5"] for row in subset])),
                "ndcg_lift_over_random": float(np.mean([
                    row["quality_consensus"]["ndcg_lift_over_random"] for row in subset
                ])),
                "spearman": float(np.mean([row["quality_consensus"]["spearman"] for row in subset])),
                "rankable_groups": int(subset[0]["rankable_groups"]),
                "seeds": list(SEEDS),
                "partial_rho_range": [float(min(row["partial_rho"] for row in subset)),
                                      float(max(row["partial_rho"] for row in subset))],
                "between_p_range": [float(min(row["between_p"] for row in subset)),
                                    float(max(row["between_p"] for row in subset))],
                "within_p_range": [float(min(row["within_p"] for row in subset)),
                                   float(max(row["within_p"] for row in subset))],
            })
            family_p[construction] = max(
                max(row["between_p"], row["within_p"]) for row in subset
            )
            conditions[construction] = result
        adjusted = holm_correction(family_p)
        for construction, result in conditions.items():
            result["intersection_union_p"] = family_p[construction]
            result["holm_p"] = adjusted[construction]["adjusted_p_value"]
            result["passes_c4"] = bool(
                result["partial_rho_range"][0] > 0 and adjusted[construction]["reject"]
            )
        summary[evaluation_name] = {
            "conditions": conditions,
            "paired_lift_independent_minus_shared": _hierarchical_paired_interval(
                per_query_lift[evaluation_name]["shared"],
                per_query_lift[evaluation_name]["independent"],
                samples=paired_samples, seed=20260921,
            ),
        }
    return {
        "status": "complete", "protocol": "zero-GPT4-exposure-LOSO-v1",
        "training_groups": len(train), "training_candidates": len(train) * 9,
        "seeds": list(SEEDS), "score_space": "within_group_softmax_of_member_logits",
        "uncertainty_control": "maximum_consensus_probability",
        "bootstrap_samples_per_run": bootstrap,
        "permutations_per_scheme_per_run": permutations,
        "multiplicity_rule": "max over both nulls and all 3 seeds; Holm over 2 constructions within each evaluation",
        "interpretation_caution": (
            "The two evaluation sets contain different QIDs and must not be pooled. "
            "Three-candidate NDCG has a high random baseline; compare lift and rankable-group coverage. "
            "The specific held-out student (GPT-4) was an implementation choice pending collaborator confirmation."
        ),
        "summary": summary, "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/r4_exposure/loso"))
    parser.add_argument("--data-root", type=Path, default=Path("data/arr/r4_exposure"))
    parser.add_argument("--output", type=Path, default=Path("runs/r4_exposure/loso_analysis/results.json"))
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--paired-samples", type=int, default=10000)
    args = parser.parse_args()
    result = analyse(args.root, args.data_root, bootstrap=args.bootstrap,
                     permutations=args.permutations, paired_samples=args.paired_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Analysed {len(result['runs'])} scored ensembles; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
