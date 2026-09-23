"""Analyse the fixed-evaluation GPT-4 exposure and data-dose matrix.

The qid100/exposure000 baseline is reused from the completed LOSO campaign.
Every other cell lives under ``runs/r4_exposure/dose``. All comparisons use
the same 54 disjoint QIDs and their three GPT-4 candidates; cells are never
compared through their training or validation predictions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr

from scripts.analyze_r4_loso import _read_final, _records
from src.arr.data import load_groups
from src.arr.metrics import evaluate_predictions, holm_correction
from src.arr.schema import RankingGroup
from src.arr.tier1 import (
    group_bootstrap_partial_spearman,
    group_risk_coverage_curve,
    js_divergence_to_consensus,
    participation_ratio_conventions,
    permutation_null_partial_spearman,
    prepare_listwise_arrays,
    risk_coverage_curve,
)
from src.arr.utils import write_json


SEEDS = (42, 123, 777)
QID_FRACTIONS = ("025", "050", "100")
EXPOSURES = ("000", "050", "100")
CONSTRUCTIONS = ("shared", "independent")
PROVENANCE = "DS_Critique_Bank.explanation_annotations.human_crowd_mean"


def scored_run(root: Path, qid: str, exposure: str, seed: int, construction: str) -> Path:
    """Resolve one scored ensemble, including the separately stored 100/0 baseline."""

    if qid == "100" and exposure == "000":
        base = root / "loso" / f"seed{seed}" / construction
        return base / ("gpt4_holdout" if construction == "shared" else "gpt4_ensemble")
    base = root / "dose" / f"qid{qid}" / f"exposure{exposure}" / f"seed{seed}" / construction
    return base / ("gpt4_holdout" if construction == "shared" else "gpt4_ensemble")


def training_root(root: Path, qid: str, exposure: str, seed: int, construction: str) -> Path:
    if qid == "100" and exposure == "000":
        return root / "loso" / f"seed{seed}" / construction
    return root / "dose" / f"qid{qid}" / f"exposure{exposure}" / f"seed{seed}" / construction


def _per_group_js(member_probabilities: np.ndarray, group_size: int = 3) -> np.ndarray:
    blocks = member_probabilities.reshape(-1, group_size, member_probabilities.shape[1])
    blocks = np.clip(blocks, 1e-12, None)
    blocks /= blocks.sum(axis=1, keepdims=True)
    consensus = blocks.mean(axis=2)
    h_consensus = -np.sum(consensus * np.log(consensus), axis=1)
    h_members = np.mean(-np.sum(blocks * np.log(blocks), axis=1), axis=1)
    return (h_consensus - h_members) / np.log(group_size)


def hierarchical_endpoint_interval(
    initial: np.ndarray, final: np.ndarray, *, samples: int, seed: int
) -> dict[str, float | int]:
    """Bootstrap a paired endpoint change over seeds and whole QIDs."""

    if initial.shape != final.shape or initial.ndim != 3:
        raise ValueError("endpoint arrays must be [seed, group, observation]")
    difference = final - initial
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        seeds = generator.integers(0, difference.shape[0], size=difference.shape[0])
        groups = generator.integers(0, difference.shape[1], size=difference.shape[1])
        draws[index] = difference[np.ix_(seeds, groups, np.arange(difference.shape[2]))].mean()
    return {
        "final_minus_initial": float(difference.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "two_sided_p": float(min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
        "samples": samples,
        "seeds": int(difference.shape[0]),
        "groups": int(difference.shape[1]),
    }


def _audit_training_cell(path: Path, qid: str, exposure: str) -> dict[str, Any]:
    groups = load_groups(path)
    expected_groups = int(round(216 * int(qid) / 100))
    if len(groups) != expected_groups or any(len(group.candidates) != 9 for group in groups):
        raise ValueError(f"unexpected group count/size: {path}")
    exposed = sum(bool(group.metadata["r4_exposed_to_gpt4"]) for group in groups)
    expected_exposed = int(round(expected_groups * int(exposure) / 100))
    if exposed != expected_exposed:
        raise ValueError(f"unexpected exposure count: {path}")
    if {candidate.score_provenance for group in groups for candidate in group.candidates} != {PROVENANCE}:
        raise ValueError(f"non-human target: {path}")
    scores = [candidate.score for group in groups for candidate in group.candidates]
    return {
        "path": str(path), "groups": len(groups), "candidates": len(scores),
        "exposed_qids": exposed, "target_mean": float(np.mean(scores)),
        "fingerprint": groups[0].data_fingerprint,
    }


def analyse(
    root: Path, data_root: Path, *, bootstrap: int, permutations: int,
    response_bootstrap: int,
) -> dict[str, Any]:
    evaluation = load_groups(data_root / "gpt4_qid_holdout.jsonl")
    if len(evaluation) != 54 or any(len(group.candidates) != 3 for group in evaluation):
        raise ValueError("R4 evaluation must contain 54 QIDs x 3 GPT-4 candidates")
    if any(candidate.metadata["student_model"] != "gpt-4-0613"
           for group in evaluation for candidate in group.candidates):
        raise ValueError("R4 evaluation contains a non-GPT-4 candidate")
    if {candidate.score_provenance for group in evaluation
        for candidate in group.candidates} != {PROVENANCE}:
        raise ValueError("R4 evaluation contains a non-human target")
    eval_qids = {group.metadata["qid"] for group in evaluation}
    eval_candidates = {candidate.candidate_id for group in evaluation for candidate in group.candidates}

    cells = []
    cell_audits: dict[tuple[str, str], dict[str, Any]] = {}
    width_values: dict[tuple[str, str, str, int], np.ndarray] = {}
    js_values: dict[tuple[str, str, str, int], np.ndarray] = {}
    for qid in QID_FRACTIONS:
        for exposure in EXPOSURES:
            data = data_root / f"train_qid{qid}_exposure{exposure}.jsonl"
            audit = _audit_training_cell(data, qid, exposure)
            train_groups = load_groups(data)
            if ({group.metadata["qid"] for group in train_groups} & eval_qids
                    or {candidate.candidate_id for group in train_groups
                        for candidate in group.candidates} & eval_candidates):
                raise ValueError(f"training/evaluation leakage in {data}")
            cell_audits[(qid, exposure)] = audit
            for construction in CONSTRUCTIONS:
                for seed in SEEDS:
                    scored = scored_run(root, qid, exposure, seed, construction)
                    train = training_root(root, qid, exposure, seed, construction)
                    final = _read_final(scored / "_final.json")
                    if (final.get("global_seed") != seed or final.get("loss") != "listnet"
                            or final.get("evaluation_fingerprint") != evaluation[0].data_fingerprint
                            or final.get("evaluation_groups") != 54
                            or final.get("evaluation_candidates") != 162):
                        raise ValueError(f"scoring manifest mismatch: {scored}")
                    if construction == "shared":
                        train_final = _read_final(train / "_final.json")
                        train_finals = [train_final]
                    else:
                        ensemble = _read_final(train / "ensemble" / "_final.json")
                        if len(ensemble.get("epochs", [])) != 50:
                            raise ValueError(f"incomplete independent ensemble: {train}")
                        members = sorted(train.glob("member_*_seed*"))
                        if len(members) != 5:
                            raise ValueError(f"wrong independent member count: {train}")
                        train_finals = [_read_final(member / "_final.json") for member in members]
                    if any(item.get("epochs_completed") != 50 or item.get("loss") != "listnet"
                           or item.get("train_fingerprint") != audit["fingerprint"]
                           for item in train_finals):
                        raise ValueError(f"training manifest mismatch: {train}")

                    records = _records(scored / final["prediction_file"], evaluation)
                    arrays = prepare_listwise_arrays(evaluation, records)
                    consensus = arrays.member_probabilities.mean(axis=1)
                    identifiable = [replace(record, score=float(consensus[index]))
                                    for index, record in enumerate(records)]
                    quality = evaluate_predictions(evaluation, identifiable)["aggregate"]
                    interval = group_bootstrap_partial_spearman(
                        arrays, samples=bootstrap, seed=seed
                    )
                    between = permutation_null_partial_spearman(
                        arrays, permutations=permutations, seed=seed, scheme="between_groups"
                    )
                    within = permutation_null_partial_spearman(
                        arrays, permutations=permutations, seed=seed, scheme="within_groups"
                    )
                    candidate_aurc = risk_coverage_curve(arrays, seed=seed)
                    group_aurc = group_risk_coverage_curve(
                        evaluation, identifiable, arrays, seed=seed
                    )
                    js = js_divergence_to_consensus(arrays)
                    pr = participation_ratio_conventions(
                        arrays.member_probabilities - arrays.target_probabilities[:, None]
                    )
                    width_values[(qid, exposure, construction, seed)] = arrays.width.reshape(54, 3)
                    js_values[(qid, exposure, construction, seed)] = _per_group_js(
                        arrays.member_probabilities
                    )[:, None]
                    cells.append({
                        "qid_fraction": int(qid) / 100,
                        "exposure_fraction": int(exposure) / 100,
                        "construction": construction, "seed": seed,
                        "ndcg_at_5": quality["ndcg_at_5"],
                        "random_ndcg_at_5": quality["random_ndcg_at_5"],
                        "ndcg_lift_over_random": quality["ndcg_lift_over_random"],
                        "spearman": quality["spearman"],
                        "rankable_groups": quality["evaluated_query_count"],
                        "mean_probability_width": float(arrays.width.mean()),
                        "js_div_log_c": float(js["js_divergence_normalised_non_singleton"]),
                        "centred_participation_ratio": float(pr["disagreement_participation_ratio"]),
                        "partial_rho": float(interval["estimate"]),
                        "partial_rho_95ci": [float(interval["low"]), float(interval["high"])],
                        "between_p": float(between["two_sided_p_value"]),
                        "within_p": float(within["two_sided_p_value"]),
                        "candidate_aurc_gain": float(candidate_aurc["normalised_aurc_gain"]),
                        "group_aurc_gain": float(group_aurc["normalised_aurc_gain"]),
                        "prediction_path": str(scored / final["prediction_file"]),
                    })

    # C4 uses an intersection-union rule over both nulls and every seed, with
    # Holm correction across the 18 exposure/data/construction conditions.
    condition_p = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in cells:
        key = (f"{round(row['qid_fraction'] * 100):03d}",
               f"{round(row['exposure_fraction'] * 100):03d}", row["construction"])
        grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        if sorted(row["seed"] for row in rows) != list(SEEDS):
            raise ValueError(f"missing seed for {key}")
        condition_p["/".join(key)] = max(max(row["between_p"], row["within_p"]) for row in rows)
    adjusted = holm_correction(condition_p)
    summaries = []
    for key, rows in sorted(grouped.items()):
        name = "/".join(key)
        fields = ("ndcg_at_5", "random_ndcg_at_5", "ndcg_lift_over_random", "spearman",
                  "mean_probability_width", "js_div_log_c", "centred_participation_ratio",
                  "partial_rho", "candidate_aurc_gain", "group_aurc_gain")
        summary = {field: float(np.mean([row[field] for row in rows])) for field in fields}
        summary.update({
            "qid_fraction": int(key[0]) / 100, "exposure_fraction": int(key[1]) / 100,
            "construction": key[2], "seeds": list(SEEDS),
            "intersection_union_p": condition_p[name],
            "holm_p": adjusted[name]["adjusted_p_value"],
            "passes_c4": bool(
                min(row["partial_rho"] for row in rows) > 0 and adjusted[name]["reject"]
            ),
        })
        summaries.append(summary)

    responses = []
    for construction in CONSTRUCTIONS:
        for qid in QID_FRACTIONS:
            for metric, values in (("probability_width", width_values), ("js_div_log_c", js_values)):
                series = [np.asarray([values[(qid, exposure, construction, seed)].mean()
                                      for exposure in EXPOSURES]) for seed in SEEDS]
                initial = np.asarray([values[(qid, "000", construction, seed)] for seed in SEEDS])
                final = np.asarray([values[(qid, "100", construction, seed)] for seed in SEEDS])
                responses.append({
                    "axis": "exposure", "construction": construction,
                    "qid_fraction": int(qid) / 100, "metric": metric,
                    "mean_at_0_50_100": np.mean(series, axis=0).tolist(),
                    "spearman_by_seed": [float(spearmanr((0, .5, 1), row).statistic) for row in series],
                    "nonincreasing_by_seed": [bool(np.all(np.diff(row) <= 0)) for row in series],
                    "endpoint": hierarchical_endpoint_interval(
                        initial, final, samples=response_bootstrap,
                        seed=20260923 + int(qid),
                    ),
                })
        for exposure in EXPOSURES:
            for metric, values in (("probability_width", width_values), ("js_div_log_c", js_values)):
                series = [np.asarray([values[(qid, exposure, construction, seed)].mean()
                                      for qid in QID_FRACTIONS]) for seed in SEEDS]
                initial = np.asarray([values[("025", exposure, construction, seed)] for seed in SEEDS])
                final = np.asarray([values[("100", exposure, construction, seed)] for seed in SEEDS])
                responses.append({
                    "axis": "training_qid_fraction", "construction": construction,
                    "exposure_fraction": int(exposure) / 100, "metric": metric,
                    "mean_at_25_50_100": np.mean(series, axis=0).tolist(),
                    "spearman_by_seed": [float(spearmanr((.25, .5, 1), row).statistic) for row in series],
                    "nonincreasing_by_seed": [bool(np.all(np.diff(row) <= 0)) for row in series],
                    "endpoint": hierarchical_endpoint_interval(
                        initial, final, samples=response_bootstrap,
                        seed=20261023 + int(exposure),
                    ),
                })

    return {
        "status": "complete", "protocol": "arr-r4-exposure-v1",
        "heldout_provenance": "gpt-4-0613", "evaluation_groups": 54,
        "evaluation_candidates": 162, "rankable_groups": 44,
        "score_space": "within_group_softmax_of_member_logits",
        "seeds": list(SEEDS), "bootstrap_samples_per_run": bootstrap,
        "permutations_per_scheme_per_run": permutations,
        "response_bootstrap_samples": response_bootstrap,
        "training_cells": [cell_audits[key] for key in sorted(cell_audits)],
        "cell_summaries": summaries, "responses": responses, "runs": cells,
        "limitations": [
            "GPT-4 as held-out provenance is an implementation choice because the draft leaves the LOSO student TODO.",
            "Replacing known candidates with GPT-4 keeps nine candidates and optimizer steps fixed but shifts the human-target distribution.",
            "Only 44/54 evaluation QIDs have nonconstant human scores for ranking-quality metrics.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/r4_exposure"))
    parser.add_argument("--data-root", type=Path, default=Path("data/arr/r4_exposure"))
    parser.add_argument("--output", type=Path, default=Path("runs/r4_exposure/dose_analysis/results.json"))
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--response-bootstrap", type=int, default=10000)
    args = parser.parse_args()
    result = analyse(args.root, args.data_root, bootstrap=args.bootstrap,
                     permutations=args.permutations,
                     response_bootstrap=args.response_bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Analysed {len(result['runs'])} R4 seed-runs; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
