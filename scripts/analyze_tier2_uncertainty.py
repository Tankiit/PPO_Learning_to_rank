"""Audit saved Pythia QID predictions and run the ARR R1--R3 analyses.

No model is loaded or trained here. The primary family consists of the 27
predefined main-matrix conditions (three objectives, nine constructions/arms)
on three seeds. The fixed-k mask sweep is reported as exploratory R1/R2 only;
selecting its best k on the held-out QIDs would invalidate a confirmatory test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from src.arr.data import load_groups
from src.arr.losses import get_loss, listmle_loss
from src.arr.schema import RankingGroup, ScoreRecord
from src.arr.tier1 import (
    ListwiseArrays,
    group_bootstrap_partial_spearman,
    group_risk_coverage_curve,
    js_divergence_to_consensus,
    participation_ratio_conventions,
    permutation_null_partial_spearman,
    risk_coverage_curve,
)
from src.arr.utils import write_json


SEEDS = (42, 123, 777)
LOSSES = ("listnet", "listmle", "mse")
MAIN_SHARED_ARMS = (
    "baseline", "bootstrap", "features", "bootstrap_features",
    "lambda_0p01", "lambda_0p1", "lambda_1",
)
MAIN_INDEPENDENT_ARMS = ("independent", "independent_bootstrap")
EXPECTED_MAIN_RUNS = len(SEEDS) * len(LOSSES) * (
    len(MAIN_SHARED_ARMS) + len(MAIN_INDEPENDENT_ARMS)
)
EXPECTED_SWEEP_RUNS = len(SEEDS) * len(LOSSES) * 2 * 5


@dataclass(frozen=True)
class Run:
    name: str
    loss: str
    arm: str
    seed: int
    family: str
    prediction_dir: Path
    final: dict[str, Any]

    @property
    def condition(self) -> str:
        return f"{self.loss}/{self.arm}"


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=axis, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def discover_runs(root: Path, *, require_complete: bool = True) -> list[Run]:
    logs = root / "cril_logs"
    runs = []
    for path in sorted(logs.rglob("_final.json")):
        relative = path.parent.relative_to(logs)
        top = relative.parts[0]
        if top.startswith("independent_backbones"):
            if path.parent.name != "ensemble":
                continue
            family = "main"
            prediction_dir = root / "independent_predictions" / relative
        elif top.startswith("shared_ablation_qid_"):
            family = "main"
            prediction_dir = root / "final_predictions" / relative
        elif top.startswith("shared_feature_k_sweep_qid_"):
            family = "sweep"
            prediction_dir = root / "final_predictions" / relative
        else:
            continue
        final = json.loads(path.read_text(encoding="utf-8"))
        if final.get("status") != "complete":
            raise ValueError(f"incomplete run: {path}")
        loss, arm, seed = final["loss"], final["arm"], int(final["global_seed"])
        if family == "sweep":
            arm = path.parent.name
            base_arm, _, count = arm.rpartition("_k")
            if (
                base_arm != final["arm"]
                or not count.isdecimal()
                or int(count) != final.get("feature_keep_count_effective")
            ):
                raise ValueError(f"sweep path/mask manifest mismatch: {path}")
        if loss not in LOSSES or seed not in SEEDS:
            raise ValueError(f"unexpected objective/seed: {path}")
        if family == "main" and arm not in MAIN_SHARED_ARMS + MAIN_INDEPENDENT_ARMS:
            raise ValueError(f"unexpected primary arm: {path}")
        if family == "sweep" and not (arm.startswith("features_k") or arm.startswith("bootstrap_features_k")):
            raise ValueError(f"unexpected sweep arm: {path}")
        epochs = final.get("epochs_completed", len(final.get("epochs", [])))
        if epochs != 50:
            raise ValueError(f"run has {epochs} rather than 50 epochs: {path}")
        final_prediction = prediction_dir / final["final_prediction_file"]
        if not final_prediction.is_file():
            raise ValueError(f"missing final predictions: {final_prediction}")
        runs.append(Run(str(relative), loss, arm, seed, family, prediction_dir, final))
    if require_complete:
        counts = {family: sum(run.family == family for run in runs) for family in ("main", "sweep")}
        if counts != {"main": EXPECTED_MAIN_RUNS, "sweep": EXPECTED_SWEEP_RUNS}:
            raise ValueError(f"incomplete campaign: {counts}")
        grouped = defaultdict(set)
        for run in runs:
            if run.seed in grouped[(run.family, run.condition)]:
                raise ValueError(f"duplicate seed for {run.family}/{run.condition}")
            grouped[(run.family, run.condition)].add(run.seed)
        if any(seeds != set(SEEDS) for seeds in grouped.values()):
            raise ValueError("at least one condition is missing a seed")
    return runs


def audit_inputs(
    root: Path, runs: Sequence[Run], train: Sequence[RankingGroup],
    heldout: Sequence[RankingGroup],
) -> dict[str, Any]:
    """Check provenance, QID disjointness, and the exact archived run matrix."""

    if len(train) != 216 or len(heldout) != 54:
        raise ValueError("expected 216 training and 54 held-out QIDs")
    if any(len(group.candidates) != 12 for group in (*train, *heldout)):
        raise ValueError("every QID must contain 12 explanations")
    train_ids = {str(group.metadata["qid"]) for group in train}
    heldout_ids = {str(group.metadata["qid"]) for group in heldout}
    if len(train_ids) != 216 or len(heldout_ids) != 54 or train_ids & heldout_ids:
        raise ValueError("train/held-out QIDs overlap or repeat")
    provenance = {candidate.score_provenance for group in (*train, *heldout)
                  for candidate in group.candidates}
    if provenance != {"DS_Critique_Bank.explanation_annotations.human_crowd_mean"}:
        raise ValueError(f"non-human target provenance: {provenance}")
    if any(group.data_fingerprint != heldout[0].data_fingerprint for group in heldout):
        raise ValueError("held-out data fingerprints differ")
    checksums = {}
    for run in runs:
        if run.final.get("validation_fingerprint") != heldout[0].data_fingerprint:
            raise ValueError(f"run fingerprint differs: {run.name}")
        path = run.prediction_dir / run.final["final_prediction_file"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        checksums[run.name] = digest.hexdigest()
        if run.final["construction"] == "independent_ensemble":
            present = list(run.prediction_dir.glob("validation_predictions_epoch_*.jsonl"))
            if len(present) != 50:
                raise ValueError(f"independent ensemble has {len(present)} epoch files: {run.name}")
    return {
        "status": "complete", "train_qids": len(train_ids),
        "heldout_qids": len(heldout_ids), "heldout_candidates": len(heldout) * 12,
        "overlap_qids": 0, "target_provenance": sorted(provenance),
        "main_runs": EXPECTED_MAIN_RUNS, "exploratory_sweep_runs": EXPECTED_SWEEP_RUNS,
        "independent_ensemble_epoch_files": 18 * 50,
        "shared_final_prediction_files": 153,
        "validation_fingerprint": heldout[0].data_fingerprint,
        "final_prediction_sha256_by_run": checksums,
    }


def load_prediction_matrices(
    path: Path, groups: Sequence[RankingGroup], expected_fingerprint: str
) -> tuple[list[ScoreRecord], np.ndarray, np.ndarray]:
    """Return ordered records and [group,candidate,member] raw/probability arrays."""

    expected = [(group.group_id, candidate.candidate_id)
                for group in groups for candidate in group.candidates]
    found: dict[tuple[str, str], ScoreRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = ScoreRecord.from_dict(json.loads(line))
            key = (record.group_id, record.candidate_id)
            if key in found:
                raise ValueError(f"duplicate prediction key {key} in {path}")
            if record.data_fingerprint != expected_fingerprint or record.parsing_status != "ok":
                raise ValueError(f"fingerprint/parsing failure in {path}")
            found[key] = record
    if set(found) != set(expected):
        raise ValueError(f"missing or unexpected candidates in {path}")
    ordered = [found[key] for key in expected]
    sizes = {len(group.candidates) for group in groups}
    if sizes != {12}:
        raise ValueError(f"expected 12 candidates per group, got {sizes}")
    raw = np.asarray([r.metadata["raw_head_scores"] for r in ordered], dtype=np.float64)
    saved_probability = np.asarray(
        [r.metadata["group_softmax_scores"] for r in ordered], dtype=np.float64
    )
    if raw.shape != (len(groups) * 12, 5) or saved_probability.shape != raw.shape:
        raise ValueError(f"wrong score shape in {path}: {raw.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(saved_probability).all():
        raise ValueError(f"non-finite prediction in {path}")
    raw = raw.reshape(len(groups), 12, 5)
    saved_probability = saved_probability.reshape(raw.shape)
    probability = _softmax(raw, axis=1)
    if np.max(np.abs(saved_probability - probability)) > 5e-6:
        raise ValueError(f"saved softmax does not match raw scores in {path}")
    return ordered, raw, probability


def _target(groups: Sequence[RankingGroup]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray([[c.score for c in group.candidates] for group in groups], dtype=np.float64)
    return scores, _softmax(scores, axis=1)


def _loss_value(raw: np.ndarray, target: np.ndarray, name: str) -> float:
    scores = torch.from_numpy(raw)
    labels = torch.from_numpy(target)
    mask = torch.ones_like(labels, dtype=torch.bool)
    values = []
    for member in range(raw.shape[-1]):
        if name == "listmle":
            generator = torch.Generator().manual_seed(20260920)
            value = listmle_loss(scores[:, :, member], labels, mask, generator=generator)
        else:
            value = get_loss(name)(scores[:, :, member], labels, mask)
        values.append(float(value))
    return float(np.mean(values))


def r1_gauge_control(raw: np.ndarray, target: np.ndarray, loss: str) -> dict[str, float | str]:
    """Use a per-query, per-member shift; preserve listwise loss and rankings."""

    generator = np.random.default_rng(20260920)
    offsets = generator.uniform(-1.5, 1.5, size=(raw.shape[0], 1, raw.shape[2]))
    moved = raw + offsets
    probability = _softmax(raw, axis=1)
    shifted_probability = _softmax(moved, axis=1)
    centred = raw - raw.mean(axis=1, keepdims=True)
    shifted_centred = moved - moved.mean(axis=1, keepdims=True)
    before_loss = _loss_value(raw, target, loss)
    after_loss = _loss_value(moved, target, loss)
    def width(value: np.ndarray) -> float:
        return float(np.ptp(value, axis=2).mean())
    return {
        "control": "per_group_per_member_additive_shift",
        "loss_before": before_loss,
        "loss_after": after_loss,
        "loss_absolute_change": abs(after_loss - before_loss),
        "ranking_preserved": bool(np.array_equal(np.argsort(raw, axis=1), np.argsort(moved, axis=1))),
        "raw_width_before": width(raw),
        "raw_width_after": width(moved),
        "sigmoid_width_before": width(_sigmoid(raw)),
        "sigmoid_width_after": width(_sigmoid(moved)),
        "centred_sigmoid_width_before": width(_sigmoid(centred)),
        "centred_sigmoid_width_after": width(_sigmoid(shifted_centred)),
        "softmax_width_before": width(probability),
        "softmax_width_after": width(shifted_probability),
        "softmax_max_absolute_change": float(np.max(np.abs(probability - shifted_probability))),
    }


def _arrays(
    groups: Sequence[RankingGroup], probability: np.ndarray, target_probability: np.ndarray
) -> ListwiseArrays:
    consensus = probability.mean(axis=2)
    confidence = np.repeat(np.max(consensus, axis=1), 12)
    entropy = np.repeat(
        -np.sum(consensus * np.log(np.clip(consensus, 1e-12, None)), axis=1), 12
    )
    return ListwiseArrays(
        member_probabilities=probability.reshape(-1, 5),
        target_probabilities=target_probability.reshape(-1),
        width=np.ptp(probability, axis=2).reshape(-1),
        absolute_error=np.abs(consensus - target_probability).reshape(-1),
        confidence=confidence,
        entropy=entropy,
        group_index=np.repeat(np.arange(len(groups)), 12),
        group_sizes=np.full(len(groups), 12, dtype=int),
        group_ids=tuple(group.group_id for group in groups),
    )


def r2_diversity(arrays: ListwiseArrays) -> dict[str, float]:
    residual = arrays.member_probabilities - arrays.target_probabilities[:, None]
    centred = residual - residual.mean(axis=1, keepdims=True)
    covariance = np.cov(centred, rowvar=False, ddof=0)
    pr = participation_ratio_conventions(residual)
    js = js_divergence_to_consensus(arrays)
    return {
        "participation_ratio": float(pr["disagreement_participation_ratio"]),
        "participation_fraction": float(pr["participation_fraction"]),
        "js_normalised": float(js["js_divergence_normalised_non_singleton"]),
        "probability_width": float(arrays.width.mean()),
        "covariance_trace": float(np.trace(covariance)),
        "covariance_frobenius": float(np.linalg.norm(covariance)),
    }


def synthetic_controls() -> dict[str, Any]:
    """Deterministic shape/scale sanity check in the same 12-way space."""

    values = np.arange(12, dtype=np.float64)
    base = np.sin(values[:, None] * (np.arange(5)[None, :] + 1) / 3.0)
    diverse = _softmax(base[None, :, :], axis=1)
    consensus = diverse.mean(axis=2, keepdims=True)
    cases = {
        "diverse": diverse,
        "collapsed": np.repeat(consensus, 5, axis=2),
        "affine_0p1": consensus + 0.1 * (diverse - consensus),
    }
    rows = {}
    for name, probability in cases.items():
        target = np.full((1, 12), 1.0 / 12)
        arrays = ListwiseArrays(
            member_probabilities=probability.reshape(12, 5),
            target_probabilities=target.reshape(-1),
            width=np.ptp(probability, axis=2).reshape(-1),
            absolute_error=np.abs(probability.mean(axis=2) - target).reshape(-1),
            confidence=np.full(12, float(np.max(probability.mean(axis=2)))),
            entropy=np.full(12, float(-np.sum(probability.mean(axis=2) * np.log(probability.mean(axis=2))))),
            group_index=np.zeros(12, dtype=int), group_sizes=np.asarray([12]),
            group_ids=("synthetic",),
        )
        rows[name] = r2_diversity(arrays)
    if rows["collapsed"]["js_normalised"] > 1e-12:
        raise AssertionError("collapsed synthetic ensemble must have zero JS")
    if not rows["affine_0p1"]["js_normalised"] < rows["diverse"]["js_normalised"]:
        raise AssertionError("affine shrinkage must reduce JS")
    return rows


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for one predeclared family of conditions."""

    ordered = sorted(p_values, key=lambda key: p_values[key])
    previous = 0.0
    adjusted = {}
    for index, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * p_values[key]))
        adjusted[key] = previous
    return adjusted


def _summary_by_condition(rows: list[dict[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["condition"])].append(row)
    summaries = []
    for (family, condition), members in sorted(grouped.items()):
        if sorted(item["seed"] for item in members) != list(SEEDS):
            raise ValueError(f"missing seed in {family}/{condition}")
        result: dict[str, Any] = {"family": family, "condition": condition, "seeds": list(SEEDS)}
        for field in fields:
            values = np.asarray([member[field] for member in members], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {field} in {family}/{condition}")
            result[field + "_mean"] = float(values.mean())
            result[field + "_sd"] = float(values.std(ddof=1))
        summaries.append(result)
    return summaries


def run_r1_r2(runs: Sequence[Run], groups: Sequence[RankingGroup], output: Path) -> None:
    target, target_probability = _target(groups)
    rows = []
    for run in runs:
        file = run.prediction_dir / run.final["final_prediction_file"]
        _, raw, probability = load_prediction_matrices(file, groups, groups[0].data_fingerprint)
        r1 = r1_gauge_control(raw, target, run.loss)
        if run.loss != "mse" and r1["loss_absolute_change"] > 1e-9:
            raise AssertionError(f"listwise gauge changed the loss for {run.name}")
        if run.loss == "mse" and r1["loss_absolute_change"] < 1e-6:
            raise AssertionError(f"MSE unexpectedly acted shift-invariant for {run.name}")
        if r1["softmax_max_absolute_change"] > 1e-12:
            raise AssertionError(f"listwise gauge changed softmax for {run.name}")
        r2 = r2_diversity(_arrays(groups, probability, target_probability))
        rows.append({
            "run": run.name, "family": run.family, "condition": run.condition,
            "seed": run.seed, "loss": run.loss,
            "ndcg_at_5": float(run.final["final_validation"]["tie_aware_ndcg_at_5"]),
            **r1, **r2,
        })
    fields = (
        "ndcg_at_5", "raw_width_before", "raw_width_after", "sigmoid_width_before",
        "sigmoid_width_after", "centred_sigmoid_width_before",
        "centred_sigmoid_width_after", "softmax_width_before", "softmax_width_after",
        "loss_absolute_change", "participation_ratio", "participation_fraction",
        "js_normalised", "probability_width", "covariance_trace", "covariance_frobenius",
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "r1_r2.json", {
        "status": "complete", "main_runs": EXPECTED_MAIN_RUNS,
        "exploratory_sweep_runs": EXPECTED_SWEEP_RUNS,
        "heldout_groups": len(groups), "heldout_candidates": len(groups) * 12,
        "readout_note": (
            "ListNet/ListMLE permit per-query per-member additive shifts; "
            "raw and uncentred sigmoid widths are gauge-dependent. "
            "Centred sigmoid and group softmax are shift-invariant. "
            "MSE has no such loss-preserving shift."
        ),
        "synthetic_controls": synthetic_controls(),
        "conditions": _summary_by_condition(rows, fields),
        "runs": rows,
    })


def run_trajectories(
    root: Path, runs: Sequence[Run], groups: Sequence[RankingGroup], output: Path
) -> None:
    """Recompute the read-outs at every epoch, not just selected checkpoints."""

    _, target_probability = _target(groups)
    rows = []
    curves = []
    for run in runs:
        directory = (
            run.prediction_dir if run.final["construction"] == "independent_ensemble"
            else root / "all_epoch_predictions" / run.name
        )
        per_epoch = []
        for epoch_index in range(50):
            path = directory / f"validation_predictions_epoch_{epoch_index}.jsonl"
            if not path.is_file():
                raise ValueError(f"missing epoch file: {path}")
            _, raw, probability = load_prediction_matrices(
                path, groups, groups[0].data_fingerprint
            )
            arrays = _arrays(groups, probability, target_probability)
            diversity = r2_diversity(arrays)
            centred = raw - raw.mean(axis=1, keepdims=True)
            per_epoch.append({
                "epoch": epoch_index + 1,
                "raw_width": float(np.ptp(raw, axis=2).mean()),
                "sigmoid_width": float(np.ptp(_sigmoid(raw), axis=2).mean()),
                "centred_sigmoid_width": float(np.ptp(_sigmoid(centred), axis=2).mean()),
                **diversity,
            })
        width = np.asarray([point["probability_width"] for point in per_epoch])
        trend = float(spearmanr(np.arange(1, 51), width).statistic)
        if not np.isfinite(trend):
            raise ValueError(f"undefined C3 trajectory: {run.name}")
        rows.append({
            "run": run.name, "family": run.family, "condition": run.condition,
            "seed": run.seed, "c3_spearman": trend,
            "probability_width_change": float((width[-1] - width[0]) / width[0]),
            "epoch_1": per_epoch[0], "epoch_10": per_epoch[9],
            "epoch_25": per_epoch[24], "epoch_50": per_epoch[49],
        })
        curves.append({"run": run.name, "family": run.family,
                       "condition": run.condition, "seed": run.seed,
                       "epochs": per_epoch})
    summary_fields = (
        "raw_width", "sigmoid_width", "centred_sigmoid_width", "probability_width",
        "participation_ratio", "js_normalised", "covariance_trace",
    )
    by_condition_epoch: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for curve in curves:
        for point in curve["epochs"]:
            by_condition_epoch[(curve["family"], curve["condition"], point["epoch"])].append(point)
    condition_curves = []
    for (family, condition, epoch), points in sorted(by_condition_epoch.items()):
        if len(points) != len(SEEDS):
            raise ValueError(f"incomplete trajectory seeds for {family}/{condition}/{epoch}")
        row: dict[str, Any] = {"family": family, "condition": condition, "epoch": epoch}
        for field in summary_fields:
            values = np.asarray([point[field] for point in points])
            row[field + "_mean"] = float(values.mean())
            row[field + "_sd"] = float(values.std(ddof=1))
        condition_curves.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "trajectories.json", {
        "status": "complete", "epoch_count": 50, "run_count": len(rows),
        "table_4_epochs": [1, 10, 25, 50],
        "c3_definition": "Spearman(epoch, mean within-group probability width)",
        "conditions": _summary_by_condition(rows, ("c3_spearman", "probability_width_change")),
        "condition_curves": condition_curves,
        "runs": rows, "curves": curves,
    })


def run_r3(
    runs: Sequence[Run], groups: Sequence[RankingGroup], output: Path,
    *, bootstrap: int, permutations: int,
) -> None:
    _, target_probability = _target(groups)
    rows = []
    for run in runs:
        if run.family != "main":
            continue
        file = run.prediction_dir / run.final["final_prediction_file"]
        records, _, probability = load_prediction_matrices(file, groups, groups[0].data_fingerprint)
        arrays = _arrays(groups, probability, target_probability)
        interval = group_bootstrap_partial_spearman(
            arrays, samples=bootstrap, seed=run.seed
        )
        between = permutation_null_partial_spearman(
            arrays, permutations=permutations, seed=run.seed, scheme="between_groups"
        )
        within = permutation_null_partial_spearman(
            arrays, permutations=permutations, seed=run.seed, scheme="within_groups"
        )
        candidate = risk_coverage_curve(arrays, seed=run.seed)
        group = group_risk_coverage_curve(groups, records, arrays, seed=run.seed)
        if not all(np.isfinite(value) for value in (
            interval["estimate"], interval["low"], interval["high"],
            between["two_sided_p_value"], within["two_sided_p_value"],
            candidate["normalised_aurc_gain"], group["normalised_aurc_gain"],
        )):
            raise ValueError(f"non-finite R3 statistic for {run.name}")
        rows.append({
            "run": run.name, "family": run.family, "condition": run.condition,
            "seed": run.seed, "partial_rho": float(interval["estimate"]),
            "bootstrap_low": float(interval["low"]),
            "bootstrap_high": float(interval["high"]),
            "between_p": float(between["two_sided_p_value"]),
            "within_p": float(within["two_sided_p_value"]),
            "candidate_aurc_gain": float(candidate["normalised_aurc_gain"]),
            "group_aurc_gain": float(group["normalised_aurc_gain"]),
            "candidate_risk_coverage": candidate,
            "group_risk_coverage": group,
        })
    if len(rows) != EXPECTED_MAIN_RUNS:
        raise ValueError(f"expected {EXPECTED_MAIN_RUNS} primary R3 runs, got {len(rows)}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    family_p = {
        condition: max(max(item["between_p"], item["within_p"]) for item in members)
        for condition, members in grouped.items()
    }
    adjusted = holm(family_p)
    summary = _summary_by_condition(
        rows, ("partial_rho", "candidate_aurc_gain", "group_aurc_gain")
    )
    for item in summary:
        members = grouped[item["condition"]]
        item["intersection_union_p"] = family_p[item["condition"]]
        item["holm_p"] = adjusted[item["condition"]]
        item["positive_all_seeds"] = all(row["partial_rho"] > 0 for row in members)
        item["passes_both_nulls_after_holm"] = bool(
            item["positive_all_seeds"] and item["holm_p"] < 0.05
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "r3.json", {
        "status": "complete", "family": "27_predeclared_main_conditions",
        "heldout_groups": len(groups), "singleton_groups": 0,
        "bootstrap_samples_per_run": bootstrap,
        "permutations_per_scheme_per_run": permutations,
        "multiplicity_rule": (
            "For each condition, take max p over both permutation schemes and all "
            "three seeds (intersection-union), then Holm-adjust across 27 conditions. "
            "A positive association must also hold for all seeds."
        ),
        "conditions": summary, "runs": rows,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/tier2_pythia"))
    parser.add_argument("--train-data", type=Path, default=Path("data/arr/ds_critique_qidsplit_train.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("data/arr/ds_critique_qidsplit_dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/tier2_pythia/uncertainty_analysis"))
    parser.add_argument("--stage", choices=("r1r2", "r3", "trajectory", "all"), default="all")
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    groups = load_groups(args.data)
    if len(groups) != 54 or any(len(group.candidates) != 12 for group in groups):
        raise ValueError("expected 54 held-out QIDs x 12 explanations")
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("duplicate held-out QID")
    runs = discover_runs(args.root)
    train = load_groups(args.train_data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "input_audit.json", audit_inputs(args.root, runs, train, groups))
    if args.stage in {"r1r2", "all"}:
        run_r1_r2(runs, groups, args.output)
    if args.stage in {"r3", "all"}:
        run_r3(runs, groups, args.output, bootstrap=args.bootstrap,
               permutations=args.permutations)
    if args.stage in {"trajectory", "all"}:
        run_trajectories(args.root, runs, groups, args.output)
    print(json.dumps({"status": "complete", "stage": args.stage,
                      "runs": len(runs), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
