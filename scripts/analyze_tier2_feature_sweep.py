"""Validate and summarise the Pythia fixed-k feature-mask sweep.

Runs only after every Slurm array cell completed. The 80%-retention and
unmasked reference runs are read from the original QID campaign; no result is
silently omitted or imputed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from src.arr.data import load_groups
from src.arr.schema import ScoreRecord
from src.arr.tier1 import (
    _partial_statistic,
    group_risk_coverage_curve,
    js_divergence_to_consensus,
    participation_ratio_conventions,
    prepare_listwise_arrays,
    risk_coverage_curve,
)
from src.arr.utils import write_json


def _read_records(path: Path) -> list[ScoreRecord]:
    with path.open(encoding="utf-8") as handle:
        return [
            ScoreRecord.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        ]


def _ordered_records(groups, records: list[ScoreRecord]) -> list[ScoreRecord]:
    expected = [(g.group_id, c.candidate_id) for g in groups for c in g.candidates]
    lookup = {(r.group_id, r.candidate_id): r for r in records}
    if len(lookup) != len(records) or set(lookup) != set(expected):
        raise ValueError("prediction keys are missing, duplicated or unexpected")
    return [lookup[key] for key in expected]


def _epoch_metrics(groups, records: list[ScoreRecord]) -> dict[str, float]:
    ordered = _ordered_records(groups, records)
    raw = np.asarray(
        [r.metadata["raw_head_scores"] for r in ordered], dtype=np.float64
    )
    probability = np.asarray(
        [r.metadata["group_softmax_scores"] for r in ordered], dtype=np.float64
    )
    if raw.shape[1] != 5 or probability.shape != raw.shape:
        raise ValueError("the feature-mask sweep requires exactly five members")
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(raw, -700, 700)))
    arrays = prepare_listwise_arrays(groups, records)
    residuals = arrays.member_probabilities - arrays.target_probabilities[:, None]
    pr = participation_ratio_conventions(residuals)
    js = js_divergence_to_consensus(arrays)
    values = {
        "raw_width": float(np.ptp(raw, axis=1).mean()),
        "sigmoid_width": float(np.ptp(sigmoid, axis=1).mean()),
        "probability_width": float(np.ptp(probability, axis=1).mean()),
        "participation_ratio": float(pr["disagreement_participation_ratio"]),
        "js_normalised": float(js["js_divergence_normalised_non_singleton"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("non-finite uncertainty metric")
    return values


def _run_summary(path: Path, groups, *, seed: int, loss: str, arm: str,
                 mask_label: str, k: int | None) -> dict:
    final_path = path / "_final.json"
    if not final_path.is_file():
        raise ValueError(f"missing completed run: {final_path}")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("status") != "complete"
        or final.get("epochs_completed") != 50
        or final.get("head_count") != 5
        or final.get("global_seed") != seed
        or final.get("loss") != loss
        or final.get("arm") != arm
        or final.get("validation_fingerprint") != groups[0].data_fingerprint
    ):
        raise ValueError(f"incompatible or incomplete run: {final_path}")
    if k is not None and (
        final.get("feature_keep_count_requested") != k
        or final.get("feature_keep_count_effective") != min(512, k)
    ):
        raise ValueError(f"incorrect fixed-k mask in {final_path}")
    if k is None and "feature_keep_count_requested" in final:
        raise ValueError(f"reference run unexpectedly has fixed-k mask: {final_path}")

    epochs = []
    last_records = None
    for epoch in range(50):
        prediction_file = path / f"validation_predictions_epoch_{epoch}.jsonl"
        if not prediction_file.is_file():
            raise ValueError(f"missing epoch prediction: {prediction_file}")
        records = _read_records(prediction_file)
        epochs.append(_epoch_metrics(groups, records))
        if epoch == 49:
            last_records = records
    assert last_records is not None
    arrays = prepare_listwise_arrays(groups, last_records)
    candidate_aurc = risk_coverage_curve(arrays, seed=seed)
    group_aurc = group_risk_coverage_curve(
        groups, last_records, arrays, seed=seed
    )
    widths = [row["probability_width"] for row in epochs]
    c3 = float(spearmanr(range(1, 51), widths).statistic)
    if not np.isfinite(c3):
        raise ValueError(f"undefined width/epoch correlation: {path}")
    quality = final["final_validation"]
    return {
        "seed": seed,
        "loss": loss,
        "arm": arm,
        "mask": mask_label,
        "k": k,
        "run_path": str(path),
        "ndcg_at_5": float(quality["tie_aware_ndcg_at_5"]),
        "ndcg_lift": float(quality["ndcg_lift_over_random"]),
        "probability_width_epoch_1": widths[0],
        "probability_width_epoch_50": widths[-1],
        "probability_width_change": (widths[-1] - widths[0]) / widths[0],
        "raw_width_epoch_50": epochs[-1]["raw_width"],
        "sigmoid_width_epoch_50": epochs[-1]["sigmoid_width"],
        "js_normalised_epoch_50": epochs[-1]["js_normalised"],
        "participation_ratio_epoch_50": epochs[-1]["participation_ratio"],
        "participation_ratio_min": min(
            row["participation_ratio"] for row in epochs
        ),
        "c3_width_epoch_spearman": c3,
        "partial_width_error_spearman": float(
            _partial_statistic(arrays, arrays.width)
        ),
        "candidate_aurc_gain": float(candidate_aurc["normalised_aurc_gain"]),
        "group_aurc_gain": float(group_aurc["normalised_aurc_gain"]),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["loss"], row["arm"], row["mask"], row["k"])].append(row)
    fields = (
        "ndcg_at_5", "ndcg_lift", "probability_width_epoch_1",
        "probability_width_epoch_50", "probability_width_change",
        "raw_width_epoch_50", "sigmoid_width_epoch_50",
        "js_normalised_epoch_50", "participation_ratio_epoch_50",
        "participation_ratio_min", "c3_width_epoch_spearman",
        "partial_width_error_spearman", "candidate_aurc_gain",
        "group_aurc_gain",
    )
    result = []
    for (loss, arm, mask, k), members in sorted(
        grouped.items(), key=lambda item: (
            item[0][0], item[0][1], -1 if item[0][3] is None else item[0][3],
            item[0][2],
        )
    ):
        if sorted(row["seed"] for row in members) != [42, 123, 777]:
            raise ValueError(f"incomplete seed set for {loss}/{arm}/{mask}")
        item = {"loss": loss, "arm": arm, "mask": mask, "k": k,
                "seeds": [42, 123, 777]}
        for field in fields:
            values = np.asarray([row[field] for row in members], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {field} for {loss}/{arm}/{mask}")
            item[f"{field}_mean"] = float(values.mean())
            item[f"{field}_sd"] = float(values.std(ddof=1))
        result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/tier2_pythia"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/arr/tier2_pythia.yaml")
    )
    parser.add_argument("--output", type=Path,
                        default=Path("runs/tier2_pythia/feature_sweep_analysis"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    seeds = experiment["seeds"]
    losses = experiment["losses"]
    counts = experiment["feature_keep_counts"]
    if seeds != [42, 123, 777] or losses != ["listnet", "listmle", "mse"]:
        raise ValueError("feature sweep seed/loss matrix changed unexpectedly")
    if counts != [20, 100, 180, 260, 340]:
        raise ValueError("feature sweep counts differ from the Slurm array")
    groups = load_groups(config["data"]["validation"])
    if len(groups) != 54 or any(len(group.candidates) != 12 for group in groups):
        raise ValueError("expected the 54×12 QID held-out split")

    rows = []
    for seed in seeds:
        for loss in losses:
            reference_root = args.root / f"shared_ablation_qid_50ep_seed{seed}" / loss
            sweep_root = args.root / f"shared_feature_k_sweep_qid_50ep_seed{seed}" / loss
            for arm in ("baseline", "features", "bootstrap_features"):
                label = "unmasked" if arm == "baseline" else "fraction_0p8"
                rows.append(_run_summary(
                    reference_root / arm, groups, seed=seed, loss=loss,
                    arm=arm, mask_label=label, k=None,
                ))
            for arm in ("features", "bootstrap_features"):
                for k in counts:
                    rows.append(_run_summary(
                        sweep_root / f"{arm}_k{k}", groups, seed=seed, loss=loss,
                        arm=arm, mask_label=f"k_{k}", k=k,
                    ))
    if len(rows) != 117:
        raise ValueError(f"expected 90 sweep plus 27 reference runs, got {len(rows)}")
    aggregates = _aggregate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "per_run.json", rows)
    write_json(args.output / "aggregate.json", {
        "status": "complete", "heldout_groups": len(groups),
        "heldout_candidates": sum(len(group.candidates) for group in groups),
        "validation_fingerprint": groups[0].data_fingerprint,
        "sweep_runs": 90, "reference_runs": 27, "conditions": aggregates,
    })
    with (args.output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps({"status": "complete", "runs": len(rows),
                      "conditions": len(aggregates), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
