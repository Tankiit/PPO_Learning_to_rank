"""Build QID uncertainty tables and diagnostic figures from completed JSON only.

No scientific cell is entered manually. The plotted conditions are fixed in
the code to show shared/independent and ListNet/ListMLE/MSE controls; they are
not selected by their measured outcome. These are QID-holdout diagnostics, not
the paper's missing unseen-provenance results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOTTED = (
    "listnet/baseline", "listnet/independent",
    "listmle/independent", "mse/independent",
)
COLOURS = ("#365d8d", "#d89032", "#4a9459", "#aa4f71")


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"incomplete analysis: {path}")
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_tables(r1r2: dict, trajectories: dict, r3: dict, output: Path) -> None:
    r1 = {row["condition"]: row for row in r1r2["conditions"] if row["family"] == "main"}
    trend = {row["condition"]: row for row in trajectories["conditions"] if row["family"] == "main"}
    functional = {row["condition"]: row for row in r3["conditions"]}
    if len(r1) != 27 or set(r1) != set(trend) or set(r1) != set(functional):
        raise ValueError("R1/R2/R3 condition families differ or are incomplete")
    rows = []
    for condition in sorted(r1):
        shape, curve, error = r1[condition], trend[condition], functional[condition]
        rows.append({
            "condition": condition, "seeds": 3,
            "ndcg_at_5_mean": shape["ndcg_at_5_mean"],
            "ndcg_at_5_sd": shape["ndcg_at_5_sd"],
            "centred_participation_ratio_mean": shape["participation_ratio_mean"],
            "js_over_log_c_mean": shape["js_normalised_mean"],
            "probability_width_mean": shape["probability_width_mean"],
            "covariance_trace_mean": shape["covariance_trace_mean"],
            "c3_spearman_mean": curve["c3_spearman_mean"],
            "width_change_epoch1_to50": curve["probability_width_change_mean"],
            "partial_width_error_rho_mean": error["partial_rho_mean"],
            "candidate_aurc_gain_mean": error["candidate_aurc_gain_mean"],
            "group_aurc_gain_mean": error["group_aurc_gain_mean"],
            "r3_intersection_union_p": error["intersection_union_p"],
            "r3_holm_p": error["holm_p"],
            "r3_passes_both_nulls_all_seeds": error["passes_both_nulls_after_holm"],
        })
    _write_csv(output / "primary_qid_table.csv", rows)
    selected_epochs = {1, 10, 25, 50}
    readouts = []
    for row in trajectories["condition_curves"]:
        if row["family"] != "main" or row["epoch"] not in selected_epochs:
            continue
        readouts.append({
            "condition": row["condition"], "epoch": row["epoch"], "seeds": 3,
            "raw_width_mean": row["raw_width_mean"],
            "raw_width_sd": row["raw_width_sd"],
            "sigmoid_width_mean": row["sigmoid_width_mean"],
            "sigmoid_width_sd": row["sigmoid_width_sd"],
            "centred_sigmoid_width_mean": row["centred_sigmoid_width_mean"],
            "centred_sigmoid_width_sd": row["centred_sigmoid_width_sd"],
            "softmax_width_mean": row["probability_width_mean"],
            "softmax_width_sd": row["probability_width_sd"],
        })
    if len(readouts) != 27 * 4:
        raise ValueError("incomplete R1 read-out checkpoint table")
    _write_csv(output / "r1_readouts_epochs_1_10_25_50.csv", readouts)


def plot_shape_scale(trajectories: dict, output: Path) -> None:
    curves = {
        row["condition"]: row for row in trajectories["condition_curves"]
        if row["family"] == "main" and row["condition"] in PLOTTED[:2]
    }
    by_condition = defaultdict(list)
    for row in trajectories["condition_curves"]:
        if row["family"] == "main" and row["condition"] in PLOTTED[:2]:
            by_condition[row["condition"]].append(row)
    if any(len(by_condition[condition]) != 50 for condition in PLOTTED[:2]):
        raise ValueError("missing trajectory for figure")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for condition, colour, label in (
        ("listnet/baseline", COLOURS[0], "Shared backbone, five heads"),
        ("listnet/independent", COLOURS[1], "Five independent backbones"),
    ):
        rows = sorted(by_condition[condition], key=lambda row: row["epoch"])
        epochs = np.asarray([row["epoch"] for row in rows])
        for axis, field in zip(axes, ("participation_ratio", "js_normalised")):
            mean = np.asarray([row[field + "_mean"] for row in rows])
            sd = np.asarray([row[field + "_sd"] for row in rows])
            axis.plot(epochs, mean, label=label, color=colour, linewidth=1.8)
            axis.fill_between(epochs, np.maximum(mean - sd, 1e-12), mean + sd,
                              color=colour, alpha=0.17, linewidth=0)
    axes[0].set(title="Shape: centred participation ratio", ylabel="PR (ceiling 4)",
                xlabel="Epoch", ylim=(0, 4.1))
    axes[1].set(title="Scale: JS / log(12)", ylabel="Normalized JS",
                xlabel="Epoch")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"qid_shape_vs_scale.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(r3: dict, output: Path) -> None:
    grouped = defaultdict(list)
    for row in r3["runs"]:
        if row["condition"] in PLOTTED:
            grouped[row["condition"]].append(row)
    if any(len(grouped[condition]) != 3 for condition in PLOTTED):
        raise ValueError("missing seeds for risk--coverage figure")
    fig, axes = plt.subplots(4, 2, figsize=(8.4, 10.0), sharex=True)
    for row_index, condition in enumerate(PLOTTED):
        for column, key in enumerate(("candidate_risk_coverage", "group_risk_coverage")):
            axis = axes[row_index, column]
            runs = grouped[condition]
            coverage = np.asarray(runs[0][key]["coverage_grid"])
            for field, colour, style, label in (
                ("risk_by_width", COLOURS[row_index], "-", "Width"),
                ("risk_by_random", "#7b7b7b", "--", "Random"),
                ("risk_by_oracle", "#222222", ":", "Oracle"),
            ):
                values = np.asarray([run[key][field] for run in runs])
                if values.shape != (3, len(coverage)):
                    raise ValueError(f"incompatible risk curves for {condition}")
                mean = values.mean(axis=0)
                axis.plot(coverage, mean, style, color=colour, linewidth=1.5, label=label)
                if field == "risk_by_width":
                    sd = values.std(axis=0, ddof=1)
                    axis.fill_between(coverage, mean - sd, mean + sd,
                                      color=colour, alpha=0.15, linewidth=0)
            axis.set(title=f"{condition} — {('candidate' if column == 0 else 'group')}",
                     ylabel="Selective risk", xlim=(0.1, 1.0))
            axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Coverage")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"qid_risk_coverage.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path,
                        default=Path("runs/tier2_pythia/uncertainty_analysis"))
    args = parser.parse_args()
    r1r2 = _read(args.analysis / "r1_r2.json")
    trajectories = _read(args.analysis / "trajectories.json")
    r3 = _read(args.analysis / "r3.json")
    build_tables(r1r2, trajectories, r3, args.analysis)
    plot_shape_scale(trajectories, args.analysis)
    plot_risk_coverage(r3, args.analysis)
    print(json.dumps({"status": "complete", "output": str(args.analysis),
                      "primary_conditions": 27, "figures": 2}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
