from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import holm_correction
from .utils import write_json

TABLE_METRICS = (
    "ndcg_at_5",
    "tie_aware_ndcg_at_5",
    "random_ndcg_at_5",
    "ndcg_lift_over_random",
    "spearman",
    "kendall",
    "top1",
    "fractional_top1",
    "separation_ratio",
    "score_mean",
    "score_std",
    "score_range",
    "tie_rate",
    "high_saturation",
    "low_saturation",
    "parsing_coverage",
    "complete_query_coverage",
    "rankable_query_coverage",
)

RUN_LEVEL_METRICS = {
    "parsing_coverage",
    "complete_query_coverage",
    "rankable_query_coverage",
}


def _arr_evaluations(roots: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        root_path = Path(root)
        for path in root_path.rglob("metrics.json"):
            if "pilots" in path.relative_to(root_path).parts:
                continue
            try:
                metrics = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            manifest_path = path.parent / "run_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("pipeline") != "arr" or manifest.get("status") != "complete":
                continue
            if "aggregate" not in metrics or "per_query" not in metrics:
                continue
            rows.append({"path": str(path), "manifest": manifest, "metrics": metrics})
    return rows


def _hierarchical_ci(
    runs: Sequence[dict[str, Any]], metric: str, samples: int, seed: int
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    per_seed = []
    for run in runs:
        values = np.asarray(
            [row.get(metric, float("nan")) for row in run["metrics"]["per_query"]], dtype=float
        )
        values = values[np.isfinite(values)]
        if values.size:
            per_seed.append(values)
    if not per_seed:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    estimate = float(np.mean([values.mean() for values in per_seed]))
    draws = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected_seed_indices = generator.integers(0, len(per_seed), size=len(per_seed))
        seed_means = []
        for seed_index in selected_seed_indices:
            values = per_seed[int(seed_index)]
            seed_means.append(float(generator.choice(values, size=values.size, replace=True).mean()))
        draws[sample] = np.mean(seed_means)
    return {
        "estimate": estimate,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def _run_level_ci(
    runs: Sequence[dict[str, Any]], metric: str, samples: int, seed: int
) -> dict[str, float]:
    """Bootstrap metrics stored once per run rather than once per query."""

    values = np.asarray(
        [run["metrics"]["aggregate"].get(metric, float("nan")) for run in runs],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def _condition(manifest: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(manifest.get("dataset", "unknown")),
        str(manifest.get("model", manifest.get("base_model", "unknown"))),
        str(manifest.get("loss", "prompted")),
        str(manifest.get("prompt_mode", "scalar")),
    )


def _run_seed_map(runs: Sequence[dict[str, Any]], metric: str) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for run in runs:
        seed = int(run["manifest"].get("seed", 0))
        output[seed] = {
            str(row["group_id"]): float(row.get(metric, float("nan")))
            for row in run["metrics"]["per_query"]
            if math.isfinite(float(row.get(metric, float("nan"))))
        }
    return output


def _hierarchical_paired_test(
    left_runs: Sequence[dict[str, Any]],
    right_runs: Sequence[dict[str, Any]],
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float] | None:
    left = _run_seed_map(left_runs, metric)
    right = _run_seed_map(right_runs, metric)
    common_seeds = sorted(set(left) & set(right))
    differences: dict[int, np.ndarray] = {}
    for run_seed in common_seeds:
        group_ids = sorted(set(left[run_seed]) & set(right[run_seed]))
        if group_ids:
            differences[run_seed] = np.asarray(
                [left[run_seed][group_id] - right[run_seed][group_id] for group_id in group_ids],
                dtype=float,
            )
    if not differences:
        return None
    valid_seeds = sorted(differences)
    estimate = float(np.mean([differences[value].mean() for value in valid_seeds]))
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected_seeds = generator.choice(valid_seeds, size=len(valid_seeds), replace=True)
        means = []
        for selected_seed in selected_seeds:
            values = differences[int(selected_seed)]
            means.append(float(generator.choice(values, size=values.size, replace=True).mean()))
        draws[sample] = np.mean(means)
    p_value = min(1.0, 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0))))
    return {
        "difference": estimate,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "p_value": p_value,
        "seed_count": float(len(valid_seeds)),
        "minimum_paired_queries": float(min(len(values) for values in differences.values())),
    }


def _ppo_summaries(roots: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        root_path = Path(root)
        for manifest_path in root_path.rglob("run_manifest.json"):
            if "pilots" in manifest_path.relative_to(root_path).parts:
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                manifest.get("pipeline") != "arr"
                or manifest.get("task") != "train-ppo-generator"
                or manifest.get("status") != "complete"
            ):
                continue
            directory = manifest_path.parent
            baseline_path = directory / "baseline_evaluation.json"
            if not baseline_path.exists():
                continue
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_score = float(baseline.get("bertscore_f1", float("nan")))
            points = [(0, baseline_score)]
            for evaluation_path in directory.glob("checkpoints/update_*/evaluation.json"):
                try:
                    update = int(evaluation_path.parent.name.split("_")[-1])
                    value = float(json.loads(evaluation_path.read_text(encoding="utf-8"))["bertscore_f1"])
                except (ValueError, KeyError, OSError, json.JSONDecodeError):
                    continue
                points.append((update, value))
            points = sorted((update, value) for update, value in points if math.isfinite(value))
            if len(points) < 2 or points[-1][0] <= 0 or not math.isfinite(baseline_score):
                continue
            updates = np.asarray([point[0] for point in points], dtype=float)
            improvement = np.asarray([point[1] - baseline_score for point in points], dtype=float)
            integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            normalised_auc = float(integrate(improvement, updates) / updates[-1])
            reward_manifest_path = Path(str(manifest["reward_checkpoint"])) / "arr_model_manifest.json"
            reward_manifest = (
                json.loads(reward_manifest_path.read_text(encoding="utf-8"))
                if reward_manifest_path.exists()
                else {}
            )
            rows.append(
                {
                    "seed": int(manifest.get("seed", 0)),
                    "reward_model": str(reward_manifest.get("base_model", manifest["reward_checkpoint"])),
                    "reward_loss": str(reward_manifest.get("loss", "unknown")),
                    "baseline_bertscore": baseline_score,
                    "final_bertscore": points[-1][1],
                    "final_improvement": points[-1][1] - baseline_score,
                    "normalised_improvement_auc": normalised_auc,
                    "updates": int(points[-1][0]),
                    "run_dir": str(directory),
                }
            )
    return rows


def aggregate_runs(
    roots: Sequence[str | Path],
    output_dir: str | Path,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    evaluations = _arr_evaluations(roots)
    if not evaluations:
        raise ValueError("no completed ARR metrics.json with an ARR run_manifest.json was found")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in evaluations:
        grouped[_condition(run["manifest"])].append(run)
    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for condition, runs in sorted(grouped.items()):
        key = " | ".join(condition)
        metric_values = {
            metric: (
                _run_level_ci(runs, metric, bootstrap_samples, seed)
                if metric in RUN_LEVEL_METRICS
                else _hierarchical_ci(runs, metric, bootstrap_samples, seed)
            )
            for metric in TABLE_METRICS
        }
        row: dict[str, Any] = {
            "dataset": condition[0],
            "model": condition[1],
            "loss": condition[2],
            "prompt_mode": condition[3],
            "seeds": len(runs),
        }
        for metric, values in metric_values.items():
            row[metric] = values["estimate"]
            row[f"{metric}_low"] = values["low"]
            row[f"{metric}_high"] = values["high"]
        rows.append(row)
        detailed[key] = {
            "runs": [run["path"] for run in runs],
            "metrics": metric_values,
        }
    paired_comparisons: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    conditions = sorted(grouped)
    for left_index, left_condition in enumerate(conditions):
        for right_condition in conditions[left_index + 1 :]:
            if left_condition[0] != right_condition[0]:
                continue
            for metric in (
                "tie_aware_ndcg_at_5",
                "spearman",
                "kendall",
                "separation_ratio",
            ):
                comparison = _hierarchical_paired_test(
                    grouped[left_condition],
                    grouped[right_condition],
                    metric,
                    bootstrap_samples,
                    seed,
                )
                if comparison is None:
                    continue
                name = f"{metric} | {' | '.join(left_condition)} || {' | '.join(right_condition)}"
                paired_comparisons[name] = comparison
                p_values[name] = comparison["p_value"]
    corrected = holm_correction(p_values)
    for name, values in corrected.items():
        paired_comparisons[name]["holm"] = values
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "arr_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "paired_comparisons.json", paired_comparisons)
    latex_lines = [
        "\\begin{tabular}{llllrrrr}",
        "Dataset & Model & Loss & Prompt & Tie-NDCG@5 & Spearman & Kendall & Separation \\\\",
        "\\hline",
    ]
    for row in rows:
        latex_lines.append(
            f"{row['dataset']} & {row['model']} & {row['loss']} & {row['prompt_mode']} & "
            f"{row['tie_aware_ndcg_at_5']:.3f} & {row['spearman']:.3f} & {row['kendall']:.3f} & "
            f"{row['separation_ratio']:.3f} \\\\"
        )
    latex_lines.append("\\end{tabular}")
    (output / "arr_results.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    ppo_rows = _ppo_summaries(roots)
    if ppo_rows:
        separation_lookup: dict[tuple[str, str, int], float] = {}
        for run in evaluations:
            manifest = run["manifest"]
            if str(manifest.get("dataset")) != "e-SNLI":
                continue
            separation_lookup[
                (
                    str(manifest.get("model", manifest.get("base_model", "unknown"))),
                    str(manifest.get("loss", "unknown")),
                    int(manifest.get("seed", 0)),
                )
            ] = float(run["metrics"]["aggregate"]["separation_ratio"])
        for row in ppo_rows:
            row["initial_separation_ratio"] = separation_lookup.get(
                (row["reward_model"], row["reward_loss"], row["seed"]), float("nan")
            )
        with (output / "ppo_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ppo_rows[0]))
            writer.writeheader()
            writer.writerows(ppo_rows)

    figure_status = "created"
    try:
        import matplotlib.pyplot as pyplot

        figure, axis = pyplot.subplots(figsize=(8, max(3, 0.35 * len(rows))))
        labels = [f"{row['model']} / {row['loss']} / {row['dataset']}" for row in rows]
        values = np.asarray([row["separation_ratio"] for row in rows])
        lower = values - np.asarray([row["separation_ratio_low"] for row in rows])
        upper = np.asarray([row["separation_ratio_high"] for row in rows]) - values
        positions = np.arange(len(rows))
        axis.errorbar(values, positions, xerr=np.vstack([lower, upper]), fmt="o")
        axis.axvline(0.8, linestyle="--", color="grey", label="ACL26 hypothesis (0.8)")
        axis.set_yticks(positions, labels)
        axis.set_xlabel("Macro separation ratio with 95% hierarchical bootstrap CI")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "separation_ratio.pdf")
        pyplot.close(figure)
        plot_rows = [
            row for row in ppo_rows if math.isfinite(float(row.get("initial_separation_ratio", float("nan"))))
        ]
        if plot_rows:
            figure, axis = pyplot.subplots(figsize=(6, 5))
            axis.scatter(
                [row["initial_separation_ratio"] for row in plot_rows],
                [row["normalised_improvement_auc"] for row in plot_rows],
            )
            axis.set_xlabel("Initial reward-model separation ratio")
            axis.set_ylabel("Normalised BERTScore improvement AUC")
            figure.tight_layout()
            figure.savefig(output / "separation_vs_ppo_auc.pdf")
            pyplot.close(figure)
    except ImportError:
        figure_status = "matplotlib_missing"
    result = {
        "pipeline": "arr",
        "task": "aggregate",
        "run_count": len(evaluations),
        "condition_count": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "legacy_results_used": False,
        "paired_comparison_count": len(paired_comparisons),
        "ppo_run_count": len(ppo_rows),
        "figure_status": figure_status,
        "conditions": detailed,
        "table_csv": str(csv_path),
    }
    write_json(output / "aggregate_manifest.json", result)
    return result
