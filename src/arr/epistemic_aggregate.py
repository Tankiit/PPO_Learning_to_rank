from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .utils import write_json


def collect_epistemic_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "evaluations").glob("**/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("task") != "evaluate-epistemic-scalar-judge"
        ):
            continue
        run_dir = manifest_path.parent
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        epistemic = json.loads(
            (run_dir / "epistemic_metrics.json").read_text(encoding="utf-8")
        )
        central = metrics["aggregate"]
        uncertainty = epistemic["aggregate"]
        head_separations = [
            float(head["separation_ratio"])
            for head in epistemic["individual_head_metrics"]
        ]
        relative = run_dir.relative_to(root / "evaluations").parts
        if len(relative) < 4:
            raise ValueError(f"unexpected epistemic run layout: {run_dir}")
        rows.append(
            {
                "model": relative[0],
                "loss": relative[1],
                "seed": manifest["seed"],
                "dataset": relative[-1],
                "head_count": manifest["head_count"],
                "separation_ratio": central["separation_ratio"],
                "separation_ci_low": metrics["confidence_intervals"]["separation_ratio"]["low"],
                "separation_ci_high": metrics["confidence_intervals"]["separation_ratio"]["high"],
                "ndcg_at_5": central["tie_aware_ndcg_at_5"],
                "spearman": central["spearman"],
                "score_mean": central["score_mean"],
                "score_std": central["score_std"],
                "mean_epistemic_variance": uncertainty["mean_epistemic_variance"],
                "mean_credal_width": uncertainty["mean_credal_width"],
                "reference_interval_coverage": uncertainty["reference_interval_coverage"],
                "epistemic_error_spearman": uncertainty[
                    "spearman_epistemic_vs_absolute_error"
                ],
                "pairwise_correct_nonoverlap_rate": uncertainty[
                    "pairwise_correct_nonoverlap_rate"
                ],
                "pairwise_interval_overlap_rate": uncertainty[
                    "pairwise_interval_overlap_rate"
                ],
                "head_separation_min": min(head_separations),
                "head_separation_max": max(head_separations),
                "run_dir": str(run_dir),
            }
        )
    return sorted(rows, key=lambda row: (row["model"], row["loss"], row["dataset"]))


def _markdown(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Epistemic scalar-head results",
        "",
        "Five CREDENCE-style scalar heads share one QLoRA backbone. The reported scalar score is "
        "their mean; epistemic uncertainty is their population variance and the credal interval "
        "is their min–max envelope.",
        "",
        "| Model | Loss | Dataset | Separation ratio | NDCG@5 | Mean epistemic variance | "
        "Mean interval width | ρ(Uepi, abs. error) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['loss']} | {row['dataset']} | "
            f"{row['separation_ratio']:.3f} "
            f"[{row['separation_ci_low']:.3f}, {row['separation_ci_high']:.3f}] | "
            f"{row['ndcg_at_5']:.3f} | {row['mean_epistemic_variance']:.6f} | "
            f"{row['mean_credal_width']:.3f} | {row['epistemic_error_spearman']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation is deferred until every expected run is complete. In particular, "
            "non-zero variance alone is not evidence that uncertainty resolves compression; it "
            "must track errors or distinguish relevant candidate orderings.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(root: Path, output: Path, expected_runs: int | None = 8) -> list[dict[str, Any]]:
    rows = collect_epistemic_runs(root)
    if expected_runs is not None and len(rows) != expected_runs:
        raise RuntimeError(f"expected {expected_runs} complete runs, found {len(rows)}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", rows)
    if rows:
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "RESULTS.md").write_text(_markdown(rows), encoding="utf-8")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate epistemic scalar-head experiments")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=8)
    args = parser.parse_args(argv)
    aggregate(args.root, args.output, args.expected_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
