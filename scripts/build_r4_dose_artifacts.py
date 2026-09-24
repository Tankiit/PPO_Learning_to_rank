"""Build CSV and figures from the machine-readable R4 dose analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt


COLORS = {
    0.0: "#4c78a8", 0.25: "#4c78a8", 0.5: "#f58518", 1.0: "#54a24b"
}
LABELS = {0.0: "0% exposure", 0.5: "50% exposure", 1.0: "100% exposure"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _response_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for response in result["responses"]:
        row = {key: value for key, value in response.items()
               if key not in {"endpoint", "spearman_by_seed", "nonincreasing_by_seed",
                              "mean_at_0_50_100", "mean_at_25_50_100"}}
        row.update({f"endpoint_{key}": value for key, value in response["endpoint"].items()})
        row["spearman_by_seed"] = json.dumps(response["spearman_by_seed"])
        row["nonincreasing_by_seed"] = json.dumps(response["nonincreasing_by_seed"])
        values = response.get("mean_at_0_50_100", response.get("mean_at_25_50_100"))
        for label, value in zip(("low", "mid", "high"), values):
            row[f"mean_{label}"] = value
        rows.append(row)
    return rows


def _plot_axis(
    result: dict[str, Any], axis: str, output: Path, *, formats: Sequence[str]
) -> None:
    x = [0, 0.5, 1] if axis == "exposure" else [0.25, 0.5, 1]
    x_label = "GPT-4 exposure fraction" if axis == "exposure" else "Training QID fraction"
    fixed_key = "qid_fraction" if axis == "exposure" else "exposure_fraction"
    fixed_values = (0.25, 0.5, 1.0) if axis == "exposure" else (0.0, 0.5, 1.0)
    value_key = "mean_at_0_50_100" if axis == "exposure" else "mean_at_25_50_100"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex="col")
    for row, construction in enumerate(("shared", "independent")):
        for column, metric in enumerate(("probability_width", "js_div_log_c")):
            panel = axes[row, column]
            for fixed in fixed_values:
                match = [item for item in result["responses"]
                         if item["axis"] == axis
                         and item["construction"] == construction
                         and item["metric"] == metric
                         and abs(item[fixed_key] - fixed) < 1e-12]
                if len(match) != 1:
                    raise ValueError(f"missing response: {axis}/{construction}/{metric}/{fixed}")
                label = (f"{round(fixed * 100)}% train QIDs" if axis == "exposure"
                         else LABELS[fixed])
                panel.plot(x, match[0][value_key], marker="o", linewidth=2,
                           color=COLORS[fixed], label=label)
            panel.grid(alpha=0.25)
            panel.set_title(f"{construction.capitalize()} — " +
                            ("probability width" if metric == "probability_width" else "JS / log(C)"))
            if row == 1:
                panel.set_xlabel(x_label)
            if column == 0:
                panel.set_ylabel("Mean disagreement")
            panel.legend(frameon=False, fontsize=8)
    fig.suptitle("R4: disagreement response on the fixed GPT-4 evaluation")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def build(input_path: Path, output: Path) -> None:
    result = json.loads(input_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or len(result.get("runs", [])) != 54:
        raise ValueError("R4 analysis is incomplete")
    _write_csv(output / "r4_cell_summary.csv", result["cell_summaries"])
    _write_csv(output / "r4_seed_runs.csv", result["runs"])
    _write_csv(output / "r4_response_summary.csv", _response_rows(result))
    _plot_axis(result, "exposure", output / "r4_exposure_response", formats=("png", "pdf"))
    _plot_axis(result, "training_qid_fraction", output / "r4_data_dose_response",
               formats=("png", "pdf"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("runs/r4_exposure/dose_analysis/results.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/r4_exposure/dose_analysis"))
    args = parser.parse_args()
    build(args.input, args.output)
    print(f"Built R4 tables and figures in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
