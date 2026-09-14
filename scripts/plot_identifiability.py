"""The identifiability spine: three width read-outs of one checkpoint.

A listwise model determines scores only up to an additive constant per group,
per member. Width can be read in three places and they are not interchangeable:

  raw logit width   spread of pre-sigmoid member scores - carries the arbitrary
                    offset directly.
  sigmoid width     spread of the bounded scores, which is what ``credal_width``
                    stores today. Squashing compresses the offset non-linearly;
                    it does not remove it.
  quotient width    spread of within-group softmax probabilities - width in the
                    quotient by the gauge group, and the only read-out that is a
                    function of the identified object.

On the canonical representative all three track each other, which is exactly why
the distinction is easy to miss and worth drawing: the pipeline fixes the gauge
at combine time, so the numbers agree *by construction of that step*, not
because the statistics are equivalent.

The right panel shows what the agreement is worth. The same checkpoints are
re-expressed in randomly chosen representatives - each an equally valid solution
of the same model - and the two unidentified read-outs fan into bands while the
quotient read-out stays on a single line. The band width is the size of the
claim that a raw or sigmoid width can make on its own: none.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arr.data import load_groups
from src.arr.schema import ScoreRecord
from src.arr.tier1 import apply_gauge_shift, width_readouts
from src.arr.utils import read_jsonl, write_json

READOUTS = (
    ("raw_logit_width", "Raw logit width", "#2a78d6"),
    ("sigmoid_width", "Sigmoid width (credal_width)", "#eb6834"),
    ("quotient_width", "Quotient width (identified)", "#1baf7a"),
)
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument(
        "--ensemble",
        default="independent_backbones_50ep_v1/runs/independent/ensemble",
    )
    parser.add_argument("--dev", type=Path, default=Path("data/arr/ds_critique_external_dev.jsonl"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gauges", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("arr_figures"))
    args = parser.parse_args(argv)

    groups = [g for g in load_groups(args.dev) if len(g.candidates) >= 2]
    wanted = {g.group_id for g in groups}
    base = args.volume / args.ensemble

    canonical, bands = [], []
    for epoch in range(args.epochs):
        records = [
            ScoreRecord.from_dict(r)
            for r in read_jsonl(base / f"validation_predictions_epoch_{epoch}.jsonl")
        ]
        records = [r for r in records if r.group_id in wanted]
        canonical.append({"epoch": epoch, **width_readouts(groups, records)})
        per_gauge = []
        for seed in range(args.gauges):
            shifted = apply_gauge_shift(
                records, magnitude=1.5, per_member=True, seed=seed
            )
            per_gauge.append(width_readouts(groups, shifted))
        bands.append(
            {
                key: {
                    "low": float(np.min([g[key] for g in per_gauge])),
                    "high": float(np.max([g[key] for g in per_gauge])),
                    "median": float(np.median([g[key] for g in per_gauge])),
                }
                for key, _, _ in READOUTS
            }
        )

    plt.rcParams.update(
        {
            "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
            "axes.edgecolor": INK_SECONDARY, "text.color": INK,
            "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
            "axes.spines.top": False, "axes.spines.right": False,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    epochs = np.arange(1, args.epochs + 1)

    for axis in (left, right):
        axis.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
        axis.set_axisbelow(True)
        axis.set_yscale("log")
        axis.set_xlabel("Training epoch")

    for key, label, colour in READOUTS:
        left.plot(
            epochs, [row[key] for row in canonical],
            linewidth=2.0, color=colour, label=label, zorder=3,
        )
        low = np.array([b[key]["low"] for b in bands])
        high = np.array([b[key]["high"] for b in bands])
        right.fill_between(epochs, low, high, color=colour, alpha=0.30, linewidth=0, zorder=2)
        right.plot(epochs, [b[key]["median"] for b in bands],
                   linewidth=1.6, color=colour, zorder=3)

    left.set_ylabel("Mean width (log scale)")
    left.set_title("Canonical representative", fontsize=8, loc="left")
    right.set_title(
        f"{args.gauges} arbitrary representatives of the same checkpoints",
        fontsize=8, loc="left",
    )
    left.legend(frameon=False, fontsize=6.5, loc="upper right")

    # Quantify the fan-out in the caption rather than leaving it to the eye.
    spread = {
        key: float(
            np.median(
                [
                    (b[key]["high"] - b[key]["low"]) / max(b[key]["median"], 1e-12)
                    for b in bands
                ]
            )
        )
        for key, _, _ in READOUTS
    }
    fig.suptitle(
        "Only the quotient read-out is a function of the identified object",
        fontsize=9.5, x=0.01, ha="left",
    )
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "fig_identifiability"
    for suffix in (".pdf", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=320, bbox_inches="tight")
    plt.close(fig)

    write_json(
        Path("runs/tier1/identifiability.json"),
        {"canonical": canonical, "gauge_bands": bands, "median_relative_spread": spread},
    )
    print(f"wrote {output.with_suffix('.pdf')}")
    print("median relative spread across arbitrary representatives:")
    for key, label, _ in READOUTS:
        print(f"   {label:32s} {spread[key]:8.2%}")
    first, last = canonical[0], canonical[-1]
    print("\ncanonical trajectory, epoch 1 -> 50:")
    for key, label, _ in READOUTS:
        print(
            f"   {label:32s} {first[key]:.5f} -> {last[key]:.5f} "
            f"({(last[key] - first[key]) / first[key]:+.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
