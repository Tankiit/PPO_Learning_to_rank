"""Tier 1 paper figures: e-SNLI ranking quality and abstention behaviour.

Two figures, both drawn from ``runs/tier1/tier1_results.json`` and the saved
predictions it was computed from.

``fig4_esnli_quality_vs_association``
    Figure 4 redrawn on e-SNLI quality. The previous version put in-domain
    NDCG on the quality axis, which is an average over 52 rankable groups of
    two to four candidates and sits a hair above what random ranking scores on
    the same groups. e-SNLI is 1,000 groups of five, so it is the axis that can
    carry the comparison. The random-ranking reference line is drawn because
    an NDCG near 0.8 means nothing without it.

``fig_risk_coverage``
    Whether abstaining on the widest candidates actually lowers error. Oracle
    and random orderings bound what any ordering can do on the same errors, so
    the gap between them is the only space a width ranking can win in.

Colours follow the validated three-slot categorical palette (blue/orange/aqua,
all-pairs CVD and normal-vision checks passing on a light surface). Family is
carried by colour *and* by a direct label on every mark, so identity never
rests on colour alone.
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
from src.arr.tier1 import _risk_curve, prepare_listwise_arrays
from src.arr.utils import read_jsonl

# Validated categorical slots 1-3 (light surface) plus recessive ink.
FAMILY_COLOUR = {
    "independent": "#2a78d6",
    "shared": "#eb6834",
    "mc_dropout": "#1baf7a",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d8d7d2"
REFERENCE = "#8a8983"

LABELS = {
    "independent_listnet": "Independent",
    "independent_listnet_bootstrap": "Independent + bootstrap",
    "independent_mse": "Independent MSE",
    "independent_mse_bootstrap": "Independent MSE + bootstrap",
    "shared_baseline": "Shared baseline",
    "shared_bootstrap": "Shared + bootstrap",
    "shared_features": "Shared + masks",
    "shared_bootstrap_features": "Shared + boot/masks",
    "shared_lambda_0.01": "Shared $\\lambda$=0.01",
    "shared_lambda_0.1": "Shared $\\lambda$=0.1",
    "shared_lambda_1.0": "Shared $\\lambda$=1.0",
    "mc_dropout_k8": "MC-dropout K=8",
}


def _family(name: str) -> str:
    if name.startswith("independent"):
        return "independent"
    if name.startswith("mc_dropout"):
        return "mc_dropout"
    return "shared"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _place_labels(
    ax, fig, points: list[tuple[float, float, str, str]], reserved=()
) -> None:
    """Direct-label every mark, choosing offsets that do not collide.

    Matplotlib will happily stack annotations on top of each other. The palette
    check for these figures ends in a contrast WARN, whose relief is exactly
    these labels, so an unreadable label is not a cosmetic issue: it is the
    accessibility fallback failing. Candidate offsets are tried in order and the
    first one that clears the already-placed boxes and stays inside the axes is
    kept.
    """

    candidates = (
        (7, 4, "left", "bottom"),
        (7, -11, "left", "top"),
        (-7, 4, "right", "bottom"),
        (-7, -11, "right", "top"),
        (0, 11, "center", "bottom"),
        (0, -15, "center", "top"),
        (7, 13, "left", "bottom"),
        (-7, 13, "right", "bottom"),
        (7, -20, "left", "top"),
        (-7, -20, "right", "top"),
    )
    renderer = fig.canvas.get_renderer()
    # Anything already drawn as text (an axis reference label, say) has to be
    # treated as occupied space, or the placer will happily draw straight over it.
    placed: list[Any] = [
        item.get_window_extent(renderer) if hasattr(item, "get_window_extent") else item
        for item in reserved
    ]
    axes_box = ax.get_window_extent(renderer)
    for x, y, text, colour in points:
        best = None
        for dx, dy, ha, va in candidates:
            annotation = ax.annotate(
                text, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=6.5, color=colour, ha=ha, va=va,
            )
            fig.canvas.draw()
            box = annotation.get_window_extent(renderer)
            clash = any(box.overlaps(other) for other in placed)
            inside = axes_box.containsx(box.x0) and axes_box.containsx(box.x1)
            if not clash and inside:
                best = (annotation, box)
                break
            annotation.remove()
        if best is None:
            dx, dy, ha, va = candidates[0]
            annotation = ax.annotate(
                text, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=6.5, color=colour, ha=ha, va=va,
            )
            fig.canvas.draw()
            best = (annotation, annotation.get_window_extent(renderer))
        placed.append(best[1])


def figure_quality_versus_association(results: dict, output: Path) -> None:
    """Ranking quality on e-SNLI against the OOD width/error association."""

    arms = [name for name in LABELS if name in results and "esnli" in results[name]]
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    random_reference = np.mean(
        [
            results[name]["esnli"]["ndcg"]["non_singleton"]["random_ndcg_at_5"]
            for name in arms
        ]
    )
    ax.axvline(random_reference, color=REFERENCE, linestyle="--", linewidth=1.0)
    reference_label = ax.annotate(
        f"random ranking ({random_reference:.3f})",
        xy=(random_reference, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(5, -8),
        textcoords="offset points",
        rotation=90,
        ha="left",
        va="top",
        fontsize=7,
        color=INK_SECONDARY,
    )
    ax.axhline(0.0, color=REFERENCE, linewidth=0.8)

    seen: set[str] = set()
    pending_labels: list[tuple[float, float, str, str]] = []
    for name in arms:
        summary = results[name]["esnli"]
        partial = summary["partial_spearman_given_confidence"]
        x = summary["ndcg"]["non_singleton"]["tie_aware_ndcg_at_5"]
        y = partial["bootstrap"]["estimate"]
        low, high = partial["bootstrap"]["low"], partial["bootstrap"]["high"]
        p_between = partial["permutation_null_between_groups"]["two_sided_p_value"]
        p_within = partial["permutation_null_within_groups"]["two_sided_p_value"]
        # A filled mark means the association survives BOTH nulls. A hollow one
        # means a bootstrap interval that excludes zero is all the evidence
        # there is, which the permutation tests show is not enough.
        survives = bool(p_between < 0.05 and p_within < 0.05)
        family = _family(name)
        colour = FAMILY_COLOUR[family]
        ax.errorbar(
            x, y, yerr=[[y - low], [high - y]],
            fmt="none", ecolor=colour, elinewidth=1.4, capsize=2.5, capthick=1.0,
            alpha=0.9, zorder=2,
        )
        ax.scatter(
            x, y, s=46, zorder=3,
            facecolor=colour if survives else "white",
            edgecolor=colour, linewidth=1.4,
            label=family.replace("_", "-") if family not in seen else None,
        )
        seen.add(family)
        pending_labels.append((x, y, LABELS[name], INK_SECONDARY))

    # Headroom so an edge label is never clipped by the axes.
    x_low, x_high = ax.get_xlim()
    ax.set_xlim(x_low - 0.06 * (x_high - x_low), x_high + 0.10 * (x_high - x_low))
    y_low, y_high = ax.get_ylim()
    ax.set_ylim(y_low - 0.06 * (y_high - y_low), y_high + 0.08 * (y_high - y_low))
    fig.canvas.draw()
    _place_labels(ax, fig, pending_labels, reserved=(reference_label,))

    ax.set_xlabel("e-SNLI tie-aware NDCG@5 (1,000 groups of five candidates)")
    ax.set_ylabel("Partial Spearman $\\rho$\n(width vs error | confidence)")
    ax.set_title(
        "Ranking quality and uncertainty association on e-SNLI",
        fontsize=9, loc="left", pad=10,
    )
    handles, legend_labels = ax.get_legend_handles_labels()
    filled = plt.Line2D(
        [], [], marker="o", linestyle="none", markersize=6,
        markerfacecolor=INK_SECONDARY, markeredgecolor=INK_SECONDARY,
    )
    hollow = plt.Line2D(
        [], [], marker="o", linestyle="none", markersize=6,
        markerfacecolor="white", markeredgecolor=INK_SECONDARY, markeredgewidth=1.4,
    )
    ax.legend(
        handles + [filled, hollow],
        legend_labels + ["survives both nulls", "not distinguishable from null"],
        frameon=False, fontsize=7, loc="lower left", ncol=2,
    )
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output.with_suffix('.pdf')}")


def figure_risk_coverage(
    volume: Path, esnli_groups, arms: dict[str, Path], output: Path
) -> None:
    """Selective risk against coverage, each arm against its own bounds.

    Small multiples rather than shared axes, because the oracle and random
    bounds are properties of *that arm's* error vector. Drawing one arm's
    random baseline behind another arm's curve invites a comparison that is not
    defined: at full coverage every ordering returns that arm's own mean risk,
    and those means differ between arms. One panel per arm keeps each curve next
    to the only bounds it can legitimately be read against.
    """

    names = list(arms)
    fig, axes = plt.subplots(
        2, len(names), figsize=(6.9, 5.0), sharex=True, squeeze=False
    )
    generator = np.random.default_rng(42)

    from src.arr.metrics import evaluate_predictions

    for column, name in enumerate(names):
        records = [ScoreRecord.from_dict(row) for row in read_jsonl(arms[name])]
        data = prepare_listwise_arrays(esnli_groups, records)
        per_query = evaluate_predictions(esnli_groups, records)["per_query"]
        scored = {row["group_id"]: row["tie_aware_ndcg_at_5"] for row in per_query}
        positions = [
            index
            for index, gid in enumerate(data.group_ids)
            if gid in scored and np.isfinite(scored[gid])
        ]
        levels = {
            "candidate": (data.absolute_error, data.width),
            "group": (
                np.asarray(
                    [1.0 - scored[data.group_ids[i]] for i in positions], dtype=float
                ),
                np.asarray(
                    [float(np.mean(data.width[data.group_index == i])) for i in positions],
                    dtype=float,
                ),
            ),
        }
        for row, (level, (risk, order_key)) in enumerate(levels.items()):
            axis = axes[row][column]
            axis.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
            axis.set_axisbelow(True)
            count = risk.size
            coverage = np.arange(1, count + 1) / count
            random_curve = np.zeros(count)
            for _ in range(200):
                random_curve += _risk_curve(risk, generator.permutation(count))
            axis.plot(
                coverage, _risk_curve(risk, np.argsort(risk, kind="stable")),
                linewidth=1.2, color=REFERENCE, zorder=2, label="oracle ordering",
            )
            axis.plot(
                coverage, random_curve / 200,
                linewidth=1.2, color=REFERENCE, linestyle=":", zorder=2,
                label="random ordering",
            )
            axis.plot(
                coverage, _risk_curve(risk, np.argsort(order_key, kind="stable")),
                linewidth=2.0, color=FAMILY_COLOUR[_family(name)], zorder=3,
                label="width ordering",
            )
            axis.set_xlim(0.05, 1.005)
            if row == 0:
                axis.set_title(LABELS[name], fontsize=8, loc="left")
            if row == 1:
                axis.set_xlabel("Coverage (fraction retained)")
            if column == 0:
                axis.set_ylabel(
                    "Candidate risk: mean |error|" if level == "candidate"
                    else "Group risk: 1 - NDCG@5",
                    fontsize=7.5,
                )
            # Placed in the group panel, whose lower-right corner is the one
            # reliably empty region across both arms.
            if row == 1 and column == 0:
                axis.legend(frameon=False, fontsize=6.5, loc="lower right")

    fig.suptitle(
        "Abstention on e-SNLI: does dropping the widest predictions help?",
        fontsize=9, x=0.01, ha="left",
    )
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output.with_suffix('.pdf')}")


def figure_singleton_contamination(results: dict, output: Path) -> None:
    """What the 141 singleton in-domain groups were contributing.

    A singleton group has one candidate, so its within-group softmax is 1.0 by
    construction: width and error are both exactly zero. 141 such rows out of
    270 sit at the origin of the width/error scatter and pull the correlation
    up on their own. Removing them is not a robustness check, it is the
    difference between a reported association and no association at all.
    """

    arms = [
        name
        for name in LABELS
        if name in results
        and "in_domain" in results[name]
        and "in_domain_non_singleton" in results[name]
    ]
    fig, ax = plt.subplots(figsize=(6.9, 3.6))
    ax.grid(True, axis="x", color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.axvline(0.0, color=INK_SECONDARY, linewidth=0.9, zorder=1)

    positions = np.arange(len(arms))
    for row, name in zip(positions, arms):
        contaminated = results[name]["in_domain"][
            "partial_spearman_given_confidence"
        ]["bootstrap"]
        conditioned = results[name]["in_domain_non_singleton"][
            "partial_spearman_given_confidence"
        ]["bootstrap"]
        ax.plot(
            [contaminated["estimate"], conditioned["estimate"]], [row, row],
            color=GRID, linewidth=1.6, zorder=2, solid_capstyle="round",
        )
        ax.errorbar(
            conditioned["estimate"], row,
            xerr=[
                [conditioned["estimate"] - conditioned["low"]],
                [conditioned["high"] - conditioned["estimate"]],
            ],
            fmt="none", ecolor=FAMILY_COLOUR["independent"],
            elinewidth=1.3, capsize=2.5, capthick=1.0, zorder=3,
        )
        ax.scatter(
            contaminated["estimate"], row, s=44, zorder=4,
            facecolor="white", edgecolor=REFERENCE, linewidth=1.4,
            label="all 197 groups (141 singletons)" if row == 0 else None,
        )
        ax.scatter(
            conditioned["estimate"], row, s=44, zorder=5,
            color=FAMILY_COLOUR["independent"],
            label="56 non-singleton groups" if row == 0 else None,
        )

    ax.set_yticks(positions, [LABELS[name] for name in arms], fontsize=7)
    # Reserve a clear band below the last row so the legend never covers a mark.
    ax.set_ylim(len(arms) + 0.35, -0.7)
    ax.set_xlabel("In-domain partial Spearman $\\rho$ (width vs error | confidence)")
    ax.set_title(
        "Singleton groups account for the in-domain uncertainty association",
        fontsize=9, loc="left", pad=10,
    )
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# e-SNLI confound figure (added after the reversal diagnosis)
# ---------------------------------------------------------------------------

TIER_ORDER = ("gold", "good", "fair", "poor", "nonsense")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def figure_esnli_confound(reversal: dict, output: Path, highlight: list[str]) -> None:
    """Why the e-SNLI reversal is a provenance artefact, not a broken reference.

    Left: mean predicted within-group rank against the reference quality tier.
    A model that agreed with the reference would run diagonally down; a model
    with an inverted reference would run diagonally up through every tier.
    Neither happens - gold is dropped to the bottom and the four template tiers
    sit together in the middle.

    Right: the per-group correlation with and without the gold candidate, for
    every arm. Points that move to or across zero are arms whose entire
    reversal is the rejection of the one human-written candidate.
    """

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    arms = reversal["arms"]
    left.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    left.set_axisbelow(True)
    positions = np.arange(len(TIER_ORDER))
    for index, name in enumerate(highlight):
        if name not in arms:
            continue
        per_tier = arms[name]["per_tier"]
        values = [per_tier[t]["mean_predicted_rank"] for t in TIER_ORDER]
        left.plot(
            positions, values, marker="o", markersize=5, linewidth=2.0,
            color=SERIES[index % len(SERIES)], label=LABELS.get(name, name), zorder=3,
        )
    left.plot(
        positions, [1, 2, 3, 4, 5], linewidth=1.2, linestyle="--", color=REFERENCE,
        zorder=2, label="agreement with reference",
    )
    left.set_xticks(positions, TIER_ORDER, fontsize=7)
    left.set_ylim(5.4, 0.6)
    left.set_ylabel("Mean predicted rank in group\n(1 = ranked best)")
    left.set_xlabel("Reference quality tier (best to worst)")
    left.set_title("Gold is rejected; the template tiers are not ordered",
                   fontsize=8, loc="left")
    left.legend(frameon=False, fontsize=6.5, loc="lower left")

    right.grid(True, axis="x", color=GRID, linewidth=0.5, alpha=0.7)
    right.set_axisbelow(True)
    right.axvline(0.0, color=INK_SECONDARY, linewidth=0.9, zorder=1)
    names = [n for n in LABELS if n in arms]
    for row, name in enumerate(names):
        exclusion = arms[name]["gold_exclusion"]
        full = exclusion["spearman_all_candidates"]
        residual = exclusion["spearman_excluding_gold"]
        right.plot([full, residual], [row, row], color=GRID, linewidth=1.6, zorder=2)
        right.scatter(full, row, s=40, facecolor="white", edgecolor=REFERENCE,
                      linewidth=1.4, zorder=3,
                      label="all five candidates" if row == 0 else None)
        right.scatter(residual, row, s=40, color=SERIES[0], zorder=4,
                      label="gold removed" if row == 0 else None)
    right.set_yticks(range(len(names)), [LABELS[n] for n in names], fontsize=6.5)
    right.set_ylim(len(names) + 0.4, -0.7)
    right.set_xlabel("Per-group Spearman vs e-SNLI reference")
    right.set_title("Removing one human-written candidate", fontsize=8, loc="left")
    right.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2)

    fig.suptitle(
        "e-SNLI confounds explanation quality with authorship",
        fontsize=9.5, x=0.01, ha="left",
    )
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output.with_suffix('.pdf')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("runs/tier1/tier1_results.json"))
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("arr_figures"))
    parser.add_argument(
        "--reversal", type=Path, default=Path("runs/tier1/reversal_diagnosis.json")
    )
    args = parser.parse_args(argv)

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.results.read_text())
    results = payload["arms"]

    figure_quality_versus_association(
        results, args.out / "fig4_esnli_quality_vs_association"
    )
    figure_singleton_contamination(
        results, args.out / "fig_singleton_contamination"
    )
    if args.reversal.exists():
        figure_esnli_confound(
            json.loads(args.reversal.read_text()),
            args.out / "fig_esnli_confound",
            highlight=[
                "independent_listnet",
                "independent_mse",
                "shared_baseline",
                "shared_lambda_0.1",
            ],
        )

    esnli_groups = load_groups(
        args.volume / "independent_backbones_50ep_v1/data/esnli_test.jsonl"
    )
    shared = args.volume / "shared_ablation_50ep_v1/listnet"
    independent = args.volume / "independent_backbones_50ep_v1/runs"
    figure_risk_coverage(
        args.volume,
        esnli_groups,
        {
            "independent_listnet": independent
            / "independent/ensemble/evaluation_esnli/predictions.jsonl",
            "shared_baseline": shared / "baseline/evaluation_esnli/predictions.jsonl",
        },
        args.out / "fig_risk_coverage_esnli",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
