"""Across-seed, across-split analysis of the Tier 2 matrix.

Applies the Tier 1 statistics to every (split, seed, loss, arm) combination that
exists on the Modal volume, and reports the spread across seeds. Until this
runs, every number in the report rests on a single seed.

Two splits are handled. ``published`` is the original pairing, whose held-out
set is 52 rankable groups of two to four candidates against a random-ranking
NDCG@5 baseline of 0.926 - too little headroom to separate arms. ``qid`` is the
question-disjoint partition, 54 groups of twelve against a baseline of 0.669.

Missing combinations are reported, never silently skipped, and every downloaded
file is verified to parse before use: a truncated download is otherwise
indistinguishable from a short file.

    python scripts/seed_replicate_analysis.py --mirror <dir> --split qid \
        --seeds 42,123,777
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arr.data import load_groups
from src.arr.schema import ScoreRecord
from src.arr.tier1 import (
    group_bootstrap_partial_spearman,
    js_divergence_to_consensus,
    ndcg_by_group_size,
    participation_ratio_conventions,
    permutation_null_partial_spearman,
    prepare_listwise_arrays,
    risk_coverage_curve,
)
from src.arr.utils import read_jsonl, write_json

VOLUME = "ppo-ltr-epistemic-runs"
ESNLI_REMOTE = "independent_backbones_50ep_v1/data/esnli_test.jsonl"
DEV_FILES = {
    "published": Path("data/arr/ds_critique_external_dev.jsonl"),
    "qid": Path("data/arr/ds_critique_qidsplit_dev.jsonl"),
}
INDEPENDENT_ARMS = ("independent", "independent_bootstrap")
SHARED_ARMS = (
    "baseline", "bootstrap", "features", "bootstrap_features",
    "lambda_0p01", "lambda_0p1", "lambda_1",
)
INK, INK_SECONDARY, GRID, REFERENCE = "#0b0b0b", "#52514e", "#d8d7d2", "#8a8983"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def independent_root(loss: str, seed: int, split: str) -> str:
    stem = "independent_backbones" if loss == "listnet" else "independent_backbones_mse"
    if split == "qid":
        return f"{stem}_qid_50ep_seed{seed}"
    return f"{stem}_50ep_v1" if seed == 42 else f"{stem}_50ep_seed{seed}"


def shared_root(seed: int, split: str) -> str:
    if split == "qid":
        return f"shared_ablation_qid_50ep_seed{seed}"
    return "shared_ablation_50ep_v1" if seed == 42 else f"shared_ablation_50ep_seed{seed}"


def _parses(path: Path) -> bool:
    try:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
    except Exception:
        return False
    return True


def fetch(mirror: Path, remote: str) -> Path | None:
    local = mirror / remote
    if local.exists() and local.stat().st_size > 0 and _parses(local):
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["modal", "volume", "get", VOLUME, remote, str(local), "--force"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not local.exists():
        return None
    if not _parses(local):
        local.unlink(missing_ok=True)
        return None
    return local


def analyse(groups, records, *, bootstrap: int, permutations: int) -> dict:
    arrays = prepare_listwise_arrays(groups, records)
    residuals = arrays.member_probabilities - arrays.target_probabilities[:, None]
    return {
        "ndcg": ndcg_by_group_size(groups, records),
        "participation_ratio": participation_ratio_conventions(residuals),
        "diversity": js_divergence_to_consensus(arrays),
        "partial_spearman_given_confidence": {
            "bootstrap": group_bootstrap_partial_spearman(arrays, samples=bootstrap),
            "permutation_null_between_groups": permutation_null_partial_spearman(
                arrays, permutations=permutations, scheme="between_groups"
            ),
            "permutation_null_within_groups": permutation_null_partial_spearman(
                arrays, permutations=permutations, scheme="within_groups"
            ),
        },
        "risk_coverage_candidate": risk_coverage_curve(arrays),
    }


def targets(split: str, seed: int, losses: list[str]) -> dict[str, dict[str, str]]:
    """Map arm label -> {portion: remote path} for one (split, seed)."""

    out: dict[str, dict[str, str]] = {}
    for loss in losses:
        root = independent_root(loss, seed, split)
        for arm in INDEPENDENT_ARMS:
            label = f"independent_{loss}" + ("_bootstrap" if arm.endswith("bootstrap") else "")
            base = f"{root}/runs/{arm}/ensemble"
            out[label] = {
                "in_domain": f"{base}/validation_predictions_epoch_49.jsonl",
                "esnli": f"{base}/evaluation_esnli/predictions.jsonl",
            }
        shared = shared_root(seed, split)
        for arm in SHARED_ARMS:
            out[f"shared_{loss}_{arm}"] = {
                "in_domain": f"{shared}/{loss}/{arm}/validation_predictions_epoch_49.jsonl",
                "esnli": f"{shared}/{loss}/{arm}/evaluation_esnli/predictions.jsonl",
            }
    return out


def forest_plot(results: dict, seeds: list[int], split: str, output: Path) -> None:
    """One panel per statistic, one row per arm, one mark per global seed."""

    panels = [
        ("Held-out NDCG@5",
         lambda r: r["in_domain"]["ndcg"]["non_singleton"]["tie_aware_ndcg_at_5"]),
        ("NDCG lift over random",
         lambda r: r["in_domain"]["ndcg"]["lift_over_random_ci"]["estimate"]),
        ("Participation fraction",
         lambda r: r["in_domain"]["participation_ratio"]["participation_fraction"]),
        ("Normalised AURC gain",
         lambda r: r["in_domain"]["risk_coverage_candidate"]["normalised_aurc_gain"]),
    ]
    arms = sorted({a for s in results.values() for a in s})
    if not arms:
        print("nothing to plot")
        return
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
        "axes.edgecolor": INK_SECONDARY, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    height = max(2.6, 0.28 * len(arms) + 1.2)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, height))
    for axis, (title, getter) in zip(axes, panels):
        axis.grid(True, axis="x", color=GRID, linewidth=0.5, alpha=0.7)
        axis.set_axisbelow(True)
        for row, arm in enumerate(arms):
            values = []
            for seed in seeds:
                entry = results.get(str(seed), {}).get(arm)
                if entry and "in_domain" in entry:
                    try:
                        values.append(getter(entry))
                    except (KeyError, TypeError):
                        pass
            if not values:
                continue
            colour = SERIES[0] if arm.startswith("independent") else SERIES[1]
            jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else [0.0]
            axis.scatter(values, [row + j for j in jitter], s=20,
                         facecolor="white", edgecolor=colour, linewidth=1.1, zorder=3)
            mean = float(np.mean(values))
            axis.plot([mean, mean], [row - 0.28, row + 0.28],
                      color=colour, linewidth=2.0, zorder=4)
            if len(values) > 1:
                axis.plot([min(values), max(values)], [row, row],
                          color=colour, linewidth=0.9, alpha=0.6, zorder=2)
        axis.set_yticks(range(len(arms)))
        axis.set_yticklabels(
            [a.replace("_", " ") for a in arms] if axis is axes[0] else [""] * len(arms),
            fontsize=6,
        )
        axis.set_ylim(len(arms) - 0.5, -0.5)
        axis.set_title(title, fontsize=7, loc="left")
        if "lift" in title or "AURC" in title:
            low, high = axis.get_xlim()
            # Only draw the zero reference when it is near the data; forcing it
            # into view otherwise compresses every point into one stripe.
            span = high - low
            if low - 0.25 * span <= 0.0 <= high + 0.25 * span:
                axis.axvline(0.0, color=REFERENCE, linewidth=0.9, zorder=1)
                axis.set_xlim(min(low, -0.02 * span), max(high, 0.02 * span))
    fig.suptitle(
        f"Tier 2 replicates, {split} split, held-out evaluation "
        f"(seeds {', '.join(map(str, seeds))}); bar = mean, circles = seeds",
        fontsize=8.5, x=0.01, ha="left",
    )
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output.with_suffix('.pdf')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(DEV_FILES), default="qid")
    parser.add_argument("--seeds", default="42,123,777")
    parser.add_argument("--losses", default="listnet,mse")
    parser.add_argument("--output", type=Path, default=Path("runs/tier2"))
    parser.add_argument("--figure-dir", type=Path, default=Path("arr_figures"))
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args(argv)

    seeds = [int(p) for p in args.seeds.replace(",", " ").split()]
    losses = [p for p in args.losses.replace(",", " ").split()]
    # Singleton groups have a one-candidate softmax, so their width and error
    # are identically zero and they dominate every uncertainty statistic
    # computed over the published split. They are excluded here for the same
    # reason the Tier 1 recomputation excludes them; on the qid split, where
    # every group has twelve candidates, this filter is a no-op.
    all_groups = load_groups(DEV_FILES[args.split])
    dev_groups = [g for g in all_groups if len(g.candidates) >= 2]
    dropped = len(all_groups) - len(dev_groups)
    esnli_path = fetch(args.mirror, ESNLI_REMOTE)
    esnli_groups = load_groups(esnli_path) if esnli_path else None
    print(
        f"split={args.split}  held-out: {len(dev_groups)} groups / "
        f"{sum(len(g.candidates) for g in dev_groups)} candidates"
        + (f"  ({dropped} singleton groups excluded)" if dropped else "")
        + "\n"
    )

    results: dict[str, dict] = {}
    missing: list[str] = []
    for seed in seeds:
        results[str(seed)] = {}
        for arm, portions in targets(args.split, seed, losses).items():
            entry: dict = {}
            for portion, remote in portions.items():
                groups = dev_groups if portion == "in_domain" else esnli_groups
                if groups is None:
                    continue
                path = fetch(args.mirror, remote)
                if path is None:
                    missing.append(f"seed {seed} / {arm} / {portion}")
                    continue
                records = [ScoreRecord.from_dict(r) for r in read_jsonl(path)]
                if portion == "in_domain":
                    wanted = {g.group_id for g in groups}
                    records = [r for r in records if r.group_id in wanted]
                entry[portion] = analyse(
                    groups, records,
                    bootstrap=args.bootstrap_samples, permutations=args.permutations,
                )
            if entry:
                results[str(seed)][arm] = entry
                if "in_domain" in entry:
                    n = entry["in_domain"]["ndcg"]["non_singleton"]
                    lift = entry["in_domain"]["ndcg"]["lift_over_random_ci"]
                    print(
                        f"  seed {seed:3d} {arm:34s} "
                        f"NDCG={n['tie_aware_ndcg_at_5']:.4f} "
                        f"(rand {n['random_ndcg_at_5']:.4f}) "
                        f"lift={lift['estimate']:+.3f} "
                        f"[{lift['low']:+.3f},{lift['high']:+.3f}] n={n['evaluated_query_count']}"
                    )

    if missing:
        print(f"\nnot yet available ({len(missing)} combinations):")
        for item in missing[:20]:
            print(f"  {item}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    available = [s for s in seeds if results.get(str(s))]
    if not available:
        print("\nno replicates available yet")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split, "seeds": available, "losses": losses,
        "missing": missing, "arms": results,
        "held_out_groups": len(dev_groups),
        "held_out_candidates": sum(len(g.candidates) for g in dev_groups),
    }
    write_json(args.output / f"replicates_{args.split}.json", payload)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    forest_plot(results, available, args.split, args.figure_dir / f"fig_seed_forest_{args.split}")
    print(f"\nwrote {args.output / f'replicates_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
