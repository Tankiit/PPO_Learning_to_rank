"""Rebuild independent-ensemble predictions locally from finished members.

Training for the independent replicates completed, but the combine stage that
turns five member prediction files into one ensemble file did not run for most
arms. That stage is pure CPU arithmetic over saved JSONL, so it does not need
Modal and it does not need a GPU: this script does it in the local mirror,
using the same two functions the Modal job would have called, in the same
order.

``center_listwise_member_logits`` first, because independently trained ListNet
models fix the additive logit gauge arbitrarily and their raw levels are not
comparable; then ``combine_independent_member_records``, which aligns members by
identifier rather than file order so a member score can never attach to the
wrong candidate.

Where Modal did produce an ensemble file, this script recomputes it anyway and
compares, so the local reconstruction is checked against the remote one rather
than trusted.

    python scripts/salvage_ensembles.py --mirror <dir> --seeds 47,52,57
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arr.epistemic import (
    center_listwise_member_logits,
    combine_independent_member_records,
)
from src.arr.schema import ScoreRecord
from src.arr.utils import read_jsonl, write_jsonl

VOLUME = "ppo-ltr-epistemic-runs"
MODEL = "EleutherAI/pythia-70m"
ARMS = ("independent", "independent_bootstrap")
MEMBERS = 5


def _parses(path: Path) -> bool:
    try:
        for line in path.read_text().splitlines():
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
    if result.returncode != 0 or not local.exists() or not _parses(local):
        local.unlink(missing_ok=True)
        return None
    return local


def root(seed: int, loss: str = "listnet") -> str:
    stem = "independent_backbones" if loss == "listnet" else "independent_backbones_mse"
    return f"{stem}_50ep_v1" if seed == 42 else f"{stem}_50ep_seed{seed}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--seeds", default="47,52,57")
    parser.add_argument("--loss", default="listnet")
    parser.add_argument("--epoch", type=int, default=49)
    args = parser.parse_args(argv)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
    built = skipped = verified = 0
    for seed in seeds:
        for arm in ARMS:
            base = f"{root(seed, args.loss)}/runs/{arm}"
            target = args.mirror / base / "ensemble" / (
                f"validation_predictions_epoch_{args.epoch}.jsonl"
            )
            members = []
            complete = True
            for member in range(MEMBERS):
                remote = (
                    f"{base}/member_{member}_seed{seed + member}"
                    f"/validation_predictions_epoch_{args.epoch}.jsonl"
                )
                path = fetch(args.mirror, remote)
                if path is None:
                    print(f"  seed {seed} {arm}: member {member} missing, skipping arm")
                    complete = False
                    break
                records = [ScoreRecord.from_dict(r) for r in read_jsonl(path)]
                # Verify the member file really is the seed it claims to be.
                seeds_seen = {r.seed for r in records}
                if seeds_seen and seeds_seen != {seed + member}:
                    raise ValueError(
                        f"{remote} carries seeds {seeds_seen}, expected {seed + member}"
                    )
                members.append(
                    center_listwise_member_logits(records)
                    if args.loss == "listnet"
                    else records
                )
            if not complete:
                skipped += 1
                continue
            combined = combine_independent_member_records(
                members,
                model_name=f"{MODEL}::{arm}",
                model_revision="five-independent-backbones",
            )

            # If Modal already produced this file, check the reconstruction.
            remote_ensemble = f"{base}/ensemble/validation_predictions_epoch_{args.epoch}.jsonl"
            existing = fetch(args.mirror, remote_ensemble) if not target.exists() else None
            if existing is not None:
                reference = {
                    (r["group_id"], r["candidate_id"]): r["score"]
                    for r in read_jsonl(existing)
                }
                deviation = max(
                    abs(float(r.score) - reference[(r.group_id, r.candidate_id)])
                    for r in combined
                    if (r.group_id, r.candidate_id) in reference
                )
                print(
                    f"  seed {seed} {arm}: matches the Modal-produced ensemble to "
                    f"{deviation:.3e}"
                )
                if deviation > 1e-9:
                    raise ValueError("local reconstruction disagrees with Modal output")
                verified += 1
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(target, combined)
            heads = {len(r.metadata["head_scores"]) for r in combined}
            print(
                f"  seed {seed} {arm}: built {len(combined)} records, "
                f"{heads.pop()} members -> {target.relative_to(args.mirror)}"
            )
            built += 1

    print(f"\nbuilt {built}, verified against Modal {verified}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
