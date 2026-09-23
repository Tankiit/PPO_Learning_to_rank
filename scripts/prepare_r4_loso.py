"""Extract GPT-4 explanations withheld from the 216 training QIDs.

This is the source-only LOSO evaluation. The 54-QID GPT-4 holdout prepared by
``prepare_r4_exposure`` is a separate, simultaneous QID/source shift.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from scripts.prepare_r4_exposure import GPT4, PROVENANCE, gpt4_heldout
from src.arr.data import load_groups
from src.arr.utils import write_json, write_jsonl


def prepare(train: Path, output: Path) -> dict:
    groups = load_groups(train)
    if len(groups) != 216 or len({group.metadata["qid"] for group in groups}) != 216:
        raise ValueError("LOSO requires exactly 216 unique training QIDs")
    if {candidate.score_provenance for group in groups for candidate in group.candidates} != {PROVENANCE}:
        raise ValueError("LOSO requires human crowd targets only")
    rows, fingerprint = gpt4_heldout(groups)
    for row in rows:
        row["split"] = "r4_gpt4_loso_train_qids"
    if any(len(row["candidates"]) != 3 or any(
        candidate["metadata"]["student_model"] != GPT4 for candidate in row["candidates"]
    ) for row in rows):
        raise ValueError("LOSO evaluation contains an unexpected candidate")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"refusing to overwrite LOSO data or manifest: {output}")
    write_jsonl(output, rows)
    info = {
        "protocol": "arr-r4-loso-source-only-v1",
        "evaluation_file": output.name,
        "groups": len(rows),
        "candidates": sum(len(row["candidates"]) for row in rows),
        "student_model": GPT4,
        "target_provenance": PROVENANCE,
        "training_source_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "fingerprint": fingerprint,
        "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "warning": "QIDs occur in training, but these GPT-4 explanations never do.",
    }
    write_json(manifest, info)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/arr/ds_critique_qidsplit_train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/arr/r4_exposure/gpt4_loso_train_qids.jsonl"))
    args = parser.parse_args()
    print(prepare(args.train, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
