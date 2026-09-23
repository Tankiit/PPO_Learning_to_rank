"""Score a completed Pythia Tier-2 checkpoint on a second, fixed evaluation set.

The training CLI evaluates one validation file per epoch. R4 additionally needs
the same checkpoints scored on the GPT-4-only QID holdout. This command loads
saved weights offline and does inference only; it never retrains or modifies the
source run. Five independent member outputs can then be combined loss-safely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .data import load_groups
from .epistemic import center_listwise_member_logits, combine_independent_member_records
from .metrics import evaluate_predictions
from .schema import ScoreRecord
from .tier2_training import (
    LISTWISE_LOSSES,
    PYTHIA_REPO_ID,
    _build_model,
    _load_model_state,
    _prediction_records,
    validate_config,
)
from .utils import canonical_json, read_jsonl, write_json, write_jsonl


def _records(path: Path) -> list[ScoreRecord]:
    return [ScoreRecord.from_dict(item) for item in read_jsonl(path)]


def score_checkpoint(run: Path, data: Path, output: Path) -> dict:
    final = json.loads((run / "_final.json").read_text(encoding="utf-8"))
    if (
        final.get("status") != "complete"
        or final.get("epochs_completed") != final.get("epochs_requested")
        or int(final.get("epochs_completed", 0)) < 1
    ):
        raise ValueError(f"training run is not a complete checkpoint: {run}")
    config = validate_config(json.loads((run / "resolved_config.json").read_text(encoding="utf-8")))
    if (config["global_seed"], config["loss"], config["arm"]) != (
        final["global_seed"], final["loss"], final["arm"]
    ):
        raise ValueError("training config and completion manifest disagree")
    groups = load_groups(data)
    if not groups or len({group.group_id for group in groups}) != len(groups):
        raise ValueError("empty or duplicate evaluation groups")
    if output.exists() and any(output.iterdir()):
        cached_file = output / "_final.json"
        if cached_file.is_file():
            cached = json.loads(cached_file.read_text(encoding="utf-8"))
            if (
                cached.get("status") == "complete"
                and cached.get("source_config_fingerprint") == final["config_fingerprint"]
                and cached.get("evaluation_fingerprint") == groups[0].data_fingerprint
                and (output / str(cached.get("prediction_file"))).is_file()
            ):
                return cached
        raise FileExistsError(f"refusing to overwrite incompatible scored output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("H100 CUDA device is required to score a checkpoint")
    model, tokenizer, training_seed = _build_model(config)
    _load_model_state(model, run / "last" / "model.safetensors")
    model.to(torch.device("cuda"))
    records = _prediction_records(model, tokenizer, groups, config=config,
                                  training_seed=training_seed)
    if len(records) != sum(len(group.candidates) for group in groups):
        raise AssertionError("scorer emitted an incomplete candidate set")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", records)
    result = {
        "status": "complete", "source_run": str(run), "source_config_fingerprint": final["config_fingerprint"],
        "construction": config["construction"], "loss": config["loss"],
        "arm": config["arm"], "global_seed": config["global_seed"],
        "member_id": config["member_id"], "training_seed": training_seed,
        "epochs_completed": final["epochs_completed"],
        "evaluation_data": str(data), "evaluation_fingerprint": groups[0].data_fingerprint,
        "evaluation_groups": len(groups), "evaluation_candidates": len(records),
        "prediction_file": "predictions.jsonl",
        "evaluation": evaluate_predictions(groups, records)["aggregate"],
    }
    write_json(output / "_final.json", result)
    return result


def combine_scored_members(runs: Sequence[Path], data: Path, output: Path) -> dict:
    if len(runs) != 5:
        raise ValueError("an independent ensemble requires five scored members")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite ensemble output: {output}")
    finals = [json.loads((run / "_final.json").read_text(encoding="utf-8")) for run in runs]
    if any(item.get("status") != "complete" or item.get("construction") != "independent"
           for item in finals):
        raise ValueError("all source members must be complete independent scorings")
    common = {(item["global_seed"], item["loss"], item["arm"], item["evaluation_fingerprint"])
              for item in finals}
    if len(common) != 1 or sorted(item["member_id"] for item in finals) != list(range(5)):
        raise ValueError("member seeds, objectives, arms, or eval data do not match")
    groups = load_groups(data)
    if groups[0].data_fingerprint != finals[0]["evaluation_fingerprint"]:
        raise ValueError("evaluation data differs from member scoring manifests")
    ordered = sorted(zip(finals, runs), key=lambda item: item[0]["member_id"])
    source = [_records(path / item["prediction_file"]) for item, path in ordered]
    loss = finals[0]["loss"]
    aligned = [center_listwise_member_logits(records) if loss in LISTWISE_LOSSES
               else records for records in source]
    combined = combine_independent_member_records(
        aligned, model_name=f"{PYTHIA_REPO_ID}::{finals[0]['arm']}",
        model_revision="five-independent-ranking-reward-models",
    )
    original = [{(r.group_id, r.candidate_id): r for r in member} for member in source]
    enriched = []
    for record in combined:
        key = (record.group_id, record.candidate_id)
        members = [member[key] for member in original]
        raw = [float(item.metadata["raw_score"]) for item in members]
        probability = [float(item.metadata["group_softmax_score"]) for item in members]
        enriched.append(ScoreRecord(**{
            **record.to_dict(),
            "raw_output": canonical_json({
                "raw": raw, "sigmoid_aligned": record.metadata["head_scores"],
                "group_softmax": probability,
            }),
            "metadata": {
                **record.metadata, "raw_head_scores": raw,
                "group_softmax_scores": probability,
                "source_member_scores": [float(item.score) for item in members],
            },
        }))
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", enriched)
    result = {
        "status": "complete", "construction": "independent_ensemble",
        "loss": loss, "arm": finals[0]["arm"], "global_seed": finals[0]["global_seed"],
        "member_seeds": [item["training_seed"] for item, _ in ordered],
        "evaluation_fingerprint": groups[0].data_fingerprint,
        "evaluation_groups": len(groups), "evaluation_candidates": len(enriched),
        "prediction_file": "predictions.jsonl",
        "evaluation": evaluate_predictions(groups, enriched)["aggregate"],
        "listwise_alignment": (
            "within_group_mean_zero_logit" if loss in LISTWISE_LOSSES else "none"
        ),
    }
    write_json(output / "_final.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score-checkpoint")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("combine-independent")
    combine.add_argument("--runs", nargs=5, type=Path, required=True)
    combine.add_argument("--data", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "score-checkpoint":
        result = score_checkpoint(args.run, args.data, args.output)
    else:
        result = combine_scored_members(args.runs, args.data, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
