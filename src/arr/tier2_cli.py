"""Command-line interface for the Pythia Tier-2 training matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from .tier2_training import (
    PYTHIA_REVISION,
    combine_independent_runs,
    model_path_from_environment,
    train_tier2,
    validate_config,
    verify_tier2_run,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping at {path}")
    return value


def _base_config(args: argparse.Namespace) -> dict[str, Any]:
    source = _load_yaml(args.config)
    model = dict(source.get("model") or {})
    data = dict(source.get("data") or {})
    training = dict(source.get("training") or {})
    experiment = dict(source.get("experiment") or {})
    config = {
        "model_name": model.get("repo_id", "EleutherAI/pythia-70m"),
        "model_path": model_path_from_environment(args.model_path or model.get("path")),
        "revision": model.get("revision", PYTHIA_REVISION),
        "train_data": args.train_data or data["train"],
        "validation_data": args.validation_data or data["validation"],
        "global_seed": args.global_seed,
        "loss": args.loss,
        "epochs": args.epochs if args.epochs is not None else training["epochs"],
        "max_length": training["max_length"],
        "group_batch_size": training["group_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training.get("weight_decay", 0.01),
        "warmup_ratio": training.get("warmup_ratio", 0.03),
        "max_grad_norm": training.get("max_grad_norm", 1.0),
        "checkpoint_epochs": training.get("checkpoint_epochs", [1, 10, 25, 50]),
        "head_count": experiment.get("members", 5),
        "dropout": experiment.get("head_dropout", 0.1),
        "pooling": model.get("pooling", "last"),
        "max_train_groups": args.max_train_groups,
        "max_validation_groups": args.max_validation_groups,
    }
    # Keep the historical configuration fingerprint unchanged when this new
    # ablation is not requested. This lets completed 80%-mask runs remain
    # idempotently reusable after adding the fixed-count sweep.
    if args.feature_keep_count is not None:
        config["feature_keep_count"] = args.feature_keep_count
    if args.pilot:
        config.update(
            {
                "epochs": 1,
                "max_train_groups": args.max_train_groups or 8,
                "max_validation_groups": args.max_validation_groups or 4,
                "checkpoint_epochs": [1],
            }
        )
    return config


def _train_member(args: argparse.Namespace) -> int:
    config = {
        **_base_config(args),
        "construction": "independent",
        "arm": args.arm,
        "member_id": args.member_id,
    }
    result = train_tier2(config, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


def _train_shared(args: argparse.Namespace) -> int:
    config = {
        **_base_config(args),
        "construction": "shared",
        "arm": args.arm,
        "member_id": None,
    }
    result = train_tier2(config, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


def _combine(args: argparse.Namespace) -> int:
    result = combine_independent_runs(args.runs, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    config = {
        **_base_config(args),
        "construction": args.construction,
        "arm": args.arm,
        "member_id": args.member_id if args.construction == "independent" else None,
    }
    print(json.dumps(validate_config(config), indent=2, sort_keys=True))
    return 0


def _verify_run(args: argparse.Namespace) -> int:
    result = verify_tier2_run(args.run)
    print(json.dumps(result, sort_keys=True))
    return 0


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=Path("configs/arr/tier2_pythia.yaml")
    )
    parser.add_argument("--model-path")
    parser.add_argument("--train-data")
    parser.add_argument("--validation-data")
    parser.add_argument("--loss", choices=["listnet", "listmle", "mse"], required=True)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-groups", type=int)
    parser.add_argument("--max-validation-groups", type=int)
    parser.add_argument(
        "--feature-keep-count",
        type=int,
        help=(
            "fixed number k of backbone dimensions retained by each head; "
            "valid only for shared features/bootstrap_features arms"
        ),
    )
    parser.add_argument("--pilot", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    member = commands.add_parser("train-member")
    _add_training_arguments(member)
    member.add_argument(
        "--arm", choices=["independent", "independent_bootstrap"], default="independent"
    )
    member.add_argument("--member-id", type=int, required=True)
    member.add_argument("--output", type=Path, required=True)
    member.set_defaults(function=_train_member)

    shared = commands.add_parser("train-shared")
    _add_training_arguments(shared)
    shared.add_argument(
        "--arm",
        choices=[
            "baseline",
            "bootstrap",
            "features",
            "bootstrap_features",
            "lambda_0p01",
            "lambda_0p1",
            "lambda_1",
        ],
        default="baseline",
    )
    shared.add_argument("--output", type=Path, required=True)
    shared.set_defaults(function=_train_shared)

    combine = commands.add_parser("combine-independent")
    combine.add_argument("--runs", nargs=5, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.set_defaults(function=_combine)

    validate = commands.add_parser("validate-config")
    _add_training_arguments(validate)
    validate.add_argument("--construction", choices=["independent", "shared"], required=True)
    validate.add_argument("--arm", required=True)
    validate.add_argument("--member-id", type=int, default=0)
    validate.set_defaults(function=_validate)

    verify = commands.add_parser("verify-run")
    verify.add_argument("--run", type=Path, required=True)
    verify.set_defaults(function=_verify_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
