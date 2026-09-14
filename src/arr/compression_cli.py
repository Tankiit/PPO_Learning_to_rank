"""Compression-only command line interface.

This entry point contains only the data, prompted-judge, scalar-judge and
aggregation stages used by the score-compression diagnostic.  It deliberately
does not import the PPO or human-evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from .aggregate import aggregate_runs
from .config import load_config, save_resolved_config
from .data import (
    build_ds_critique_groups,
    build_esnli_groups,
    load_groups,
    stratified_group_sample,
    write_ds_critique_dataset,
    write_esnli_dataset,
)
from .epistemic import EpistemicScalarJudge, evaluate_epistemic_predictions
from .judges import PromptedJudge, ScalarJudge
from .metrics import bootstrap_query_ci, evaluate_predictions
from .schema import ScoreRecord
from .training import train_judge
from .utils import read_jsonl, write_json, write_jsonl


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a dotted YAML key; repeat as needed",
    )


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, args.overrides)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {name!r} must be a mapping")
    return dict(value)


def _prepare_data(args: argparse.Namespace) -> int:
    config = _config(args)
    data_config = _section(config, "data")
    output = args.output_dir or Path(data_config.get("output_dir", "data/arr"))
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output)
    if args.dataset in {"esnli", "all"}:
        groups = build_esnli_groups(
            source=args.source or data_config.get("esnli_source"),
            train_groups=int(data_config.get("esnli_train_groups", 10_000)),
            limit_per_split=args.limit_per_split,
        )
        write_esnli_dataset(groups, output)
    if args.dataset in {"ds-main", "all"}:
        source = args.ds_main_source or data_config.get(
            "ds_main_source", "data/ds_critique_bank/DSCB-train-crowd-anno.jsonl"
        )
        groups = build_ds_critique_groups(
            source,
            split="external_test",
            expected_groups=None if args.skip_size_check else 270,
            expected_candidates=None if args.skip_size_check else 3240,
        )
        write_ds_critique_dataset(groups, output, "external_test")
    if args.dataset in {"ds-dev", "all"}:
        source = args.ds_dev_source or data_config.get(
            "ds_dev_source", "data/ds_critique_bank/DSCB-dev-crowd-anno.jsonl"
        )
        groups = build_ds_critique_groups(
            source,
            split="external_dev",
            expected_candidates=None if args.skip_size_check else 270,
        )
        write_ds_critique_dataset(groups, output, "external_dev")
    return 0


def _score_prompted(args: argparse.Namespace) -> int:
    config = _config(args)
    prompted = _section(config, "prompted")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output)
    groups = load_groups(args.data)
    group_count = args.group_count
    if group_count is None:
        group_count = (
            prompted.get("esnli_group_count")
            if all(group.domain == "nli" for group in groups)
            else len(groups)
        )
    if group_count is not None and int(group_count) < len(groups):
        groups = stratified_group_sample(
            groups, int(group_count), int(args.seed), fields=("label_name", "domain")
        )
    model = args.model or prompted.get("model")
    if not model:
        raise ValueError("--model or prompted.model is required")
    prompt_mode = args.prompt_mode or str(prompted.get("prompt_mode", "direct"))
    manifest = {
        "pipeline": "arr",
        "task": "score-prompted",
        "status": "running",
        "started_at_unix": time.time(),
        "model": model,
        "prompt_mode": prompt_mode,
        "seed": args.seed,
        "dataset": groups[0].metadata.get("dataset", "unknown") if groups else "unknown",
        "data_fingerprint": groups[0].data_fingerprint if groups else None,
        "query_count": len(groups),
    }
    write_json(output / "run_manifest.json", manifest)
    try:
        judge = PromptedJudge(
            model_name=str(model),
            prompt_mode=prompt_mode,
            revision=str(prompted.get("revision", "main")),
            seed=args.seed,
            batch_size=int(prompted.get("batch_size", 4)),
            max_new_tokens=prompted.get("max_new_tokens"),
            max_input_length=int(prompted.get("max_input_length", 2048)),
            cache_path=output / "predictions.jsonl",
            dtype=str(prompted.get("dtype", "bfloat16")),
            local_files_only=bool(prompted.get("local_files_only", False)),
        )
        records = judge.score(groups)
        write_jsonl(output / "predictions.jsonl", records)
        metrics = evaluate_predictions(groups, records)
        write_json(output / "metrics.json", metrics)
        coverage = float(metrics["aggregate"]["parsing_coverage"])
        manifest.update(
            {
                "status": "complete",
                "completed_at_unix": time.time(),
                "model_revision": judge.model_revision,
                "prompt_hash": judge.prompt_hash,
                "parsing_coverage": coverage,
                "pilot_passed": coverage >= float(prompted.get("pilot_min_coverage", 0.99)),
            }
        )
        write_json(output / "run_manifest.json", manifest)
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(output / "run_manifest.json", manifest)
        raise


def _train_judge(args: argparse.Namespace) -> int:
    config = _config(args)
    training = _section(config, "judge_training")
    training.update(
        {
            key: value
            for key, value in {
                "base_model": args.model,
                "architecture": args.architecture,
                "loss": args.loss,
                "seed": args.seed,
                "resume_from": str(args.resume_from) if args.resume_from else None,
            }.items()
            if value is not None
        }
    )
    if "base_model" not in training:
        raise ValueError("--model or judge_training.base_model is required")
    if training.get("architecture") == "encoder":
        training["qlora"] = False
        training.setdefault("learning_rate", 2e-5)
        training.setdefault("patience", 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config["judge_training"] = training
    save_resolved_config(config, args.output_dir)
    train_judge(
        load_groups(args.train_data),
        load_groups(args.validation_data),
        training,
        args.output_dir,
    )
    return 0


def _train_epistemic_judge(args: argparse.Namespace) -> int:
    config = _config(args)
    training = _section(config, "judge_training")
    epistemic = _section(config, "epistemic")
    training.update(
        {
            "base_model": args.model or training.get("base_model"),
            "architecture": "decoder",
            "loss": args.loss or training.get("loss", "mse"),
            "seed": args.seed,
            "device": args.device or training.get("device", "auto"),
            "epochs": args.epochs if args.epochs is not None else training.get("epochs", 3),
            "group_batch_size": (
                args.batch_size
                if args.batch_size is not None
                else training.get("group_batch_size", 2)
            ),
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "epistemic_heads": int(args.head_count or epistemic.get("head_count", 5)),
            "epistemic_hidden_dim": int(epistemic.get("hidden_dim", 256)),
            "epistemic_dropout_min": float(epistemic.get("dropout_min", 0.05)),
            "epistemic_dropout_max": float(epistemic.get("dropout_max", 0.30)),
            "epistemic_lambda_div": float(
                args.lambda_div
                if args.lambda_div is not None
                else epistemic.get("lambda_div", 0.0)
            ),
            "epistemic_mc_dropout": int(
                args.mc_dropout
                if args.mc_dropout is not None
                else epistemic.get("mc_dropout", 0)
            ),
            "epistemic_bootstrap_members": bool(
                args.bootstrap_members or epistemic.get("bootstrap_members", False)
            ),
            "epistemic_feature_keep_fraction": float(
                args.feature_keep_fraction
                if args.feature_keep_fraction is not None
                else epistemic.get("feature_keep_fraction", 1.0)
            ),
            "epistemic_feature_seed": int(
                epistemic.get("feature_seed", args.seed)
            ),
        }
    )
    if not training.get("base_model"):
        raise ValueError("--model or judge_training.base_model is required")
    if int(training["epistemic_heads"]) < 2:
        raise ValueError("epistemic training requires at least two heads")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config["judge_training"] = training
    config["epistemic"] = epistemic
    save_resolved_config(config, args.output_dir)
    train_groups = load_groups(args.train_data)
    validation_groups = load_groups(args.validation_data)
    if args.max_train_groups is not None:
        train_groups = train_groups[: args.max_train_groups]
    if args.max_validation_groups is not None:
        validation_groups = validation_groups[: args.max_validation_groups]
    train_judge(
        train_groups,
        validation_groups,
        training,
        args.output_dir,
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    config = _config(args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output)
    groups = load_groups(args.data)
    if args.predictions:
        records = [ScoreRecord.from_dict(row) for row in read_jsonl(args.predictions)]
        model = args.model_name or "predictions-file"
    else:
        judge = ScalarJudge(
            args.checkpoint,
            seed=args.seed,
            batch_size=int(_section(config, "evaluation").get("batch_size", 16)),
            local_files_only=bool(_section(config, "evaluation").get("local_files_only", False)),
        )
        records = judge.score(groups)
        model = judge.model_name
        write_jsonl(output / "predictions.jsonl", records)
    metrics = evaluate_predictions(groups, records)
    statistics = _section(config, "statistics")
    metrics["confidence_intervals"] = {
        metric: bootstrap_query_ci(
            metrics["per_query"],
            metric,
            samples=int(statistics.get("bootstrap_samples", 10_000)),
            seed=args.seed,
        )
        for metric in (
            "ndcg_at_5",
            "tie_aware_ndcg_at_5",
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
        )
    }
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "run_manifest.json",
        {
            "pipeline": "arr",
            "task": "evaluate",
            "status": "complete",
            "completed_at_unix": time.time(),
            "model": model,
            "model_name": args.model_name,
            "loss": args.loss,
            "prompt_mode": args.prompt_mode,
            "seed": args.seed,
            "dataset": groups[0].metadata.get("dataset", "unknown") if groups else "unknown",
            "split": groups[0].split if groups else "unknown",
            "data_fingerprint": groups[0].data_fingerprint if groups else None,
            "prediction_count": len(records),
        },
    )
    return 0


def _diagnose_epistemic(args: argparse.Namespace) -> int:
    """Gate the epistemic work: is credal width a signal or an artefact?"""

    from .epistemic_diagnostics import checkpoint_label, run_diagnostics

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    groups = load_groups(args.data)
    records = [ScoreRecord.from_dict(row) for row in read_jsonl(args.predictions)]
    per_epoch = {
        checkpoint_label(path): [
            ScoreRecord.from_dict(row) for row in read_jsonl(path)
        ]
        for path in (args.epoch_predictions or ())
    }
    report = run_diagnostics(
        groups, records, per_epoch or None, listwise=bool(args.listwise)
    )
    write_json(output / "epistemic_diagnostics.json", report)
    verdict = report["verdict"]
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["epistemic_signal_supported"] else 2


def _train_ppo_epistemic(args: argparse.Namespace) -> int:
    from .data import load_groups as _load_groups
    from .ppo_epistemic import RewardSpec, train_ppo

    config = _config(args)
    evaluation = _section(config, "evaluation")
    ppo = _section(config, "ppo")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, args.output_dir)

    spec = RewardSpec(
        checkpoint=args.reward_checkpoint,
        penalty=args.penalty,
        lam=float(args.lam if args.lam is not None else ppo.get("lam", 1.0)),
        mc_dropout=int(args.mc_dropout or 0),
        batch_size=int(evaluation.get("batch_size", 16)),
        max_length=int(evaluation.get("max_length", 512)),
        local_files_only=bool(evaluation.get("local_files_only", False)),
    )
    train_ppo(
        spec,
        policy_model=args.policy or ppo.get("policy_model"),
        dataset=_load_groups(args.data),
        output_dir=args.output_dir,
        trl_api=args.trl_api,
        learning_rate=float(ppo.get("learning_rate", 1.41e-5)),
        batch_size=int(ppo.get("batch_size", 64)),
        mini_batch_size=int(ppo.get("mini_batch_size", 16)),
        ppo_epochs=int(ppo.get("ppo_epochs", 4)),
        kl_coef=float(ppo.get("kl_coef", 0.2)),
        max_new_tokens=int(ppo.get("max_new_tokens", 128)),
        seed=args.seed,
        dtype=str(ppo.get("dtype", "float32")),
        total_episodes=(
            int(args.total_episodes)
            if args.total_episodes is not None
            else (int(ppo["total_episodes"]) if "total_episodes" in ppo else None)
        ),
    )
    return 0


def _evaluate_epistemic(args: argparse.Namespace) -> int:
    config = _config(args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output)
    groups = load_groups(args.data)
    evaluation = _section(config, "evaluation")
    judge = EpistemicScalarJudge(
        args.checkpoint,
        seed=args.seed,
        batch_size=int(evaluation.get("batch_size", 16)),
        max_length=int(evaluation.get("max_length", 512)),
        local_files_only=bool(evaluation.get("local_files_only", False)),
        mc_dropout=args.mc_dropout,
    )
    records = judge.score(groups)
    write_jsonl(output / "predictions.jsonl", records)
    metrics = evaluate_predictions(groups, records)
    statistics = _section(config, "statistics")
    metrics["confidence_intervals"] = {
        metric: bootstrap_query_ci(
            metrics["per_query"],
            metric,
            samples=int(statistics.get("bootstrap_samples", 10_000)),
            seed=args.seed,
        )
        for metric in (
            "ndcg_at_5",
            "tie_aware_ndcg_at_5",
            "spearman",
            "kendall",
            "top1",
            "separation_ratio",
            "score_mean",
            "score_std",
            "score_range",
            "tie_rate",
            "high_saturation",
        )
    }
    write_json(output / "metrics.json", metrics)
    epistemic_metrics = evaluate_epistemic_predictions(groups, records)
    write_json(output / "epistemic_metrics.json", epistemic_metrics)
    write_json(
        output / "run_manifest.json",
        {
            "pipeline": "arr",
            "task": "evaluate-epistemic-scalar-judge",
            "status": "complete",
            "completed_at_unix": time.time(),
            "method": "credence_head_disagreement",
            "checkpoint": str(args.checkpoint),
            "model": judge.model_name,
            "model_revision": judge.model_revision,
            "model_name": args.model_name,
            "loss": args.loss,
            "seed": args.seed,
            "head_count": judge.head_count,
            "dataset": groups[0].metadata.get("dataset", "unknown") if groups else "unknown",
            "split": groups[0].split if groups else "unknown",
            "data_fingerprint": groups[0].data_fingerprint if groups else None,
            "prediction_count": len(records),
        },
    )
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    config = _config(args)
    statistics = _section(config, "statistics")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, args.output_dir)
    aggregate_runs(
        args.roots,
        args.output_dir,
        bootstrap_samples=int(statistics.get("bootstrap_samples", 10_000)),
        seed=args.seed,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.arr.compression_cli",
        description="Score-compression diagnostic pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    _add_config_arguments(prepare)
    prepare.add_argument("--dataset", choices=("esnli", "ds-main", "ds-dev", "all"), default="all")
    prepare.add_argument("--source", type=Path)
    prepare.add_argument("--ds-main-source", type=Path)
    prepare.add_argument("--ds-dev-source", type=Path)
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--limit-per-split", type=int)
    prepare.add_argument("--skip-size-check", action="store_true")
    prepare.set_defaults(handler=_prepare_data)

    prompted = subparsers.add_parser("score-prompted")
    _add_config_arguments(prompted)
    prompted.add_argument("--data", type=Path, required=True)
    prompted.add_argument("--model")
    prompted.add_argument("--prompt-mode", choices=("direct", "cot"))
    prompted.add_argument("--output-dir", type=Path, required=True)
    prompted.add_argument("--group-count", type=int)
    prompted.add_argument("--seed", type=int, default=42)
    prompted.set_defaults(handler=_score_prompted)

    judge = subparsers.add_parser("train-judge")
    _add_config_arguments(judge)
    judge.add_argument("--train-data", type=Path, required=True)
    judge.add_argument("--validation-data", type=Path, required=True)
    judge.add_argument("--output-dir", type=Path, required=True)
    judge.add_argument("--model")
    judge.add_argument("--architecture", choices=("decoder", "encoder"))
    judge.add_argument("--loss", choices=("mse", "ranknet", "lambdarank", "listnet"))
    judge.add_argument("--seed", type=int)
    judge.add_argument("--resume-from", type=Path)
    judge.set_defaults(handler=_train_judge)

    epistemic_train = subparsers.add_parser("train-epistemic-judge")
    _add_config_arguments(epistemic_train)
    epistemic_train.add_argument("--train-data", type=Path, required=True)
    epistemic_train.add_argument("--validation-data", type=Path, required=True)
    epistemic_train.add_argument("--output-dir", type=Path, required=True)
    epistemic_train.add_argument("--model")
    epistemic_train.add_argument(
        "--loss", choices=("mse", "ranknet", "lambdarank", "listnet")
    )
    epistemic_train.add_argument("--seed", type=int, default=42)
    epistemic_train.add_argument("--head-count", type=int)
    epistemic_train.add_argument("--resume-from", type=Path)
    epistemic_train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    epistemic_train.add_argument("--epochs", type=int)
    epistemic_train.add_argument("--batch-size", type=int)
    epistemic_train.add_argument(
        "--bootstrap-members",
        action="store_true",
        help="give each member an exact group-level bootstrap data view",
    )
    epistemic_train.add_argument(
        "--feature-keep-fraction",
        type=float,
        help="fixed fraction of shared hidden features visible to each member",
    )
    epistemic_train.add_argument(
        "--max-train-groups", type=int, help="cap groups for a smoke run"
    )
    epistemic_train.add_argument(
        "--max-validation-groups", type=int, help="cap groups for a smoke run"
    )
    epistemic_train.add_argument(
        "--lambda-div",
        dest="lambda_div",
        type=float,
        help="decorrelation weight on head residuals; 0.0 reproduces the "
             "previous objective exactly",
    )
    epistemic_train.add_argument(
        "--mc-dropout",
        dest="mc_dropout",
        type=int,
        help="stochastic passes per head at inference; 0 keeps deterministic scoring",
    )
    epistemic_train.set_defaults(handler=_train_epistemic_judge)

    diagnose = subparsers.add_parser("diagnose-epistemic")
    _add_config_arguments(diagnose)
    diagnose.add_argument("--data", type=Path, required=True)
    diagnose.add_argument("--predictions", type=Path, required=True)
    diagnose.add_argument(
        "--epoch-predictions",
        type=Path,
        nargs="*",
        help="one predictions file per epoch, in order, for the D3 check",
    )
    diagnose.add_argument("--output-dir", type=Path, required=True)
    diagnose.add_argument(
        "--listwise",
        action="store_true",
        help="diagnose ListNet outputs in within-group softmax(logit(score)) space",
    )
    diagnose.set_defaults(handler=_diagnose_epistemic)

    ppo = subparsers.add_parser("train-ppo-epistemic")
    _add_config_arguments(ppo)
    ppo.add_argument("--data", type=Path, required=True)
    ppo.add_argument("--reward-checkpoint", type=Path, required=True)
    ppo.add_argument("--output-dir", type=Path, required=True)
    ppo.add_argument("--policy")
    ppo.add_argument(
        "--penalty", choices=("none", "var", "credal"), default="credal"
    )
    ppo.add_argument("--lam", type=float)
    ppo.add_argument("--mc-dropout", dest="mc_dropout", type=int)
    ppo.add_argument("--trl-api", choices=("modern", "legacy"), default="modern")
    ppo.add_argument(
        "--total-episodes",
        dest="total_episodes",
        type=int,
        help="cap PPO episodes; useful for smoke runs, leave unset for a real run",
    )
    ppo.add_argument("--seed", type=int, default=0)
    ppo.set_defaults(handler=_train_ppo_epistemic)

    evaluate = subparsers.add_parser("evaluate")
    _add_config_arguments(evaluate)
    evaluate.add_argument("--data", type=Path, required=True)
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--predictions", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--model-name")
    evaluate.add_argument("--loss", default="unknown")
    evaluate.add_argument("--prompt-mode", default="scalar")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.set_defaults(handler=_evaluate)

    epistemic_evaluate = subparsers.add_parser("evaluate-epistemic")
    _add_config_arguments(epistemic_evaluate)
    epistemic_evaluate.add_argument("--data", type=Path, required=True)
    epistemic_evaluate.add_argument("--checkpoint", type=Path, required=True)
    epistemic_evaluate.add_argument("--output-dir", type=Path, required=True)
    epistemic_evaluate.add_argument("--model-name")
    epistemic_evaluate.add_argument("--loss", default="unknown")
    epistemic_evaluate.add_argument("--seed", type=int, default=42)
    epistemic_evaluate.add_argument("--mc-dropout", type=int)
    epistemic_evaluate.set_defaults(handler=_evaluate_epistemic)

    aggregate = subparsers.add_parser("aggregate")
    _add_config_arguments(aggregate)
    aggregate.add_argument("--roots", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--seed", type=int, default=42)
    aggregate.set_defaults(handler=_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
