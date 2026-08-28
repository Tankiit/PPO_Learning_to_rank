from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path


def main() -> int:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one CUDA GPU, found {torch.cuda.device_count()}")
    device = torch.cuda.get_device_name(0)
    if "H100" not in device.upper():
        raise RuntimeError(f"expected an H100, found {device}")

    from src.arr.data import load_groups
    from src.arr.epistemic import EpistemicScalarJudge, evaluate_epistemic_predictions
    from src.arr.metrics import evaluate_predictions
    from src.arr.training import train_judge

    root = Path(os.environ.get("ARR_PROJECT_ROOT", Path.cwd()))
    train = load_groups(root / "data" / "arr" / "esnli_train.jsonl")
    validation = load_groups(root / "data" / "arr" / "esnli_val.jsonl")
    external = load_groups(root / "data" / "arr" / "ds_critique_external_test.jsonl")
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    revision = "c170c708c41dac9275d15a8fff4eca08d52bab71"
    with tempfile.TemporaryDirectory(prefix="arr-epistemic-preflight-") as temporary:
        run_dir = Path(temporary) / "mistral-listnet"
        config = {
            "base_model": base_model,
            "revision": revision,
            "local_files_only": True,
            "architecture": "decoder",
            "qlora": True,
            "dtype": "bfloat16",
            "epochs": 1,
            "group_batch_size": 2,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_length": 64,
            "loss": "listnet",
            "seed": 42,
            "epistemic_heads": 3,
            "epistemic_hidden_dim": 32,
            "epistemic_dropout_min": 0.05,
            "epistemic_dropout_max": 0.30,
        }
        manifest = train_judge(train[:2], validation[:1], config, run_dir)
        if manifest.get("status") != "complete":
            raise RuntimeError("epistemic training smoke test did not complete")
        del manifest
        gc.collect()
        torch.cuda.empty_cache()

        judge = EpistemicScalarJudge(
            run_dir / "best",
            seed=42,
            batch_size=2,
            max_length=64,
            local_files_only=True,
        )
        records = judge.score(external[:1])
        expected = len(external[0].candidates)
        if len(records) != expected or judge.head_count != 3:
            raise RuntimeError("reloaded epistemic checkpoint returned an invalid shape")
        central = evaluate_predictions(external[:1], records)
        epistemic = evaluate_epistemic_predictions(external[:1], records)
        if central["aggregate"]["parsing_coverage"] != 1.0:
            raise RuntimeError("epistemic central-score coverage is incomplete")
        if epistemic["head_count"] != 3:
            raise RuntimeError("epistemic head metadata was not preserved")
        if not all(
            record.score is not None
            and len(record.metadata.get("head_scores", [])) == 3
            and 0.0 <= float(record.metadata["epistemic_variance"])
            for record in records
        ):
            raise RuntimeError("invalid epistemic candidate record")

    report = {
        "status": "passed",
        "device": device,
        "method": "credence_head_disagreement",
        "smoke_head_count": 3,
        "checkpoint_reload": True,
        "cross_domain_scoring": True,
    }
    destination = root / "arr_runs" / "epistemic" / "slurm" / "preflight_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
