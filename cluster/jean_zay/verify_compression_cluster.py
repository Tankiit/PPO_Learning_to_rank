from __future__ import annotations

import importlib
import gc
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    versions: dict[str, str] = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "scipy",
        "yaml",
        "safetensors",
    ):
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "present"))

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one CUDA GPU, found {torch.cuda.device_count()}")
    device = torch.cuda.get_device_name(0)
    if "H100" not in device.upper():
        raise RuntimeError(f"expected an H100, found {device}")

    from src.arr.utils import resolve_hf_source
    from transformers import AutoConfig, AutoTokenizer

    specifications = {
        "mistral": (
            "mistralai/Mistral-7B-Instruct-v0.3",
            "c170c708c41dac9275d15a8fff4eca08d52bab71",
        ),
        "llama": (
            "meta-llama/Llama-3.1-8B-Instruct",
            "0e9e39f249a16976918f6564b8830bc894c89659",
        ),
    }
    resolved: dict[str, str] = {}
    for name, (repository, revision) in specifications.items():
        source = resolve_hf_source(repository, revision, local_files_only=True)
        AutoConfig.from_pretrained(source, local_files_only=True)
        AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
        resolved[name] = source

    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(
        64,
        32,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).cuda()
    result = layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
    if result.shape != (2, 32) or not torch.isfinite(result).all():
        raise RuntimeError("NF4/bfloat16 smoke test failed")
    del layer, result
    gc.collect()
    torch.cuda.empty_cache()

    # Exercise the real ARR path, not only the CUDA kernel: one QLoRA update,
    # PEFT checkpoint reload, then scoring on a different domain.
    from src.arr.data import load_groups
    from src.arr.judges import ScalarJudge
    from src.arr.training import train_judge

    root = Path(os.environ.get("ARR_PROJECT_ROOT", Path.cwd()))
    esnli = load_groups(root / "data" / "arr" / "esnli_train.jsonl")
    validation = load_groups(root / "data" / "arr" / "esnli_val.jsonl")
    external = load_groups(root / "data" / "arr" / "ds_critique_external_test.jsonl")
    with tempfile.TemporaryDirectory(prefix="arr-compression-preflight-") as temporary:
        smoke_dir = Path(temporary) / "mistral-listnet"
        smoke_config = {
            "base_model": specifications["mistral"][0],
            "revision": specifications["mistral"][1],
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
        }
        smoke_manifest = train_judge(esnli[:2], validation[:1], smoke_config, smoke_dir)
        del smoke_manifest
        gc.collect()
        torch.cuda.empty_cache()
        records = ScalarJudge(
            smoke_dir / "best",
            seed=42,
            batch_size=2,
            max_length=64,
            local_files_only=True,
        ).score(external[:1])
        if len(records) != len(external[0].candidates):
            raise RuntimeError("cross-domain checkpoint reload returned an invalid record count")
        if not all(record.score is not None and 0.0 <= record.score <= 1.0 for record in records):
            raise RuntimeError("cross-domain checkpoint reload returned an invalid scalar score")

    report = {
        "status": "passed",
        "python": sys.version,
        "device": device,
        "capability": torch.cuda.get_device_capability(0),
        "torch_cuda": torch.version.cuda,
        "versions": versions,
        "resolved_models": resolved,
        "offline": {
            key: os.environ.get(key)
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        },
    }
    destination = root / "arr_runs" / "compression" / "slurm" / "preflight_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
