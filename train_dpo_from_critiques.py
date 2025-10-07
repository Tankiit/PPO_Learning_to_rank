"""
Train a DPO (Direct Preference Optimization) policy from JSONL preference pairs.

Assumptions about the JSONL (one JSON object per line):
    {
        "prompt":   "<text prompt>",
        "chosen":   "<preferred continuation>",
        "rejected": "<dispreferred continuation>",
        // optional, kept as metadata:
        "source_model": {"chosen": "...", "rejected": "..."},
        "scores": {"chosen": float, "rejected": float},
        "qid": "..."
    }

This script:
1) Loads the JSONL into a Hugging Face Dataset.
2) (Optionally) creates a validation split by holding out a ratio from train.
3) Loads a chat LLM (e.g., Llama-2/3, Mistral) with optional 4-bit quantization.
4) Applies QLoRA adapters via PEFT.
5) Trains with TRL's `DPOTrainer`.

Example:
    python train_dpo_from_critiques.py \
        --data dpo_preferences_from_critiques.jsonl \
        --model meta-llama/Llama-2-7b-chat-hf \
        --output dpo-llama2-7b-qlora \
        --epochs 1 --train-batch-size 2 --grad-accum 4 \
        --lr 2e-5 --bf16 true --use-4bit true

Notes:
- Pass `--ref-model` to use a frozen reference model; otherwise DPOTrainer may create
  an internal reference (costly with large models). Keeping it `None` is common with LoRA.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


LOG = logging.getLogger("train_dpo")


# ---------------------------------------------------------------------------
# CLI / Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    data_path: Path
    output_dir: Path

    # Model
    model_name: str
    ref_model_name: Optional[str]
    trust_remote_code: bool
    use_4bit: bool
    bf16: bool
    fp16: bool

    # LoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: Tuple[str, ...]  # target module names

    # Training
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    grad_accum: int
    learning_rate: float
    logging_steps: int
    save_steps: int
    eval_strategy: str
    eval_ratio: float  # proportion of train set to hold out for validation
    seed: int

    # Sequence lengths
    max_length: int
    max_prompt_length: int
    max_target_length: int

    # DPO specifics
    beta: float
    label_smoothing: float


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Train a DPO policy from JSONL preference pairs.")

    # Data / IO
    p.add_argument("--data", type=Path, required=True, help="Path to JSONL with DPO pairs.")
    p.add_argument("--output", type=Path, default=Path("./dpo-llama-qlora"), help="Output directory.")

    # Model
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--ref-model", type=str, default=None, help="Optional frozen reference model.")
    p.add_argument("--trust-remote-code", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--use-4bit", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=False)

    # LoRA
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-targets",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated target module names (e.g., 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj').",
    )

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    p.add_argument("--eval-ratio", type=float, default=0.0, help="0.0 disables holdout; else fraction of train used as validation.")
    p.add_argument("--seed", type=int, default=42)

    # Lengths
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-target-length", type=int, default=512)

    # DPO specifics
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta coefficient.")
    p.add_argument("--label-smoothing", type=float, default=0.0)

    a = p.parse_args()
    return RunConfig(
        data_path=a.data,
        output_dir=a.output,
        model_name=a.model,
        ref_model_name=a.ref_model,
        trust_remote_code=a.trust_remote_code,
        use_4bit=a.use_4bit,
        bf16=a.bf16,
        fp16=a.fp16,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        lora_targets=tuple(t.strip() for t in a.lora_targets.split(",") if t.strip()),
        epochs=a.epochs,
        train_batch_size=a.train_batch_size,
        eval_batch_size=a.eval_batch_size,
        grad_accum=a.grad_accum,
        learning_rate=a.lr,
        logging_steps=a.logging_steps,
        save_steps=a.save_steps,
        eval_strategy=a.eval_strategy,
        eval_ratio=a.eval_ratio,
        seed=a.seed,
        max_length=a.max_length,
        max_prompt_length=a.max_prompt_length,
        max_target_length=a.max_target_length,
        beta=a.beta,
        label_smoothing=a.label_smoothing,
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dpo_dataset(jsonl_path: Path) -> Dataset:
    """Load a JSONL into a Dataset; expect 'prompt','chosen','rejected' fields."""
    ds = load_dataset("json", data_files={"train": str(jsonl_path)}, split="train")
    required = {"prompt", "chosen", "rejected"}
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"Missing required fields in dataset: {missing}")

    # Filter invalid/empty examples
    def _ok(ex: Dict[str, Any]) -> bool:
        return all(isinstance(ex[k], str) and ex[k].strip() for k in ("prompt", "chosen", "rejected"))

    ds = ds.filter(_ok)
    return ds


def maybe_make_val_split(ds: Dataset, eval_ratio: float, seed: int) -> DatasetDict:
    """Optionally create a validation split from the training set."""
    if eval_ratio and 0.0 < eval_ratio < 1.0:
        split = ds.train_test_split(test_size=eval_ratio, seed=seed)
        return DatasetDict(train=split["train"], validation=split["test"])
    return DatasetDict(train=ds)


# ---------------------------------------------------------------------------
# Model / PEFT
# ---------------------------------------------------------------------------

def build_bnb_config(use_4bit: bool, bf16: bool) -> Optional[BitsAndBytesConfig]:
    if not use_4bit:
        return None
    compute = torch.bfloat16 if bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute,
    )


def load_tokenizer(model_name: str, trust_remote_code: bool) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    # Ensure pad token exists for DPO losses
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # safer for causal LMs with batched sequences
    return tok


def load_model_with_lora(
    model_name: str,
    lcfg: LoraConfig,
    bnb_cfg: Optional[BitsAndBytesConfig],
    trust_remote_code: bool,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model = get_peft_model(model, lcfg)
    return model


def load_reference_model(ref_name: Optional[str], bnb_cfg: Optional[BitsAndBytesConfig], trust_remote_code: bool):
    if not ref_name:
        return None
    # Reference model should be *frozen* for DPO; no LoRA adapters applied here.
    ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(cfg: RunConfig) -> None:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    LOG.info("Loading dataset from %s", cfg.data_path)
    base_ds = load_dpo_dataset(cfg.data_path)
    ds = maybe_make_val_split(base_ds, cfg.eval_ratio, cfg.seed)

    LOG.info("Dataset sizes: %s", {k: len(v) for k, v in ds.items()})

    LOG.info("Loading tokenizer & model: %s", cfg.model_name)
    tokenizer = load_tokenizer(cfg.model_name, cfg.trust_remote_code)

    bnb_cfg = build_bnb_config(cfg.use_4bit, cfg.bf16)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora_targets),
    )

    model = load_model_with_lora(
        model_name=cfg.model_name,
        lcfg=lora_cfg,
        bnb_cfg=bnb_cfg,
        trust_remote_code=cfg.trust_remote_code,
    )

    ref_model = load_reference_model(cfg.ref_model_name, bnb_cfg, cfg.trust_remote_code)

    # DPO training configuration
    dpo_args = DPOConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy=cfg.eval_strategy,
        # Precision
        bf16=cfg.bf16,
        fp16=(cfg.fp16 and not cfg.bf16),
        # Sequence lengths (used by TRL tokenization under the hood)
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_target_length,
        # DPO specifics
        beta=cfg.beta,
        label_smoothing=cfg.label_smoothing,
        # Make sure padding value exists
        padding_value=tokenizer.pad_token_id,
        seed=cfg.seed,
        remove_unused_columns=False,
        report_to=[],  # set to ["wandb"] to enable Weights & Biases
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,           # None is fine (saves memory with LoRA)
        args=dpo_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        tokenizer=tokenizer,
    )

    LOG.info("Starting DPO training…")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    LOG.info("Saving model & tokenizer to %s", cfg.output_dir)
    trainer.model.save_pretrained(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    LOG.info("Done.")


def main() -> None:
    setup_logging()
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run(cfg)


if __name__ == "__main__":
    main()
