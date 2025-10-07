"""
Generate model predictions (explanation + answer) on Digital Socrates Critique Bank.

Two modes:
- baseline : load a base chat LLM (optionally 4-bit) and generate directly.
- adapter  : load a base LLM + LoRA adapters (e.g., DPO fine-tuned) via PEFT.

The script:
1) Loads the DSCB DatasetDict from disk (e.g., "train"/"validation"/"test").
2) Loads prompt templates JSON (system + 3 prompt variants).
3) Filters the evaluation split by student model name (optional).
4) Builds a Llama-style chat prompt from the template and the question.
5) Generates with HF pipeline, parses the model output to extract:
   - student_raw_output, student_explanation, student_answer, student_accuracy (0/1)
6) Saves the enriched split to disk.

Example (LoRA adapters):
    python xp_QA_student_model.py \
      --mode adapter \
      --adapter-dir ./dpo-llama2-7b-qlora \
      --dataset-path datasets/Digital_Socrate_Critique_Bank \
      --prompts-json datasets/DS_Critique_Bank/DSCB-prompts.json \
      --split validation \
      --student-model-filter Llama-2-7b-chat \
      --output-dir datasets/DSCB_dev_ppo_100K_ES_DPO

Example (baseline, 4-bit):
    python xp_QA_student_model.py \
      --mode baseline \
      --model-name meta-llama/Llama-2-7b-chat-hf \
      --use-4bit true \
      --dataset-path datasets/Digital_Socrate_Critique_Bank \
      --prompts-json datasets/DS_Critique_Bank/DSCB-prompts.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

LOG = logging.getLogger("dscb_generate")


# -----------------------------------------------------------------------------
# CLI / Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    # Mode & IO
    mode: str  # "baseline" | "adapter"
    dataset_path: Path
    split: str
    prompts_json: Path
    output_dir: Path

    # Filtering
    student_model_filter: Optional[str]
    max_examples: Optional[int]

    # Model options
    model_name: Optional[str]
    adapter_dir: Optional[str]
    trust_remote_code: bool
    use_4bit: bool
    bf16: bool
    fp16: bool

    # Generation
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    return_full_text: bool


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Generate predictions on Digital Socrates Critique Bank.")

    # Mode & IO
    p.add_argument("--mode", type=str, choices=["baseline", "adapter"], default="adapter")
    p.add_argument("--dataset-path", type=Path, default=Path("datasets/Digital_Socrate_Critique_Bank"))
    p.add_argument("--split", type=str, default="validation", help="Split to process.")
    p.add_argument("--prompts-json", type=Path, default=Path("datasets/DS_Critique_Bank/DSCB-prompts.json"))
    p.add_argument("--output-dir", type=Path, default=Path("datasets/DSCB_dev_predictions"))

    # Filtering
    p.add_argument("--student-model-filter", type=str, default="Llama-2-7b-chat",
                   help="Keep only rows with this student_model (substring match). Empty to keep all.")
    p.add_argument("--max-examples", type=int, default=None)

    # Model options
    p.add_argument("--model-name", type=str, default=None,
                   help="Required in baseline mode; optional in adapter mode (read from PEFT config if omitted).")
    p.add_argument("--adapter-dir", type=str, default="./dpo-llama2-7b-qlora",
                   help="PEFT adapter dir (required in adapter mode).")
    p.add_argument("--trust-remote-code", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--use-4bit", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=False)

    # Generation
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--do-sample", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--return-full-text", type=lambda x: str(x).lower() == "true", default=False)

    a = p.parse_args()
    return RunConfig(
        mode=a.mode,
        dataset_path=a.dataset_path,
        split=a.split,
        prompts_json=a.prompts_json,
        output_dir=a.output_dir,
        student_model_filter=a.student_model_filter or None,
        max_examples=a.max_examples,
        model_name=a.model_name,
        adapter_dir=a.adapter_dir,
        trust_remote_code=a.trust_remote_code,
        use_4bit=a.use_4bit,
        bf16=a.bf16,
        fp16=a.fp16,
        max_new_tokens=a.max_new_tokens,
        do_sample=a.do_sample,
        temperature=a.temperature,
        top_p=a.top_p,
        return_full_text=a.return_full_text,
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSet:
    sys_tmpl: str
    zero_tmpl: str
    expl_tmpl: str
    reas_tmpl: str


def load_prompts(json_path: Path) -> PromptSet:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    sys_tmpl = data["QA_zeroshot1"]["system"]
    zero_tmpl = data["QA_zeroshot1"]["main"]
    expl_tmpl = data["QA_explanation1"]["main"]
    reas_tmpl = data["QA_reasoning_step1"]["main"]
    return PromptSet(sys_tmpl=sys_tmpl, zero_tmpl=zero_tmpl, expl_tmpl=expl_tmpl, reas_tmpl=reas_tmpl)


def select_prompt_template(student_prompt: str, prompts: PromptSet) -> str:
    key = (student_prompt or "").strip()
    if key == "QA_explanation1":
        return prompts.expl_tmpl
    if key == "QA_reasoning_step1":
        return prompts.reas_tmpl
    # default to zero-shot
    return prompts.zero_tmpl


def build_chat_prompt(system_tmpl: str, user_tmpl_with_question: str) -> str:
    """Build a Llama-2 style prompt with system + user."""
    return (
        f"<s>[INST] <<SYS>>\n{system_tmpl}\n<</SYS>>\n\n"
        f"{user_tmpl_with_question} [/INST]"
    )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def build_bnb_config(use_4bit: bool, bf16: bool) -> Optional[BitsAndBytesConfig]:
    if not use_4bit:
        return None
    compute = torch.bfloat16 if bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute,
    )


@dataclass(frozen=True)
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def load_baseline_model(cfg: RunConfig) -> ModelBundle:
    if not cfg.model_name:
        raise ValueError("--model-name is required in baseline mode.")
    bnb = build_bnb_config(cfg.use_4bit, cfg.bf16)

    # Optional model config (max_new_tokens hint, etc.)
    _ = AutoConfig.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=False,
    )

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype="auto",
    )
    return ModelBundle(model=model, tokenizer=tok)


def load_adapter_model(cfg: RunConfig) -> ModelBundle:
    if not cfg.adapter_dir:
        raise ValueError("--adapter-dir is required in adapter mode.")
    peft_cfg = PeftConfig.from_pretrained(cfg.adapter_dir)
    base_model_name = cfg.model_name or peft_cfg.base_model_name_or_path
    if not base_model_name:
        raise ValueError("Base model name could not be inferred from PEFT config; please set --model-name.")

    tok = AutoTokenizer.from_pretrained(cfg.adapter_dir, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=cfg.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, cfg.adapter_dir)
    return ModelBundle(model=model, tokenizer=tok)


def load_model(cfg: RunConfig) -> ModelBundle:
    return load_adapter_model(cfg) if cfg.mode == "adapter" else load_baseline_model(cfg)


# -----------------------------------------------------------------------------
# Parsing generated text
# -----------------------------------------------------------------------------

# Regex to capture:
#   Explanation: <text>  \n Answer: (A)
# or  Reasoning:  <text>  \n Answer: (B)
_EXPL_ANS_RE = re.compile(
    r"(?:Explanation|Reasoning)\s*:\s*(.*?)[\r\n]+.*?Answer\s*:\s*\(([A-D])\)",
    flags=re.IGNORECASE | re.DOTALL,
)

def parse_explanation_and_answer(generated: str) -> tuple[str, str]:
    """Return (explanation, answer_letter) with best-effort fallbacks."""
    if not generated:
        return "", ""
    m = _EXPL_ANS_RE.search(generated)
    if m:
        expl = m.group(1).strip()
        letter = m.group(2).upper().strip()
        return expl, letter

    # Fallbacks
    # try to split at 'Answer:' first
    idx = re.search(r"Answer\s*:", generated, flags=re.IGNORECASE)
    if idx:
        expl = generated[: idx.start()].strip()
        # try to find letter
        letter_m = re.search(r"\(([A-D])\)", generated[idx.start():], flags=re.IGNORECASE)
        letter = letter_m.group(1).upper() if letter_m else ""
        return expl, letter

    # last resort: everything is explanation, no letter
    return generated.strip(), ""


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------

def process_row(
    ex: Dict,
    prompts: PromptSet,
    generator,
) -> Dict:
    """Build prompt, generate, parse, and enrich the row with new fields."""
    question = (ex.get("question") or "").strip()
    template = select_prompt_template(ex.get("student_prompt"), prompts)
    user = template.replace("[[QUESTION]]", question)
    prompt = build_chat_prompt(prompts.sys_tmpl, user)

    out = generator(
        prompt,
        max_new_tokens=cfg.max_new_tokens,   # will be injected via closure in run()
        do_sample=cfg.do_sample,
        temperature=cfg.temperature if cfg.do_sample else None,
        top_p=cfg.top_p if cfg.do_sample else None,
        return_full_text=cfg.return_full_text,
    )
    # pipeline returns a list[{"generated_text": "..."}]
    response = out[0]["generated_text"] if out and isinstance(out, list) else ""

    expl, letter = parse_explanation_and_answer(response)
    gold = str(ex.get("gold_answer", "")).strip().upper()
    acc = int(letter.upper() == gold) if letter else 0

    ex["student_raw_output"] = response
    ex["student_explanation"] = expl
    ex["student_answer"] = letter
    ex["student_accuracy"] = acc
    return ex


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def run(cfg: RunConfig) -> None:
    LOG.info("Loading dataset from: %s", cfg.dataset_path)
    ds: DatasetDict = load_from_disk(str(cfg.dataset_path))
    if cfg.split not in ds:
        raise ValueError(f"Split '{cfg.split}' not found. Available: {list(ds.keys())}")

    split_ds: Dataset = ds[cfg.split]
    LOG.info("Split size before filtering: %d", len(split_ds))

    if cfg.student_model_filter:
        needle = cfg.student_model_filter
        split_ds = split_ds.filter(lambda x: needle in str(x.get("student_model", "")))
        LOG.info("After student_model filter (%s): %d", needle, len(split_ds))

    if cfg.max_examples is not None:
        split_ds = split_ds.select(range(min(cfg.max_examples, len(split_ds))))
        LOG.info("After max_examples cap: %d", len(split_ds))

    LOG.info("Loading prompts from: %s", cfg.prompts_json)
    prompts = load_prompts(cfg.prompts_json)

    LOG.info("Loading %s model...", cfg.mode)
    bundle = load_model(cfg)
    model, tokenizer = bundle.model, bundle.tokenizer
    model.eval()

    LOG.info("Building generation pipeline…")
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    # Map with closure over cfg & gen
    def _map_fn(ex):
        return process_row(ex, prompts=prompts, generator=gen)

    LOG.info("Generating for %d examples…", len(split_ds))
    out_ds = split_ds.map(_map_fn, desc=f"generate[{cfg.split}]")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir
    LOG.info("Saving enriched split to: %s", out_path)
    out_ds.save_to_disk(str(out_path))
    LOG.info("Done. Example:\n%s", {k: out_ds[0][k] for k in ["student_explanation", "student_answer", "student_accuracy"]})


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    cfg = parse_args()
    run(cfg)
