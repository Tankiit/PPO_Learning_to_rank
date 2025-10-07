"""
Generate critique annotations on Digital Socrates Critique Bank using a DS-style LLM.

This script:
1) Loads a dataset (either a Dataset or a DatasetDict+split) from disk.
2) Loads a prompt template (system + main) from a JSON file.
3) Builds Llama-style prompts with placeholders:
   [[QUESTION]], [[EXPLANATION]], [[PREDICTEDANSWER]], [[ANSWERKEY]]
4) Runs batched generation with a causal LLM (e.g., allenai/digital-socrates-7b).
5) Parses the model output into a structured critique dict and attaches it to each row.
6) Saves the enriched dataset back to disk.

Expected input fields per row:
    - "question"              (str)
    - "student_explanation"   (str)
    - "student_answer"        (one-letter 'A'..'D' typically)
    - "gold_answer"           (one-letter 'A'..'D' typically)

Output:
    - Adds a "new_critiques" field containing a list with one dict:
      {
        "critique_model": "<model_name>",
        "critique_llm_options": {"max_tokens": ..., "temperature": ...},
        "critique_text": "<raw LLM output>",
        "critique_elements": {
            "main_flaw": <str|None>,
            "dimension": <str|None>,
            "general_feedback": <str|None>,
            "specific_feedback": <str|None>,
            "explanation_score": <int|None>
        }
      }

Example:
    python xp_digital_socrates_v1.py \
        --dataset-path datasets/DSCB_dev_ppo_100K_ES_DPO \
        --output-dir datasets/DSCB_digital_socrates_v1_ppo_100K_ES_DPO \
        --prompts-json datasets/DS_Critique_Bank/DSCB-prompts.json \
        --prompt-set digital_socrates_v1 \
        --model allenai/digital-socrates-7b \
        --max-new-tokens 512 --temperature 0.0 --do-sample false \
        --gen-batch-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

LOG = logging.getLogger("dscb.critiques")


# -----------------------------------------------------------------------------
# CLI / Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    dataset_path: Path
    output_dir: Path
    split: Optional[str]

    prompts_json: Path
    prompt_set: str  # e.g., "digital_socrates_v1"

    model_name: str
    trust_remote_code: bool
    use_4bit: bool
    bf16: bool

    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    return_full_text: bool
    gen_batch_size: int
    limit: Optional[int]
    num_proc: Optional[int]


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Generate DS-style critiques for DSCB.")
    p.add_argument("--dataset-path", type=Path, required=True, help="Path to load_from_disk() dataset or dataset dict.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to save enriched dataset.")
    p.add_argument("--split", type=str, default=None, help="Split name if dataset_path is a DatasetDict (e.g., validation).")

    p.add_argument("--prompts-json", type=Path, required=True, help="JSON file with system/main templates.")
    p.add_argument("--prompt-set", type=str, default="digital_socrates_v1", help="Top-level key in the JSON.")

    p.add_argument("--model", dest="model_name", type=str, default="allenai/digital-socrates-7b")
    p.add_argument("--trust-remote-code", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--use-4bit", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)

    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--do-sample", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--return-full-text", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Cap number of examples processed.")
    p.add_argument("--num-proc", type=int, default=None, help="HF Datasets multiprocessing for map().")

    a = p.parse_args()
    return RunConfig(
        dataset_path=a.dataset_path,
        output_dir=a.output_dir,
        split=a.split,
        prompts_json=a.prompts_json,
        prompt_set=a.prompt_set,
        model_name=a.model_name,
        trust_remote_code=a.trust_remote_code,
        use_4bit=a.use_4bit,
        bf16=a.bf16,
        max_new_tokens=a.max_new_tokens,
        do_sample=a.do_sample,
        temperature=a.temperature,
        top_p=a.top_p,
        return_full_text=a.return_full_text,
        gen_batch_size=a.gen_batch_size,
        limit=a.limit,
        num_proc=a.num_proc,
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSet:
    system: str
    main: str


def load_prompts(json_path: Path, prompt_set: str) -> PromptSet:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if prompt_set not in data:
        raise KeyError(f"Prompt set '{prompt_set}' not found in {json_path}. Keys: {list(data.keys())}")
    entry = data[prompt_set]
    system = entry.get("system", "")
    main = entry.get("main", "")
    if not system or not main:
        raise ValueError(f"Prompt set '{prompt_set}' must define 'system' and 'main'.")
    return PromptSet(system=system, main=main)


def make_full_prompt(ps: PromptSet, question: str, explanation: str, predicted: str, answerkey: str) -> str:
    """Build a Llama-style chat prompt: <s>[INST] <<SYS>> ... <</SYS>> ... [/INST]."""
    user = (
        ps.main
        .replace("[[QUESTION]]", question)
        .replace("[[EXPLANATION]]", explanation)
        .replace("[[PREDICTEDANSWER]]", predicted)
        .replace("[[ANSWERKEY]]", answerkey)
    )
    return f"<s>[INST] <<SYS>>\n{ps.system}\n<</SYS>>\n\n{user} [/INST]"


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


def load_model_and_tokenizer(cfg: RunConfig):
    bnb = build_bnb_config(cfg.use_4bit, cfg.bf16)
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=cfg.trust_remote_code,
        quantization_config=bnb,
    )
    model.eval()
    return model, tok


# -----------------------------------------------------------------------------
# Parsing of the critique text
# -----------------------------------------------------------------------------

# Matches:
#   Main flaw (...) : "..."
_MAIN_FLAW_RE = re.compile(
    r'(?:Main flaw\s*\(standalone statement\)\s*:|Main flaw\s*:)\s*"(.*?)"',
    flags=re.IGNORECASE | re.DOTALL,
)
_DIMENSION_RE = re.compile(r"Dimension\s*:\s*([^\n*]+)", flags=re.IGNORECASE)
_GENERAL_RE = re.compile(r"General\s*:\s*(.*)", flags=re.IGNORECASE)
_SPECIFIC_RE = re.compile(r"Specific\s*:\s*(.*)", flags=re.IGNORECASE)
_SCORE_RE = re.compile(r"Explanation\s*score\s*:\s*([0-9]+)", flags=re.IGNORECASE)


def parse_critique_output(text: str, model_name: str, max_tokens: int, temperature: float) -> List[Dict]:
    """Parse Digital Socrates critique text into a normalized dict."""
    text = (text or "").strip()

    main_flaw_match = _MAIN_FLAW_RE.search(text)
    dimension_match = _DIMENSION_RE.search(text)
    general_match = _GENERAL_RE.search(text)
    specific_match = _SPECIFIC_RE.search(text)
    score_match = _SCORE_RE.search(text)  # noqa: E203 (space intentional for readability)

    # Clean overlaps (some lines may contain multiple sections)
    general_feedback = general_match.group(1).strip() if general_match else None
    if general_feedback:
        general_feedback = general_feedback.split("* Specific:")[0].strip()
    specific_feedback = specific_match.group(1).strip() if specific_match else None
    if specific_feedback:
        specific_feedback = specific_feedback.split("Explanation score:")[0].strip()

    try:
        explanation_score = int(score_match.group(1)) if score_match else None
    except (TypeError, ValueError):
        explanation_score = None

    return [{
        "critique_model": model_name,
        "critique_llm_options": {"max_tokens": max_tokens, "temperature": temperature},
        "critique_text": text,
        "critique_elements": {
            "main_flaw": (main_flaw_match.group(1).strip() if main_flaw_match else None),
            "dimension": (dimension_match.group(1).strip() if dimension_match else None),
            "general_feedback": general_feedback,
            "specific_feedback": specific_feedback,
            "explanation_score": explanation_score,
        },
    }]


# -----------------------------------------------------------------------------
# Batch generation
# -----------------------------------------------------------------------------

def build_generator(model, tokenizer):
    """Return a HF text-generation pipeline for convenience & batching."""
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )


def map_generate_batched(batch: Dict[str, List[str]], ps: PromptSet, gen, cfg: RunConfig) -> Dict[str, List]:
    """HF Datasets batched map() function: generates critiques for a batch of rows."""
    qs = batch.get("question", [])
    expls = batch.get("student_explanation", [])
    preds = batch.get("student_answer", [])
    keys = batch.get("gold_answer", [])

    prompts: List[str] = [
        make_full_prompt(ps, q or "", e or "", (p or "").strip(), (k or "").strip())
        for q, e, p, k in zip(qs, expls, preds, keys)
    ]

    results: List[str] = []
    for start in range(0, len(prompts), cfg.gen_batch_size):
        sub = prompts[start:start + cfg.gen_batch_size]
        outs = gen(
            sub,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature if cfg.do_sample else None,
            top_p=cfg.top_p if cfg.do_sample else None,
            return_full_text=cfg.return_full_text,
        )
        # outs is a list of list[{"generated_text": "..."}]
        for item in outs:
            if isinstance(item, list) and item:
                results.append(item[0].get("generated_text", ""))
            elif isinstance(item, dict) and "generated_text" in item:
                results.append(item["generated_text"])
            else:
                results.append("")

    critiques = [
        parse_critique_output(text=r, model_name=cfg.model_name, max_tokens=cfg.max_new_tokens, temperature=cfg.temperature)
        for r in results
    ]

    return {"new_critiques": critiques}


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def load_any_dataset(path: Path, split: Optional[str]) -> Dataset:
    """Load either a single Dataset or a DatasetDict+split, with an optional cap."""
    obj = load_from_disk(str(path))
    if isinstance(obj, DatasetDict):
        if not split:
            raise ValueError(f"{path} is a DatasetDict. Provide --split (available: {list(obj.keys())}).")
        ds = obj[split]
    else:
        ds = obj  # already a Dataset
    return ds


def run(cfg: RunConfig) -> None:
    LOG.info("Loading dataset from %s", cfg.dataset_path)
    ds = load_any_dataset(cfg.dataset_path, cfg.split)
    LOG.info("Rows loaded: %d", len(ds))

    if cfg.limit is not None:
        ds = ds.select(range(min(cfg.limit, len(ds))))
        LOG.info("Rows after --limit: %d", len(ds))

    LOG.info("Loading prompts from %s [set=%s]", cfg.prompts_json, cfg.prompt_set)
    ps = load_prompts(cfg.prompts_json, cfg.prompt_set)

    LOG.info("Loading model & tokenizer: %s", cfg.model_name)
    model, tokenizer = load_model_and_tokenizer(cfg)
    gen = build_generator(model, tokenizer)

    LOG.info("Generating critiques (batched=%d)…", cfg.gen_batch_size)
    out = ds.map(
        lambda batch: map_generate_batched(batch, ps=ps, gen=gen, cfg=cfg),
        batched=True,
        batch_size=cfg.gen_batch_size * 4,  # logical batch; pipeline chunks internally
        num_proc=cfg.num_proc,
        desc="critique-generation",
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Saving enriched dataset to %s", cfg.output_dir)
    out.save_to_disk(str(cfg.output_dir))
    LOG.info("Done. Example critiques: %s", out[0].get("new_critiques", [{}])[0]["critique_elements"])


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    config = parse_args()
    run(config)
