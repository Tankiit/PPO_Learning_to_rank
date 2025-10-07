"""
Build a 9x3 eSNLI dataset with generated explanations using a chat LLM.

This script:
1) Loads eSNLI from disk (HF Datasets format) and filters invalid labels.
2) Subsets train/validation/test (defaults: 20k / 2k / 2k).
3) Replicates each example into 9 question "families" × 3 variants (27 prompts):
   - Label cycles across (entails, neutral, contradicts)
   - Question templates rotate across: "why is that true?", "why is that not false?", "why is it the case?"
   - For indices (1_1, 4_1, 7_1) we keep the provided gold explanation_1, others are empty.
   - Fields produced: Question_i_j, Answer_i_j, label_i_j, gold_label_i_j
     (gold_label_i_j ∈ {1,0,0} indicates the "correct" label in each cycle).
4) Calls the LLM to generate missing explanations (Answer_* fields) in batches.
5) Saves the resulting DatasetDict back to disk.

Example:
    python build_esnli_9x3_dataset_with_llama.py \
        --dataset-path datasets/eSNLI \
        --output-dir datasets/my_dataset_eSNLI_9x3_20K \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --train-size 20000 --val-size 2000 --test-size 2000 \
        --max-prompt-length 128 --max-new-tokens 64 \
        --gen-batch-size 64 --use-quantization true --local-files-only false

Notes:
- Generation returns ONLY the assistant completion (prompt is stripped before decoding).
- Quantization (4-bit) requires a CUDA GPU. Set --use-quantization false to run full-precision.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

LOG = logging.getLogger("build_esnli_9x3")


# ---------------------------------------------------------------------------
# CLI & config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    dataset_path: str
    output_dir: str

    model_name: str
    trust_remote_code: bool
    local_files_only: bool
    use_quantization: bool
    use_bfloat16: bool

    train_size: int
    val_size: int
    test_size: int

    max_prompt_length: int
    max_new_tokens: int
    gen_batch_size: int

    seed: int


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(
        description="Build a 9x3 eSNLI dataset with generated explanations."
    )
    p.add_argument("--dataset-path", type=str, default="datasets/eSNLI", help="Path to load_from_disk() dataset.")
    p.add_argument("--output-dir", type=str, default="datasets/my_dataset_eSNLI_9x3_20K", help="Destination directory.")
    p.add_argument("--model-name", type=str, default="models/Llama-3.1-8B-Instruct", help="Causal LM checkpoint.")
    p.add_argument("--trust-remote-code", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--local-files-only", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--use-quantization", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--use-bfloat16", type=lambda x: str(x).lower() == "true", default=True)

    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--test-size", type=int, default=2000)

    p.add_argument("--max-prompt-length", type=int, default=128, help="Truncation length for prompts.")
    p.add_argument("--max-new-tokens", type=int, default=64, help="Generation length.")
    p.add_argument("--gen-batch-size", type=int, default=64, help="Micro-batch size for generation.")

    p.add_argument("--seed", type=int, default=42)

    a = p.parse_args()
    return RunConfig(
        dataset_path=a.dataset_path,
        output_dir=a.output_dir,
        model_name=a.model_name,
        trust_remote_code=a.trust_remote_code,
        local_files_only=a.local_files_only,
        use_quantization=a.use_quantization,
        use_bfloat16=a.use_bfloat16,
        train_size=a.train_size,
        val_size=a.val_size,
        test_size=a.test_size,
        max_prompt_length=a.max_prompt_length,
        max_new_tokens=a.max_new_tokens,
        gen_batch_size=a.gen_batch_size,
        seed=a.seed,
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data utils
# ---------------------------------------------------------------------------

LABEL_TEXT: Dict[int, str] = {
    0: "entails",
    1: "is neutral to",
    2: "contradicts",
}

QUESTION_TEMPLATES: List[str] = [
    "why is that true?",
    "why is that not false?",
    "why is it the case?",
]


def is_valid_label(ex: Dict) -> bool:
    """Keep only examples with a valid SNLI label."""
    return ex.get("label", -1) != -1


def build_chat_prompt(tokenizer: AutoTokenizer, premise: str, hypothesis: str, label_txt: str) -> str:
    """Return a chat-formatted prompt with a single-sentence instruction."""
    sys_msg = "Respond with a short explanation in a maximum of one sentence."
    user_msg = f"If: '{premise}' {label_txt}: '{hypothesis}', why is that true?"
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def replicate_examples_short(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """Replicate a single eSNLI example into 9x3 questions/answers/labels.

    - (i) runs from 1..9. For each i, label_index cycles over [0,1,2].
    - Question template changes every 3 steps: 1..3 -> 0, 4..6 -> 1, 7..9 -> 2.
    - Prefill gold explanation for (i in {1,4,7} and j == 1) using 'explanation_1'.
    """
    labels_cycle = [example["label"], (example["label"] + 1) % 3, (example["label"] + 2) % 3]
    gold_flags = [1, 0, 0]  # mark first label in cycle as "gold"

    for i in range(1, 10):  # 1..9
        label_index = (i - 1) % 3
        q_template = QUESTION_TEMPLATES[(i - 1) // 3]
        label_txt = LABEL_TEXT[labels_cycle[label_index]]

        # three variants j=1..3 for each i
        for j in range(1, 4):
            q_key = f"Question_{i}_{j}"
            a_key = f"Answer_{i}_{j}"
            l_key = f"label_{i}_{j}"
            g_key = f"gold_label_{i}_{j}"

            # Build user question with the selected template
            # Keep exact chat headers used by Llama-3 family
            example[q_key] = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "Respond with a short explanation in a maximum of one sentence."
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"If: '{example['premise']}' {label_txt}: '{example['hypothesis']}', "
                f"{q_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )

            # Prefill answers for (1_1), (4_1), (7_1)
            if i in (1, 4, 7) and j == 1:
                example[a_key] = example.get("explanation_1", "") or ""
            else:
                example[a_key] = ""

            example[l_key] = int(labels_cycle[label_index])
            example[g_key] = int(gold_flags[label_index])

    # (Optional) Also keep a single 'query' for inspection if useful
    example["query"] = build_chat_prompt(
        tokenizer, example["premise"], example["hypothesis"], LABEL_TEXT[int(example["label"])]
    )
    return example


# ---------------------------------------------------------------------------
# Model / generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    pad_token_id: int
    device: torch.device


def load_model(cfg: RunConfig) -> ModelBundle:
    """Load tokenizer + CausalLM with optional 4-bit quantization."""
    if cfg.use_quantization and not torch.cuda.is_available():
        raise RuntimeError("4-bit quantization requires a CUDA GPU. Set --use-quantization false to disable.")

    compute_dtype = torch.bfloat16 if cfg.use_bfloat16 else torch.float16
    quant_cfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        if cfg.use_quantization
        else None
    )

    model_config = AutoConfig.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        max_new_tokens=cfg.max_new_tokens,
        local_files_only=cfg.local_files_only,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        config=model_config,
        quantization_config=quant_cfg,
        device_map="auto",
        local_files_only=cfg.local_files_only,
        torch_dtype=compute_dtype,
    )

    # Best-effort device pick for input tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ModelBundle(tokenizer=tokenizer, model=model, pad_token_id=tokenizer.eos_token_id, device=device)


def _decode_only_new_tokens(
    tokenizer: AutoTokenizer,
    outputs: torch.LongTensor,
    input_ids: torch.LongTensor,
    pad_token_id: int,
) -> List[str]:
    """Decode only the generated continuation, not the original prompt.

    Handles left padding by measuring the non-pad length of each input row.
    """
    # Non-pad lengths per row
    with torch.no_grad():
        non_pad_lens = (input_ids != pad_token_id).sum(dim=1).tolist()

    texts: List[str] = []
    for i, seq in enumerate(outputs):
        start = non_pad_lens[i]
        gen_tokens = seq[start:]
        texts.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return texts


def generate_in_chunks(
    bundle: ModelBundle,
    prompts: Sequence[str],
    max_prompt_length: int,
    max_new_tokens: int,
    gen_batch_size: int,
) -> List[str]:
    """Generate assistant completions for a list of prompts, in micro-batches.

    Returns a list of decoded completions (without the prompt).
    """
    tokenizer, model = bundle.tokenizer, bundle.model
    pad_id = bundle.pad_token_id

    results: List[str] = []
    total = len(prompts)
    for start in range(0, total, gen_batch_size):
        end = min(start + gen_batch_size, total)
        batch_prompts = prompts[start:end]

        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        )
        # Important: with device_map="auto", many models tolerate CPU inputs;
        # for CUDA we explicitly move to GPU for speed.
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                use_cache=True,
            )

        # Decode only the continuation
        completions = _decode_only_new_tokens(tokenizer, outputs, inputs["input_ids"], pad_id)
        results.extend(completions)

    return results


# ---------------------------------------------------------------------------
# Map functions (batched)
# ---------------------------------------------------------------------------

QUESTION_KEYS: List[str] = [
    "Question_1_2", "Question_1_3",
    "Question_2_1", "Question_2_2", "Question_2_3",
    "Question_3_1", "Question_3_2", "Question_3_3",
    "Question_4_2", "Question_4_3",
    "Question_5_1", "Question_5_2", "Question_5_3",
    "Question_6_1", "Question_6_2", "Question_6_3",
    "Question_7_2", "Question_7_3",
    "Question_8_1", "Question_8_2", "Question_8_3",
    "Question_9_1", "Question_9_2", "Question_9_3",
]


def add_explanations_batched(
    examples: Dict[str, List[str]],
    bundle: ModelBundle,
    max_prompt_length: int,
    max_new_tokens: int,
    gen_batch_size: int,
) -> Dict[str, List[str]]:
    """Map function to fill Answer_* fields by generating completions for QUESTION_KEYS."""
    for q_key in QUESTION_KEYS:
        prompts = examples[q_key]
        completions = generate_in_chunks(
            bundle=bundle,
            prompts=prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            gen_batch_size=gen_batch_size,
        )
        a_key = q_key.replace("Question", "Answer")
        examples[a_key] = completions
    return examples


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_and_prepare_dataset(cfg: RunConfig) -> DatasetDict:
    """Load from disk, filter invalid labels, subset splits, and replicate examples."""
    LOG.info("Loading dataset from: %s", cfg.dataset_path)
    ds: DatasetDict = load_from_disk(cfg.dataset_path)
    LOG.info("Original splits: %s", {k: len(v) for k, v in ds.items()})

    # Subset
    ds = DatasetDict({
        "train": ds["train"].select(range(min(cfg.train_size, len(ds["train"])))),
        "validation": ds["validation"].select(range(min(cfg.val_size, len(ds["validation"])))),
        "test": ds["test"].select(range(min(cfg.test_size, len(ds["test"])))),
    })

    # Filter invalid labels
    ds = ds.filter(is_valid_label)

    # Tokenizer needed for the optional 'query' field in replicate
    tok = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    LOG.info("Replicating examples into 9x3 prompts...")
    for split in ds.keys():
        ds[split] = ds[split].map(
            lambda ex: replicate_examples_short(ex, tok),
            desc=f"replicate[{split}]",
        )
    LOG.info("After replication: %s", {k: len(v) for k, v in ds.items()})
    return ds


def run(cfg: RunConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    ds = load_and_prepare_dataset(cfg)
    LOG.info("Loading model: %s", cfg.model_name)
    bundle = load_model(cfg)

    # Generate explanations (batched) for each split
    for split in ds.keys():
        LOG.info("Generating explanations for split: %s", split)
        ds[split] = ds[split].map(
            lambda batch: add_explanations_batched(
                batch,
                bundle=bundle,
                max_prompt_length=cfg.max_prompt_length,
                max_new_tokens=cfg.max_new_tokens,
                gen_batch_size=cfg.gen_batch_size,
            ),
            batched=True,
            batch_size=2048,  # logical batch (will be micro-batched internally)
            desc=f"generate[{split}]",
        )

    # Save
    LOG.info("Saving dataset to: %s", cfg.output_dir)
    ds.save_to_disk(cfg.output_dir)
    LOG.info("Done. Example row: %s", ds["train"][0].get("Question_1_1", "")[:120] + "...")


def main() -> None:
    setup_logging()
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
