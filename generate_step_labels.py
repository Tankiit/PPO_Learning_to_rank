"""
generate_step_labels_vllm.py
============================
Local LLM-as-judge for step-level silver labels using vLLM offline
batch inference.  No OpenAI API key required.

Why vLLM over vanilla HuggingFace generate()
─────────────────────────────────────────────
  - Continuous batching: 15-40× throughput vs. HF generate on the same GPU
  - PagedAttention: fits larger batch sizes in the same VRAM
  - For 50k explanations × 3 steps avg = 150k calls:
      HF generate  @ 20 samples/min  ≈  125 hours  (impractical)
      vLLM         @ 400 samples/min ≈    6 hours   (one overnight run)

Recommended model
─────────────────
  meta-llama/Llama-3.1-8B-Instruct   (fits on 1× A100 40GB in bf16)
  mistralai/Mistral-7B-Instruct-v0.3 (slightly faster, slightly weaker)

Install
───────
  pip install vllm>=0.4.0

Run
───
  # Full dataset (~6h on 1× A100)
  python generate_step_labels_vllm.py \
      --data_dir data/processed/comprehensive_ranking_dataset \
      --split train \
      --output_path data/step_labels_train_vllm.jsonl \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --tensor_parallel_size 1 \
      --batch_size 256

  # Pilot: validate prompt on 500 examples first (always do this)
  python generate_step_labels_vllm.py \
      --n_samples 500 --split validation \
      --output_path data/step_labels_val_pilot.jsonl

Output format (same as generate_step_labels_llm.py — interchangeable)
──────────────────────────────────────────────────────────────────────
  {"qid": "esnli_123456", "step_idx": 0, "label": 1, "score": 4.2, "reason": "..."}

Boundary cases handled
──────────────────────
  1. JSON parse failure        → fallback to FreePRM heuristic label
  2. Model outputs refusal     → detected by checking for "label" key absence
  3. Steps < 5 tokens          → inherit explanation-level label directly
  4. Neutral NLI pairs         → prompt instructs model to evaluate
                                  internal coherence, not entailment validity
  5. Resume after interruption → skips already-annotated (qid, step_idx) pairs
  6. GPU OOM                   → reduce --batch_size; vLLM handles gracefully
"""

import os, re, json, argparse, logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from datasets import load_from_disk
from tqdm import tqdm

from train_step_ranking_model import split_into_steps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PROMPT TEMPLATES
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert annotator evaluating NLI explanation quality at the "
    "sentence level. Given an NLI triple (premise, hypothesis, label) and one "
    "sentence from an explanation, decide whether that sentence contributes "
    "positively to a logically valid, faithful, and coherent explanation.\n\n"
    "Respond ONLY with a JSON object. No markdown, no preamble. Format:\n"
    '{"label": 0_or_1, "score": float_1_to_5, "reason": "one sentence"}\n\n'
    "label=1 → sentence is correct, relevant, logically sound.\n"
    "label=0 → sentence is incorrect, irrelevant, contradictory, or nonsense."
)

USER_TEMPLATE = (
    "Premise:    {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Label:      {label}\n\n"
    "Full explanation: \"{explanation}\"\n\n"
    "Sentence to evaluate (step {step_idx} of {n_steps}):\n"
    "\"{step_text}\"\n\n"
    "Rate this sentence."
)

# Llama-3 / Mistral chat template applied manually
# (vLLM's tokenizer.apply_chat_template handles this automatically
#  when use_tqdm_on_prompts=True — we just pass the raw messages)
def build_prompt(premise, hypothesis, label, explanation,
                 step_idx, n_steps, step_text) -> str:
    user_msg = USER_TEMPLATE.format(
        premise=premise, hypothesis=hypothesis, label=label,
        explanation=explanation, step_idx=step_idx,
        n_steps=n_steps, step_text=step_text,
    )
    # Format as Llama-3 instruct chat turns.
    # vLLM will apply the model's own chat template via the tokenizer
    # when we pass a list-of-messages — but sending a pre-formatted
    # string is simpler and model-agnostic for offline inference.
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ---------------------------------------------------------------------------
# JSON PARSING WITH FALLBACK
# ---------------------------------------------------------------------------

def parse_response(
    raw: str,
    fallback_quality_score: float,
    quality_threshold: float = 3.0,
) -> Dict:
    """
    Parse vLLM output → {"label": int, "score": float, "reason": str}.

    Boundary cases
    ──────────────
    - Model outputs valid JSON                 → use directly
    - JSON has wrong keys / missing label      → fallback
    - Model wraps output in ```json ... ```    → strip fences first
    - Model outputs plain "1" or "0"           → treat as label directly
    - Completely unparseable                   → FreePRM fallback
    """
    raw = raw.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    # Try full JSON parse
    try:
        obj = json.loads(raw)
        if "label" in obj:
            return {
                "label":  int(obj["label"]),
                "score":  float(obj.get("score", fallback_quality_score)),
                "reason": str(obj.get("reason", "")),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting label digit from raw string
    m = re.search(r'"label"\s*:\s*([01])', raw)
    if m:
        return {
            "label":  int(m.group(1)),
            "score":  fallback_quality_score,
            "reason": "regex_extracted",
        }

    # Plain "1" or "0" response
    if raw in ("0", "1"):
        return {"label": int(raw), "score": fallback_quality_score,
                "reason": "plain_digit"}

    # Total failure → FreePRM fallback
    fallback_label = 1 if fallback_quality_score >= quality_threshold else 0
    return {
        "label":  fallback_label,
        "score":  fallback_quality_score,
        "reason": "fallback_parse_failure",
    }


# ---------------------------------------------------------------------------
# BUILD JOB LIST
# ---------------------------------------------------------------------------

def build_jobs(
    data: List[Dict],
    done: set,
    quality_threshold: float = 3.0,
    min_step_tokens: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns
    -------
    jobs         : list of dicts with all info needed to build prompt + store result
    trivial_recs : records that can be written immediately (short steps, already done)

    Boundary cases
    ──────────────
    - Steps shorter than min_step_tokens words → trivial_recs (no API call)
    - Already annotated (resume mode)          → skipped entirely
    """
    jobs, trivial_recs = [], []

    for ex in data:
        for cand in ex["explanations"]:
            steps = split_into_steps(cand["explanation"])
            for step_idx, step_text in enumerate(steps):
                qid = ex["query_id"]
                key = (qid, step_idx)

                if key in done:
                    continue

                # Trivial: too short to judge meaningfully
                if len(step_text.split()) < min_step_tokens:
                    label = 1 if cand["quality_score"] >= quality_threshold else 0
                    trivial_recs.append({
                        "qid":      qid,
                        "step_idx": step_idx,
                        "label":    label,
                        "score":    float(cand["quality_score"]),
                        "reason":   "inherited_too_short",
                    })
                    continue

                jobs.append({
                    "qid":           qid,
                    "step_idx":      step_idx,
                    "quality_score": float(cand["quality_score"]),
                    "premise":       ex["premise"],
                    "hypothesis":    ex["hypothesis"],
                    "label":         ex["label"],
                    "explanation":   cand["explanation"],
                    "n_steps":       len(steps),
                    "step_text":     step_text,
                })

    return jobs, trivial_recs


# ---------------------------------------------------------------------------
# VLLM INFERENCE
# ---------------------------------------------------------------------------

def run_vllm_inference(
    jobs:               List[Dict],
    model_name:         str,
    tensor_parallel:    int,
    batch_size:         int,
    max_new_tokens:     int,
    quality_threshold:  float,
) -> List[Dict]:
    """
    Runs vLLM offline batch inference over all jobs.
    Returns list of output records ready to write to JSONL.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM not installed. Run: pip install vllm>=0.4.0\n"
            "If on a CPU-only machine, use FreePRM (Strategy A) instead."
        )

    logger.info(f"Loading {model_name} with tensor_parallel_size={tensor_parallel}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,        # greedy — deterministic labels
        max_tokens=max_new_tokens,
        stop=["<|eot_id|>", "</s>", "<|end|>"],
    )

    results = []

    for i in tqdm(range(0, len(jobs), batch_size), desc="vLLM batches"):
        batch_jobs = jobs[i : i + batch_size]

        prompts = [
            build_prompt(
                j["premise"], j["hypothesis"], j["label"],
                j["explanation"], j["step_idx"], j["n_steps"], j["step_text"],
            )
            for j in batch_jobs
        ]

        outputs = llm.generate(prompts, sampling_params)

        for job, output in zip(batch_jobs, outputs):
            raw_text = output.outputs[0].text
            parsed   = parse_response(
                raw_text,
                fallback_quality_score=job["quality_score"],
                quality_threshold=quality_threshold,
            )
            results.append({
                "qid":      job["qid"],
                "step_idx": job["step_idx"],
                "label":    parsed["label"],
                "score":    parsed["score"],
                "reason":   parsed["reason"],
            })

    return results


# ---------------------------------------------------------------------------
# VALIDATION: check label distribution after pilot run
# ---------------------------------------------------------------------------

def validate_label_distribution(records: List[Dict], label_name: str = "pilot"):
    """
    Print label distribution statistics.
    Warn if the model is collapsing to all-1 or all-0.

    Expected healthy distribution: ~35-50% label=1
    (with 5 quality levels, levels 3+4 are positive → 2/5 = 40%)
    """
    labels = [r["label"] for r in records]
    n = len(labels)
    if n == 0:
        logger.warning("No records to validate!")
        return

    pos_rate = sum(labels) / n
    logger.info(f"\n{'='*50}")
    logger.info(f"Label distribution for {label_name} ({n} records)")
    logger.info(f"  label=1: {sum(labels):4d} ({100*pos_rate:.1f}%)")
    logger.info(f"  label=0: {n-sum(labels):4d} ({100*(1-pos_rate):.1f}%)")

    if pos_rate > 0.85:
        logger.warning(
            "⚠️  >85% positive labels — model may be saying everything is good. "
            "Check your prompt or lower quality_threshold."
        )
    elif pos_rate < 0.15:
        logger.warning(
            "⚠️  <15% positive labels — model may be too strict. "
            "Check that prompt format matches model's expected input."
        )
    else:
        logger.info("✅ Label distribution looks healthy")
    logger.info(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# ARGUMENT PARSING + MAIN
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate step-level silver labels using local vLLM inference"
    )
    p.add_argument("--data_dir",           default="data/processed/comprehensive_ranking_dataset")
    p.add_argument("--split",              default="train",
                   choices=["train", "validation", "test"])
    p.add_argument("--output_path",        default="data/step_labels_train_vllm.jsonl")
    p.add_argument("--model",              default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Any HF chat model supported by vLLM")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="Number of GPUs for tensor parallelism")
    p.add_argument("--batch_size",         type=int, default=256,
                   help="vLLM prompt batch size. Reduce if OOM.")
    p.add_argument("--max_new_tokens",     type=int, default=120)
    p.add_argument("--n_samples",          type=int, default=None,
                   help="Limit number of explanations (None = all). "
                        "Use 500 for pilot validation.")
    p.add_argument("--quality_threshold",  type=float, default=3.0,
                   help="FreePRM fallback: score >= threshold → label=1")
    p.add_argument("--min_step_tokens",    type=int, default=5,
                   help="Steps shorter than this inherit explanation label directly")
    p.add_argument("--resume",             action="store_true",
                   help="Skip already-annotated (qid, step_idx) pairs")
    p.add_argument("--validate_only",      action="store_true",
                   help="Run distribution check on existing output file and exit")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Validate-only mode ───────────────────────────────────────────────────
    if args.validate_only:
        assert Path(args.output_path).exists(), f"File not found: {args.output_path}"
        records = []
        with open(args.output_path) as f:
            for line in f:
                records.append(json.loads(line))
        validate_label_distribution(records, label_name=args.output_path)
        return

    # ── Load already-annotated pairs if resuming ─────────────────────────────
    done = set()
    if args.resume and Path(args.output_path).exists():
        with open(args.output_path) as f:
            for line in f:
                rec = json.loads(line)
                done.add((rec["qid"], rec["step_idx"]))
        logger.info(f"Resuming — {len(done)} pairs already annotated")

    # ── Load dataset ─────────────────────────────────────────────────────────
    logger.info(f"Loading {args.split} split from {args.data_dir}")
    raw  = load_from_disk(args.data_dir)
    data = list(raw[args.split])
    if args.n_samples:
        data = data[:args.n_samples]
    logger.info(f"Loaded {len(data)} examples")

    # ── Build job list ────────────────────────────────────────────────────────
    jobs, trivial_recs = build_jobs(
        data, done,
        quality_threshold=args.quality_threshold,
        min_step_tokens=args.min_step_tokens,
    )
    logger.info(f"Jobs to annotate: {len(jobs)} | Trivial (short): {len(trivial_recs)}")

    # ── Write trivial records immediately ─────────────────────────────────────
    with open(args.output_path, "a") as f:
        for rec in trivial_recs:
            f.write(json.dumps(rec) + "\n")

    if len(jobs) == 0:
        logger.info("No jobs remaining. All done.")
        return

    # ── vLLM inference ────────────────────────────────────────────────────────
    results = run_vllm_inference(
        jobs=jobs,
        model_name=args.model,
        tensor_parallel=args.tensor_parallel_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        quality_threshold=args.quality_threshold,
    )

    # ── Write results ─────────────────────────────────────────────────────────
    with open(args.output_path, "a") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"✅ Written {len(results)} records to {args.output_path}")

    # ── Validate distribution ──────────────────────────────────────────────────
    validate_label_distribution(results, label_name=f"{args.split} vLLM labels")


if __name__ == "__main__":
    main()