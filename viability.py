import os
import re
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from scipy.stats import spearmanr

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─────────────────────────────────────────────
# 1. Prompts
#    Three prompt styles matching literature:
#    (a) Absolute scoring (G-Eval style) — most common, most compression
#    (b) Comparative ranking (pairwise) — less compression
#    (c) Chain-of-thought then score — intermediate
#
#    We test (a) and (c) to show compression is robust to prompting style.
#    This strengthens the paper: even CoT prompting doesn't fix compression.
# ─────────────────────────────────────────────

ABSOLUTE_PROMPT = """You are evaluating the quality of an explanation.

Question: {question}
Explanation: {explanation}

Rate the quality of this explanation on a scale from 1 to 5:
1 = Nonsense or completely irrelevant
2 = Poor — wrong or misleading
3 = Fair — minimal but not wrong
4 = Good — correct and reasonably detailed
5 = Excellent — logically sound, factually accurate, and relevant

Respond with a single integer from 1 to 5. Nothing else."""

COT_PROMPT = """You are evaluating the quality of an explanation.

Question: {question}
Explanation: {explanation}

First, briefly assess the explanation's logical coherence, factual accuracy,
and relevance to the question. Then provide a quality score from 1 to 5:
1 = Nonsense, 2 = Poor, 3 = Fair, 4 = Good, 5 = Excellent

Format your response as:
Assessment: [your assessment in 1-2 sentences]
Score: [integer 1-5]"""

PROMETHEUS_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate,
a reference answer that gets a score of 5, and a score rubric representing
an evaluation criterion are given.

1. Write a detailed feedback that assesses the quality of the response
strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows:
Feedback: [write a feedback for criteria]
[RESULT] [an integer number between 1 and 5]

###The instruction to evaluate:
{question}

###Response to evaluate:
{explanation}

###Score Rubrics:
[Is the explanation logically coherent, factually accurate, and relevant?]
Score 1: Completely irrelevant or nonsensical.
Score 2: Wrong label or misleading reasoning.
Score 3: Minimal but not incorrect.
Score 4: Correct and reasonably detailed.
Score 5: Logically sound, factually accurate, directly answers the question.

###Feedback:"""


# ─────────────────────────────────────────────
# 2. DS-Critique loader — same as evaluate_task3_standalone.py
# ─────────────────────────────────────────────

def load_dscritique(cache_dir: str, n_queries: int = 200) -> List[Dict]:
    jsonl_files = list(Path(cache_dir).glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in {cache_dir}")

    raw = []
    for f in jsonl_files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

    groups = defaultdict(list)
    for rec in raw:
        qid = rec.get("question_id") or rec.get("qid") or rec.get("id", "unk")
        question = rec.get("question", rec.get("premise", ""))
        explanation = rec.get(
            "student_explanation",
            rec.get("explanation", rec.get("QA_reasoning_step1", ""))
        )
        score = rec.get(
            "critique_score",
            rec.get("quality_score", rec.get("score", None))
        )
        model = rec.get("student_model", rec.get("model", "unknown"))

        if not explanation or score is None:
            continue

        groups[qid].append({
            "text": explanation.strip(),
            "score": float(score),
            "model": model,
            "question": question
        })

    output = []
    for qid, exps in groups.items():
        if len(exps) < 2:
            continue
        scores = [e["score"] for e in exps]
        if len(set(scores)) == 1:
            continue
        output.append({
            "qid": str(qid),
            "question": exps[0]["question"],
            "explanations": exps
        })
        if len(output) >= n_queries:
            break

    return output


# ─────────────────────────────────────────────
# 3. Score parsers
#    Extract integer score from LLM output.
#    BOUNDARY CASES:
#    - Model refuses to score: return None
#    - Model outputs float (3.5): round to nearest int
#    - Model outputs range (3-4): take midpoint
#    - Model outputs outside 1-5: clip and warn
# ─────────────────────────────────────────────

def parse_score(text: str) -> Optional[float]:
    if text is None:
        return None

    text = text.strip()

    # Try "Score: X" pattern first (CoT and Prometheus format)
    m = re.search(r'(?:Score:|RESULT\]?)\s*([1-5](?:\.\d)?)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Try [RESULT] X pattern for Prometheus
    m = re.search(r'\[RESULT\]\s*([1-5])', text)
    if m:
        return float(m.group(1))

    # Try standalone integer at end of response
    m = re.search(r'\b([1-5])\b\s*$', text)
    if m:
        return float(m.group(1))

    # Try any integer 1-5 anywhere
    matches = re.findall(r'\b([1-5])\b', text)
    if matches:
        return float(matches[-1])  # take last occurrence

    return None


# ─────────────────────────────────────────────
# 4. Judge implementations
# ─────────────────────────────────────────────

class GPT4Judge:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("pip install openai")
        self.model = model

    def score(self, question: str, explanation: str,
              prompt_style: str = "absolute") -> Optional[float]:
        if prompt_style == "absolute":
            prompt = ABSOLUTE_PROMPT.format(
                question=question, explanation=explanation
            )
        else:
            prompt = COT_PROMPT.format(
                question=question, explanation=explanation
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50 if prompt_style == "absolute" else 150
            )
            text = resp.choices[0].message.content
            time.sleep(0.5)  # rate limit buffer
            return parse_score(text)
        except Exception as e:
            print(f"GPT-4o error: {e}")
            time.sleep(2)
            return None


class LocalLLMJudge:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_4bit: bool = True
    ):
        self.model_name = model_name
        self.device = device

        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # BOUNDARY CASE: decoder-only models need pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def _get_prompt(
        self,
        question: str,
        explanation: str,
        prompt_style: str
    ) -> str:
        if "prometheus" in self.model_name.lower():
            return PROMETHEUS_PROMPT.format(
                question=question, explanation=explanation
            )
        elif prompt_style == "absolute":
            return ABSOLUTE_PROMPT.format(
                question=question, explanation=explanation
            )
        else:
            return COT_PROMPT.format(
                question=question, explanation=explanation
            )

    def score(
        self,
        question: str,
        explanation: str,
        prompt_style: str = "absolute",
        max_new_tokens: int = 100
    ) -> Optional[float]:
        prompt = self._get_prompt(question, explanation, prompt_style)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return parse_score(text)


# ─────────────────────────────────────────────
# 5. Sep ratio and ranking metrics
#    Same formula as accepted paper
# ─────────────────────────────────────────────

def compute_sep_ratio(
    true_scores: List[float],
    pred_scores: List[float]
) -> Dict[str, float]:
    true_arr = np.array(true_scores)
    pred_arr = np.array(pred_scores)

    true_std = np.std(true_arr)
    pred_std = np.std(pred_arr)

    if true_std < 1e-8:
        return {"sep_ratio": 0.0, "pred_std": 0.0, "true_std": 0.0}

    rho, _ = spearmanr(true_arr, pred_arr)

    return {
        "sep_ratio": float(pred_std / true_std),
        "pred_std": float(pred_std),
        "true_std": float(true_std),
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "pred_mean": float(np.mean(pred_arr)),
        "pred_min": float(np.min(pred_arr)),
        "pred_max": float(np.max(pred_arr)),
        # Score distribution: what fraction of scores are in each bin?
        # This shows the ceiling/floor effect directly
        "dist_1": float(np.mean(pred_arr <= 1.5)),
        "dist_2": float(np.mean((pred_arr > 1.5) & (pred_arr <= 2.5))),
        "dist_3": float(np.mean((pred_arr > 2.5) & (pred_arr <= 3.5))),
        "dist_4": float(np.mean((pred_arr > 3.5) & (pred_arr <= 4.5))),
        "dist_5": float(np.mean(pred_arr > 4.5)),
    }


# ─────────────────────────────────────────────
# 6. Run a single judge on all groups
# ─────────────────────────────────────────────

def evaluate_judge(
    judge,
    groups: List[Dict],
    prompt_style: str = "absolute",
    judge_name: str = "unknown"
) -> Dict:
    all_true = []
    all_pred = []
    failed = 0
    total = 0

    for group in groups:
        question = group["question"]
        group_true = []
        group_pred = []

        for exp in group["explanations"]:
            total += 1
            pred = judge.score(
                question,
                exp["text"],
                prompt_style=prompt_style
            )

            if pred is None:
                failed += 1
                print(f"WARNING: {judge_name} failed to score "
                      f"group {group['qid']} model {exp['model']}")
                continue

            # Normalise true score from 1-5 to 0-1 (same as accepted paper)
            true_norm = (exp["score"] - 1) / 4.0
            pred_norm = (pred - 1) / 4.0

            group_true.append(true_norm)
            group_pred.append(pred_norm)

        if len(group_pred) >= 2:
            all_true.extend(group_true)
            all_pred.extend(group_pred)

    print(f"\n{judge_name} ({prompt_style} prompt):")
    print(f"  Total scored: {total - failed}/{total}")
    print(f"  Failed: {failed}")

    if not all_pred:
        print("  ERROR: no scores collected")
        return {"sep_ratio": 0.0, "error": "no scores"}

    metrics = compute_sep_ratio(all_true, all_pred)

    # Print interpretation
    sr = metrics["sep_ratio"]
    print(f"  Sep Ratio:    {sr:.4f}", end="  ")
    if sr < 0.2:
        print("→ COMPRESSED (paper viable ✓)")
    elif sr < 0.5:
        print("→ WEAK compression")
    elif sr < 0.8:
        print("→ MODERATE separation")
    else:
        print("→ STRONG (no compression)")

    print(f"  Spearman rho: {metrics['spearman_rho']:.4f}")
    print(f"  Score dist:   "
          f"1={metrics['dist_1']:.2f} "
          f"2={metrics['dist_2']:.2f} "
          f"3={metrics['dist_3']:.2f} "
          f"4={metrics['dist_4']:.2f} "
          f"5={metrics['dist_5']:.2f}")
    print(f"  Pred range:   [{metrics['pred_min']:.2f}, {metrics['pred_max']:.2f}]")

    return metrics


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Direction 4 viability check: LLM judge sep ratio on DS-Critique"
    )
    parser.add_argument(
        "--ds_critique_cache",
        type=str,
        default="data/ds_critique_bank"
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=200,
        help="Number of question groups to evaluate. 200 is sufficient for reliable sep ratio."
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=["llama3"],
        choices=["gpt4o", "prometheus", "llama3"],
        help="Which judges to run. Start with llama3 (free, local)."
    )
    parser.add_argument(
        "--prompt_styles",
        nargs="+",
        default=["absolute", "cot"],
        choices=["absolute", "cot"],
        help="Prompt styles to test. Both show compression persists regardless."
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="Required only if gpt4o judge selected"
    )
    parser.add_argument(
        "--prometheus_model",
        type=str,
        default="prometheus-eval/prometheus-7b-v2.0",
        help="HuggingFace model name for Prometheus"
    )
    parser.add_argument(
        "--llama3_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name for Llama-3"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/direction4_viability"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ──
    print("Loading DS-Critique Bank...")
    groups = load_dscritique(args.ds_critique_cache, n_queries=args.n_queries)

    # ── Reference: your ListNet encoder sep ratio ──
    # From accepted paper — anchor for comparison
    REFERENCE = {
        "listnet_encoder": 0.920,
        "mse_encoder": 0.089,
        "bradley_terry": 0.124
    }

    all_results = {"reference": REFERENCE, "judges": {}}

    # ── Run each judge ──
    for judge_name in args.judges:

        # Initialise judge
        if judge_name == "gpt4o":
            if not args.openai_api_key:
                print("Skipping gpt4o: no API key provided")
                continue
            judge = GPT4Judge(api_key=args.openai_api_key)

        elif judge_name == "prometheus":
            judge = LocalLLMJudge(
                model_name=args.prometheus_model,
                device=device,
                use_4bit=True
            )

        elif judge_name == "llama3":
            judge = LocalLLMJudge(
                model_name=args.llama3_model,
                device=device,
                use_4bit=True
            )

        all_results["judges"][judge_name] = {}

        # Run each prompt style
        for style in args.prompt_styles:
            print(f"\n{'='*60}")
            print(f"Running {judge_name} with {style} prompt...")
            print("="*60)

            metrics = evaluate_judge(
                judge, groups,
                prompt_style=style,
                judge_name=f"{judge_name}_{style}"
            )
            all_results["judges"][judge_name][style] = metrics

    # ── Final verdict ──
    print("\n" + "="*60)
    print("DIRECTION 4 VIABILITY VERDICT")
    print("="*60)
    print(f"\nReference sep ratios (from accepted paper):")
    print(f"  ListNet encoder RM:  {REFERENCE['listnet_encoder']:.3f}")
    print(f"  MSE encoder RM:      {REFERENCE['mse_encoder']:.3f}")
    print(f"  Bradley-Terry RM:    {REFERENCE['bradley_terry']:.3f}")

    print(f"\nLLM judge sep ratios:")
    viable = False
    for judge_name, styles in all_results["judges"].items():
        for style, metrics in styles.items():
            sr = metrics.get("sep_ratio", 0.0)
            print(f"  {judge_name} ({style}): {sr:.3f}")
            if sr < 0.3:
                viable = True

    print(f"\nVERDICT: ", end="")
    if viable:
        print("PAPER VIABLE ✓")
        print("  LLM judges show score compression (sep ratio < 0.3).")
        print("  Proceed to Step 2: fine-tune with ListNet and show fix.")
    else:
        print("WEAK — reconsider Direction 4")
        print("  LLM judges show moderate-to-strong separation already.")
        print("  The compression narrative may not hold on DS-Critique.")
        print("  Consider: different dataset, or reframe as 'LLM judges")
        print("  are less efficient than encoder RMs at equivalent sep ratio'.")

    # ── Save ──
    out_path = os.path.join(args.output_dir, "viability_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
