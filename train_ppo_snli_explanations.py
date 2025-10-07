"""
Train PPO policy on SNLI to generate short explanations, using:
- TRL (PPO with value head)
- LoRA adapters (PEFT)
- Optional 4-bit quantization via bitsandbytes
- A frozen binary reward model (2-class classifier)
- Windowed early stopping on average reward

This script:
1) Builds chat-style prompts from SNLI (premise/hypothesis/label).
2) Generates one-sentence explanations with a chat LLM.
3) Scores (query + explanation) with a frozen binary reward model.
4) Optimizes the policy with PPO.
5) Logs to Weights & Biases and saves best checkpoints by reward.

Example:
    python train_ppo_snli_explanations.py \
        --policy-model meta-llama/Llama-2-7b-chat-hf \
        --reward-model ../abductive-reasoning/models/models_bert-base-uncased_CLS-2-labels_my-eSNLI-9x3_20k_batch_32_epoch_1 \
        --save-dir saved_policy_llama2_snli_trl_EarlyS_Reward \
        --total-ppo-steps 20000 --batch-size 16 --mini-batch-size 4 \
        --window-size 500 --patience-windows 7 --min-delta 1e-4 \
        --use-quantization false

Notes:
- Assumes the reward classifier has exactly 2 labels (bad/good).
- Uses chat template via tokenizer.apply_chat_template(add_generation_prompt=True).
- Early stopping evaluates non-overlapping windows of reward means.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG = logging.getLogger("ppo_snli_explanations")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a simple, production-friendly formatter."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    # Models & paths
    policy_model: str
    reward_model: str
    save_dir: str
    save_best_dir: str

    # Data
    seed: int
    max_prompt_length: int
    reward_max_length: int

    # PPO & generation
    total_ppo_steps: int
    max_new_tokens: int
    learning_rate: float
    batch_size: int
    mini_batch_size: int
    ppo_epochs: int
    init_kl_coef: float

    # Generation sampling
    top_p: float
    top_k: int
    do_sample: bool

    # Early stopping on reward
    early_stop_reward: bool
    window_size: int
    patience_windows: int
    min_delta: float

    # Hardware / precision
    use_quantization: bool
    use_bfloat16: bool


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="TRL PPO training on SNLI for short explanations with a binary reward model."
    )

    # Models / paths
    parser.add_argument(
        "--policy-model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base chat LLM (e.g., meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default="models/models_bert-base-uncased_CLS-2-labels_my-eSNLI-9x3_20k_batch_32_epoch_1",
        help="Path to the frozen reward classifier (2 labels).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_policy_mistral_7B_snli_trl_EarlyS_Reward",
        help="Directory to save final policy (LoRA adapters + value head).",
    )

    # Data & seeds
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=128,
        help="Truncation length for input prompts before generation.",
    )
    parser.add_argument(
        "--reward-max-length",
        type=int,
        default=256,
        help="Max length for reward model tokenization.",
    )

    # PPO & generation
    parser.add_argument("--total-ppo-steps", type=int, default=20000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--init-kl-coef", type=float, default=0.02)

    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--do-sample", type=lambda x: str(x).lower() == "true", default=True)

    # Early stopping (windowed average reward)
    parser.add_argument("--early-stop-reward", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--patience-windows", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    # Hardware / precision
    parser.add_argument("--use-quantization", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--use-bfloat16", type=lambda x: str(x).lower() == "true", default=True)

    args = parser.parse_args()

    save_best_dir = os.path.join(args.save_dir, "best_by_reward")

    return RunConfig(
        policy_model=args.policy_model,
        reward_model=args.reward_model,
        save_dir=args.save_dir,
        save_best_dir=save_best_dir,
        seed=args.seed,
        max_prompt_length=args.max_prompt_length,
        reward_max_length=args.reward_max_length,
        total_ppo_steps=args.total_ppo_steps,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        init_kl_coef=args.init_kl_coef,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        early_stop_reward=args.early_stop_reward,
        window_size=args.window_size,
        patience_windows=args.patience_windows,
        min_delta=args.min_delta,
        use_quantization=args.use_quantization,
        use_bfloat16=args.use_bfloat16,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_chat_query(tokenizer: AutoTokenizer, premise: str, hypothesis: str, label_txt: str) -> str:
    """Format a chat-style prompt asking for a short explanation.

    Args:
        tokenizer: Policy tokenizer supporting .apply_chat_template()
        premise: SNLI premise
        hypothesis: SNLI hypothesis
        label_txt: Text label among {'entails', 'is neutral to', 'contradicts'}

    Returns:
        A chat-formatted string with generation prompt enabled.
    """
    sys_msg = "Respond with a short explanation in a maximum of one sentence."
    user_msg = f"If: '{premise}' {label_txt}: '{hypothesis}', why is that true?"
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_user_from_llama2(query: str) -> str:
    """Extract the [INST]...[/INST] user content, ignoring <<SYS>>...<</SYS>> if present.

    This is a best-effort parser intended to build an input for the reward model.
    If parsing fails, the original query is returned.

    Args:
        query: Chat-formatted prompt (Llama-2 style).

    Returns:
        Extracted user content as plain text.
    """
    # 1) Grab [INST] ... [/INST]
    m = re.search(r"\[INST\](.*)\[/INST\]", query, flags=re.DOTALL)
    if not m:
        LOG.warning("Failed to parse [INST] block; returning full query for reward input.")
        return query
    inside = m.group(1)

    # 2) Remove <<SYS>> ... <</SYS>> if present
    m2 = re.search(r"<<SYS>>(.+?)<</SYS>>(.*)", inside, flags=re.DOTALL)
    user_part = m2.group(2) if m2 else inside

    # 3) Cleanup residual special tokens/spaces
    user_part = re.sub(r"<\|.*?\|>", " ", user_part).strip()
    return user_part


def compute_binary_rewards(
    queries: Sequence[str],
    responses: Sequence[str],
    reward_tokenizer: AutoTokenizer,
    reward_model: AutoModelForSequenceClassification,
    device: str,
    max_length: int,
) -> List[torch.Tensor]:
    """Compute binary rewards given (query, response) pairs.

    The reward text is built as: "<extracted user prompt> <generated explanation>".

    Args:
        queries: List of chat-formatted prompts.
        responses: List of generated explanations (decoded strings).
        reward_tokenizer: Tokenizer for the reward classifier.
        reward_model: Frozen reward classifier with 2 labels.
        device: Device string ('cuda' or 'cpu').
        max_length: Max tokenization length for reward model.

    Returns:
        List of 0D tensors (shape [1]) containing reward probabilities for the "good" class.
    """
    prefixes = [extract_user_from_llama2(q) for q in queries]
    texts = [(f"{p} {r}").strip() for p, r in zip(prefixes, responses)]

    enc = reward_tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)

    with torch.no_grad():
        logits = reward_model(**enc).logits  # [B, 2]
        probs = torch.softmax(logits, dim=-1)
        good = probs[:, 1]

    return [torch.tensor([float(v)], device=device) for v in good]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_snli_queries(tokenizer: AutoTokenizer, seed: int) -> Iterable[dict]:
    """Load SNLI and map to a single 'query' column (chat prompt)."""
    snli = load_dataset("snli")  # train/validation/test
    label2txt = {0: "entails", 1: "is neutral to", 2: "contradicts"}

    def keep_valid(ex) -> bool:
        return (ex.get("label", -1) != -1) and bool(ex.get("premise")) and bool(ex.get("hypothesis"))

    def map_to_query(ex):
        chat = build_chat_query(tokenizer, ex["premise"], ex["hypothesis"], label2txt[int(ex["label"])])
        return {"query": chat}

    train_ds = (
        snli["train"]
        .filter(keep_valid)
        .shuffle(seed=seed)
        .map(map_to_query, remove_columns=snli["train"].column_names)
    )

    return train_ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_policy_and_tokenizer(cfg: RunConfig):
    """Load policy tokenizer and model (LoRA + value head)."""
    policy_tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model, use_fast=True)
    # Chat models often need left padding to keep the generation aligned
    policy_tokenizer.pad_token_id = policy_tokenizer.eos_token_id
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.use_bfloat16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if cfg.use_quantization
        else None
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Value head is added by TRL's AutoModelForCausalLMWithValueHead
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_model,
        peft_config=peft_config,
        quantization_config=quant_config,
        dtype=torch.bfloat16 if cfg.use_bfloat16 else torch.float16,
        device_map="auto",
    )
    return policy_tokenizer, model


def load_reward_model_and_tokenizer(cfg: RunConfig, device: str):
    """Load frozen binary reward classifier."""
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model, use_fast=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.reward_model, dtype=torch.bfloat16 if cfg.use_bfloat16 else torch.float16
    ).to(device)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    if getattr(reward_model.config, "num_labels", None) != 2:
        raise ValueError("Reward model must have exactly 2 labels.")

    return reward_tokenizer, reward_model


def run_training(cfg: RunConfig) -> None:
    """Main PPO training loop with windowed early stopping on reward."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.save_best_dir, exist_ok=True)

    LOG.info("Loading policy & tokenizer: %s", cfg.policy_model)
    policy_tokenizer, policy_model = load_policy_and_tokenizer(cfg)

    LOG.info("Loading reward model (frozen): %s", cfg.reward_model)
    reward_tokenizer, reward_model = load_reward_model_and_tokenizer(cfg, device=device)

    LOG.info("Preparing dataset: SNLI (train split)")
    train_ds = build_snli_queries(policy_tokenizer, seed=cfg.seed)

    # PPO configuration
    ppo_cfg = PPOConfig(
        model_name=cfg.policy_model,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        init_kl_coef=cfg.init_kl_coef,
        seed=cfg.seed,
        log_with="wandb",
        tracker_project_name="ppo-train-all-models",
    )

    ppo_trainer = PPOTrainer(
        model=policy_model,
        config=ppo_cfg,
        tokenizer=policy_tokenizer,
        dataset=train_ds,
    )

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        pad_token_id=policy_tokenizer.eos_token_id,
    )

    # Early stopping buffers
    recent_rewards: List[float] = []
    best_window_avg = float("-inf")
    best_window_idx = -1
    no_improve_windows = 0

    LOG.info("Starting PPO loop for %d steps.", cfg.total_ppo_steps)

    for step, batch in enumerate(tqdm(ppo_trainer.dataloader, desc="PPO")):
        if step >= cfg.total_ppo_steps:
            break

        # TRL yields only the 'query' column from our mapped dataset
        queries: List[str] = batch["query"]

        # Truncate prompts to ensure room for generation
        tokenized = policy_tokenizer(
            queries,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=cfg.max_prompt_length,
        )
        query_tensors = [torch.tensor(ids, device=device) for ids in tokenized["input_ids"]]

        # Generation
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, **gen_kwargs
            )
        responses = policy_tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Reward (binary prob of "good" class)
        rewards = compute_binary_rewards(
            queries=queries,
            responses=responses,
            reward_tokenizer=reward_tokenizer,
            reward_model=reward_model,
            device=device,
            max_length=cfg.reward_max_length,
        )

        # PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Logging
        ppo_trainer.log_stats(
            stats,
            batch={"query": queries, "response": responses},
            rewards=rewards,
        )

        # Compact console prints via accelerator to avoid duplication across ranks
        mean_reward_this_batch = (torch.stack([r.detach().float().squeeze() for r in rewards]).mean().item())
        ppo_trainer.accelerator.print(
            {
                "step": step,
                "reward_mean": round(mean_reward_this_batch, 6),
                "kl": stats.get("objective/kl", None),
                "kl_coef": stats.get("objective/kl_coef", None),
                "gen_lens": [int(x.shape[-1]) for x in response_tensors],
                "response_0": (responses[0][:160] + ("…" if len(responses[0]) > 160 else "")) if responses else "",
            }
        )

        # ---------------------------
        # Windowed early stopping
        # ---------------------------
        if cfg.early_stop_reward:
            recent_rewards.append(mean_reward_this_batch)

            # Evaluate a non-overlapping window every window_size steps
            if (step + 1) % cfg.window_size == 0:
                window_idx = (step + 1) // cfg.window_size
                window_avg = float(sum(recent_rewards) / len(recent_rewards))

                # W&B and console
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.accelerator.log(
                        {
                            "early_stop/window_idx": window_idx,
                            "early_stop/window_reward_mean": window_avg,
                            "early_stop/best_window_reward_mean": best_window_avg,
                            "early_stop/no_improve_windows": no_improve_windows,
                        },
                        step=step,
                    )

                ppo_trainer.accelerator.print(
                    {
                        "early_stop/window_idx": window_idx,
                        "window_reward_mean": round(window_avg, 6),
                        "best_window_reward_mean": (
                            round(best_window_avg, 6) if best_window_idx >= 0 else None
                        ),
                        "no_improve_windows": no_improve_windows,
                    }
                )

                # Check improvement and save best checkpoint
                if window_avg > best_window_avg + cfg.min_delta:
                    best_window_avg = window_avg
                    best_window_idx = window_idx
                    no_improve_windows = 0

                    ppo_trainer.save_pretrained(cfg.save_best_dir)
                    policy_tokenizer.save_pretrained(cfg.save_best_dir)
                    ppo_trainer.accelerator.print(
                        f"✅ New best checkpoint (window {window_idx}) saved to {cfg.save_best_dir}"
                    )
                else:
                    no_improve_windows += 1
                    if no_improve_windows >= cfg.patience_windows:
                        ppo_trainer.accelerator.print(
                            "🛑 Early stopping: no improvement over "
                            f"{cfg.patience_windows} windows "
                            f"(best_mean={best_window_avg:.6f}, window={best_window_idx})."
                        )
                        break

                # Clear buffer for the next non-overlapping window
                recent_rewards.clear()

    # Final save (LoRA adapters + value head)
    ppo_trainer.save_pretrained(cfg.save_dir)
    LOG.info("Policy saved to: %s", cfg.save_dir)
    LOG.info(
        "Best checkpoint (by reward): %s (window=%s, mean=%.6f)",
        cfg.save_best_dir,
        best_window_idx,
        best_window_avg,
    )


def main() -> None:
    setup_logging()
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
