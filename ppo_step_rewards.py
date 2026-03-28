"""
ppo_step_reward.py
==================
Drop-in replacement for compute_binary_rewards() in ppo-TRL.py.

Two reward modes
────────────────
  "min_step"    — reward = min over step scores (weakest-step penalises whole)
  "mean_step"   — reward = mean over step scores
  "expl"        — reward = explanation-level score from [CLS] head (baseline)

Usage in ppo-TRL.py
───────────────────
  # Replace:
  #   rewards = compute_binary_rewards(queries, responses)
  # With:
  from ppo_step_reward import StepRewardModel
  step_rm = StepRewardModel(
      model_path="results/step_prm_esnli_listnet/best_model.pt",
      encoder_name="microsoft/deberta-v3-base",
      device=device,
  )
  rewards = step_rm.compute_rewards(queries, responses, mode="min_step")

Boundary cases
──────────────
  - Single-step explanation: min == mean == the step score. No special case needed.
  - Failed tokenisation (empty response): returns reward 0.0.
  - Longer responses than max_length: truncated; any steps beyond the window
    are ignored (conservative — worst case the model sees fewer steps).
"""

import re, torch
import torch.nn.functional as F
from typing import List
from transformers import AutoTokenizer

from train_step_ranking_model import (
    StepLevelRankingModel,
    build_step_input,
    extract_user_from_llama2,
    STEP_TOKEN,
)


class StepRewardModel:
    def __init__(
        self,
        model_path: str,
        encoder_name: str = "microsoft/deberta-v3-base",
        device: str = "cuda",
        max_length: int = 256,
    ):
        self.device     = torch.device(device)
        self.max_length = max_length

        # Load tokenizer and re-add [STEP] token
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.tokenizer.add_tokens([STEP_TOKEN])

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        saved_args = checkpoint.get("args", {})
        hidden_size = 768  # DeBERTa-v3-base default

        self.model = StepLevelRankingModel(
            encoder_name=encoder_name,
            hidden_size=hidden_size,
            intermediate_size=hidden_size // 2,
        ).to(self.device)
        self.model.encoder.resize_token_embeddings(checkpoint["tokenizer_len"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def compute_rewards(
        self,
        queries: List[str],
        responses: List[str],
        mode: str = "min_step",
    ) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        queries   : raw Llama-2 chat queries (contain premise/hypothesis)
        responses : model-generated explanations
        mode      : "min_step" | "mean_step" | "expl"

        Returns
        -------
        List of scalar tensors, one per (query, response) pair.
        """
        rewards = []
        for query, response in zip(queries, responses):
            if not response.strip():
                rewards.append(torch.tensor([0.0], device=self.device))
                continue

            # Extract the NLI question from the Llama-2 query
            user_text = extract_user_from_llama2(query)

            # Parse premise and hypothesis from user_text
            # Format: "If: '{premise}' {label_txt}: '{hypothesis}', why is that true?"
            premise, hypothesis = _parse_premise_hypothesis(user_text)

            try:
                enc = build_step_input(
                    premise, hypothesis, response,
                    self.tokenizer, self.max_length,
                )
            except Exception:
                rewards.append(torch.tensor([0.0], device=self.device))
                continue

            input_ids   = enc["input_ids"].unsqueeze(0).to(self.device)
            attn_mask   = enc["attention_mask"].unsqueeze(0).to(self.device)
            step_pos    = [enc["step_positions"]]

            out = self.model(input_ids, attn_mask, step_pos)

            if mode == "min_step":
                r = out["expl_scores"][0]   # already min-pooled
            elif mode == "mean_step":
                step_scores = out["step_scores"][0]
                r = step_scores.mean()
            elif mode == "expl":
                r = out["cls_scores"][0] if out["cls_scores"] is not None \
                    else out["expl_scores"][0]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            rewards.append(torch.tensor([float(r)], device=self.device))

        return rewards


def _parse_premise_hypothesis(user_text: str):
    """
    Extract premise and hypothesis from the NLI query string.
    Handles the format produced by build_chat_query() in ppo-TRL.py:
      "If: '{premise}' {label_txt}: '{hypothesis}', why is that true?"

    Returns ("", "") if parsing fails — the model will still run
    but quality of step segmentation will be lower.
    """
    m = re.search(
        r"If:\s*['\"]?(.+?)['\"]?\s+(?:entails|is neutral to|contradicts):\s*['\"]?(.+?)['\"]?,",
        user_text, flags=re.DOTALL
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", user_text  # fallback: treat whole user_text as hypothesis


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Smoke test: StepRewardModel instantiation and forward pass")
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ppo_step_reward.py <model_path>")
        sys.exit(0)

    model_path = sys.argv[1]
    srm = StepRewardModel(model_path=model_path, device="cpu")

    fake_query    = "[INST] <<SYS>> Respond with a short explanation. <</SYS>> If: 'A man runs in the park' entails: 'Someone is exercising', why is that true? [/INST]"
    fake_response = "Running is a form of physical exercise. Therefore being in the park and running implies exercise."

    rewards = srm.compute_rewards([fake_query], [fake_response], mode="min_step")
    print(f"Reward (min_step): {rewards[0].item():.4f}")

    rewards_mean = srm.compute_rewards([fake_query], [fake_response], mode="mean_step")
    print(f"Reward (mean_step): {rewards_mean[0].item():.4f}")

    print("✅ Smoke test passed")