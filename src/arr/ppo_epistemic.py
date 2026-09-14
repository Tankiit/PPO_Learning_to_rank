"""PPO over explanation quality with an epistemic penalty on the reward.

``train_ppo_trl.py`` computes ``reward = scores[0]``: everything the epistemic
machinery produces is discarded before it reaches the objective. This module
closes that loop.

Claim under test -- a ranking reward model is overconfident on explanations far
from its training distribution, PPO finds and exploits exactly those regions,
and subtracting credal width prevents it. Three arms:

    none    r = mean_h s_h                      current behaviour
    var     r = mean_h s_h - lam * std_h s_h    scalar penalty, conflates channels
    credal  r = mean_h s_h - lam * width        separates them

TRL VERSION
``train_ppo_trl.py`` calls ``ppo_trainer.step(queries, responses, rewards)``.
That method was removed in TRL 0.12; the rewrite takes policy, ref_policy,
reward_model and value_model and you call ``.train()``. So the penalty lives
inside the reward model's ``.score`` module rather than in the PPO loop --
that is the seam TRL's own ``get_reward`` calls, and the same attachment point
``training.py`` already uses via ``model.score = diverse_head``. Both API paths
below therefore share one penalty implementation.

Pin explicitly: ``trl==0.11.4`` for legacy, ``trl>=0.14`` for modern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .epistemic import build_diverse_scalar_head, credal_summary
from .utils import hf_dtype_kwargs

PPO_PROMPT_TEMPLATE = "Problem:\n{question}\n\nExplanation:\n"

PenaltyMode = Literal["none", "var", "credal"]


def _load_checkpoint_state(checkpoint: Path) -> dict:
    """Read a full (non-adapter) checkpoint's tensors."""

    from safetensors.torch import load_file

    shard = checkpoint / "model.safetensors"
    if shard.exists():
        return load_file(str(shard))
    binary = checkpoint / "pytorch_model.bin"
    if binary.exists():
        import torch

        return torch.load(binary, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no model weights found in {checkpoint}")


@dataclass
class RewardSpec:
    checkpoint: Path
    penalty: PenaltyMode = "credal"
    lam: float = 1.0
    mc_dropout: int = 0
    batch_size: int = 16
    max_length: int = 512
    local_files_only: bool = False


class _PenaltyScore(nn.Module):
    """The ``.score`` seam TRL's ``get_reward`` calls.

    Keeping the penalty here rather than in the PPO loop is what lets the
    modern and legacy API paths share one implementation.
    """

    def __init__(self, head: nn.Module, penalty: PenaltyMode, lam: float) -> None:
        super().__init__()
        if penalty not in ("none", "var", "credal"):
            raise ValueError(f"unknown penalty {penalty}")
        self.head = head
        self.penalty = penalty
        self.lam = float(lam)

    def components(self, hidden_states: Any) -> dict[str, Any]:
        scores = torch.sigmoid(self.head(hidden_states).float())
        summary = credal_summary(scores)
        reward = summary["mean"]
        if self.penalty == "var":
            reward = reward - self.lam * summary["variance"].sqrt()
        elif self.penalty == "credal":
            reward = reward - self.lam * summary["width"]
        return {**summary, "reward": reward}

    def forward(self, hidden_states: Any) -> Any:
        """Returns [..., 1] so TRL can index the final non-pad position."""
        return self.components(hidden_states)["reward"].unsqueeze(-1)


class EpistemicRewardModel(nn.Module):
    """Wraps an ``EpistemicScalarJudge`` checkpoint as a TRL reward model.

    Must be an ``nn.Module``: TRL calls ``disable_dropout_in_model`` on the
    reward model before training, which walks ``.modules()``. Exposes the
    backbone under ``base_model_prefix`` and the penalty as ``.score``, which
    is exactly what TRL's ``get_reward`` reaches for.
    """

    def __init__(self, spec: RewardSpec) -> None:
        super().__init__()
        self.spec = spec
        self._tokenizer = None
        self._loaded = False

    # -- construction -------------------------------------------------------
    def load(self) -> "EpistemicRewardModel":
        if self._loaded:
            return self
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        manifest_path = Path(self.spec.checkpoint) / "arr_model_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing checkpoint manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        head_count = int(manifest["epistemic_head_count"])
        base_model = manifest["base_model"]

        self._tokenizer = AutoTokenizer.from_pretrained(
            base_model, local_files_only=self.spec.local_files_only
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Honour the checkpoint's dtype. Without this the backbone loads at the
        # config's dtype while the freshly built heads stay float32, and the
        # first matmul dies on Half vs Float.
        model_dtype = getattr(torch, str(manifest.get("dtype", "float32")))
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=head_count,
            local_files_only=self.spec.local_files_only,
            **hf_dtype_kwargs(model_dtype),
        )
        head, expected = build_diverse_scalar_head(
            int(base.config.hidden_size),
            head_count,
            int(manifest.get("epistemic_hidden_dim", 256)),
            float(manifest.get("epistemic_dropout_min", 0.05)),
            float(manifest.get("epistemic_dropout_max", 0.30)),
            self.spec.mc_dropout or int(manifest.get("epistemic_mc_dropout", 0)),
            float(manifest.get("epistemic_feature_keep_fraction", 1.0)),
            int(manifest.get("epistemic_feature_seed", 0)),
        )
        saved = [float(v) for v in manifest.get("epistemic_dropout_rates", [])]
        if saved and not all(abs(a - b) < 1e-12 for a, b in zip(saved, expected)):
            raise ValueError("checkpoint dropout schedule does not match its manifest")
        head.to(device=next(base.parameters()).device, dtype=model_dtype)
        base.score = head
        base.config.num_labels = head_count
        base.config.pad_token_id = self._tokenizer.pad_token_id

        # A QLoRA run leaves an adapter; a full-finetune run does not. Loading
        # unconditionally through peft makes every non-adapter checkpoint
        # unusable, so mirror the existing check in ``judges.py``.
        if (Path(self.spec.checkpoint) / "adapter_config.json").exists():
            from peft import PeftModel

            merged = PeftModel.from_pretrained(base, str(self.spec.checkpoint))
            merged = merged.merge_and_unload()
        else:
            state = _load_checkpoint_state(Path(self.spec.checkpoint))
            _, unexpected = base.load_state_dict(state, strict=False)
            if unexpected:
                raise ValueError(f"unexpected tensors in checkpoint: {sorted(unexpected)[:5]}")
            merged = base

        prefix = merged.base_model_prefix
        backbone = getattr(merged, prefix)
        # Register the backbone and the head exactly once, under the names TRL
        # looks them up by. Keeping the sequence-classification wrapper around
        # as well would double-count every parameter.
        self.base_model_prefix = prefix
        setattr(self, prefix, backbone)
        self.score = _PenaltyScore(merged.score, self.spec.penalty, self.spec.lam)
        self.config = merged.config
        self.config.pad_token_id = self._tokenizer.pad_token_id
        self.eval()
        self._loaded = True
        return self

    # -- TRL seam -----------------------------------------------------------
    def _hidden(self, input_ids: Any, attention_mask: Any) -> Any:
        backbone = getattr(self, self.base_model_prefix)
        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]

    def forward(self, input_ids: Any, attention_mask: Any = None) -> Any:
        return self.score(self._hidden(input_ids, attention_mask))

    def reward_with_stats(self, texts: list[str]) -> tuple[Any, dict[str, Any]]:
        """Per-sequence reward plus components. Log ``width`` every step -- the
        headline figure is the width trajectory of generated explanations."""

        self.load()
        encoded = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.spec.max_length,
        )
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            hidden = self._hidden(encoded["input_ids"], encoded["attention_mask"])
            index = encoded["attention_mask"].sum(1) - 1
            last = hidden[torch.arange(hidden.shape[0], device=device), index]
            parts = self.score.components(last)
        return parts["reward"], {
            "mean": parts["mean"],
            "width": parts["width"],
            "std": parts["variance"].sqrt(),
            "lower": parts["lower"],
        }


def build_prompt_dataset(groups: Any, tokenizer: Any, max_length: int = 512) -> Any:
    """Turn RankingGroups into the tokenised query set TRL's PPOTrainer wants.

    TRL iterates ``train_dataset`` expecting an ``input_ids`` column; a list of
    RankingGroup objects is not that. One query per group -- PPO generates the
    explanation, so only the question is the prompt.
    """

    from datasets import Dataset

    prompts = [PPO_PROMPT_TEMPLATE.format(question=group.question) for group in groups]
    input_ids = [
        tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
        for prompt in prompts
    ]
    # input_ids only: TRL's collator pads every column, and a string "query"
    # column makes it fail on "excessive nesting". Prompts are recovered by
    # decoding when a caller needs them.
    return Dataset.from_dict({"input_ids": input_ids})


def train_ppo(
    reward_spec: RewardSpec,
    policy_model: str,
    dataset: Any,
    output_dir: Path,
    trl_api: Literal["modern", "legacy"] = "modern",
    learning_rate: float = 1.41e-5,
    batch_size: int = 64,
    mini_batch_size: int = 16,
    ppo_epochs: int = 4,
    kl_coef: float = 0.2,
    max_new_tokens: int = 128,
    seed: int = 0,
    total_episodes: int | None = None,
    dtype: str = "float32",
) -> None:
    """Keep ``kl_coef`` FIXED across the three arms.

    The penalty shrinks reward magnitude, so an adaptive KL controller silently
    changes the effective constraint and the comparison stops meaning anything.
    """
    reward_model = EpistemicRewardModel(reward_spec).load()
    output_dir.mkdir(parents=True, exist_ok=True)
    if reward_spec.mc_dropout:
        # TRL calls disable_dropout_in_model on the reward model, which zeroes
        # every nn.Dropout it owns -- including the heads' -- so MC dropout is
        # silently defeated under PPO. Fail loudly instead of reporting a
        # spread that is not there.
        raise ValueError(
            "mc_dropout is incompatible with PPO: TRL disables dropout in the "
            "reward model. Use lambda_div for head diversity instead."
        )
    prompts = build_prompt_dataset(dataset, reward_model._tokenizer, reward_spec.max_length)

    # A half-precision policy overflows during sampling and generate() dies on
    # "probability tensor contains inf, nan or element < 0" a step or two in.
    # Default to float32 and let a CUDA run opt into bfloat16 explicitly.
    policy_kwargs = hf_dtype_kwargs(getattr(torch, dtype))

    if trl_api == "modern":
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
        from trl import PPOConfig, PPOTrainer

        config = PPOConfig(
            output_dir=str(output_dir),
            learning_rate=learning_rate,
            per_device_train_batch_size=mini_batch_size,
            num_ppo_epochs=ppo_epochs,
            # Fixed across arms, never adaptive: the penalty shrinks reward
            # magnitude, so an adaptive controller would silently change the
            # effective constraint and the comparison would stop meaning
            # anything.
            kl_coef=kl_coef,
            response_length=max_new_tokens,
            seed=seed,
            **({"total_episodes": total_episodes} if total_episodes else {}),
        )
        trainer = PPOTrainer(
            args=config,
            processing_class=reward_model._tokenizer,
            model=AutoModelForCausalLM.from_pretrained(policy_model, **policy_kwargs),
            ref_model=AutoModelForCausalLM.from_pretrained(policy_model, **policy_kwargs),
            reward_model=reward_model,
            value_model=AutoModelForSequenceClassification.from_pretrained(
                policy_model, num_labels=1, **policy_kwargs
            ),
            train_dataset=prompts,
            # TRL builds an eval dataloader unconditionally and dies on
            # len(None) at the end of training if this is left out.
            eval_dataset=prompts,
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        return

    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

    config = PPOConfig(
        model_name=policy_model,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        init_kl_coef=kl_coef,
        adap_kl_ctrl=False,
        seed=seed,
    )
    trainer = PPOTrainer(
        config=config,
        model=AutoModelForCausalLMWithValueHead.from_pretrained(policy_model, **policy_kwargs),
        ref_model=AutoModelForCausalLMWithValueHead.from_pretrained(policy_model, **policy_kwargs),
        tokenizer=reward_model._tokenizer,
        dataset=prompts,
    )
    generation = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": reward_model._tokenizer.eos_token_id,
        "max_new_tokens": max_new_tokens,
    }
    for batch in trainer.dataloader:
        queries = batch["input_ids"]
        responses = trainer.generate(queries, return_prompt=False, **generation)
        decoded = reward_model._tokenizer.batch_decode(responses)
        prompts_text = reward_model._tokenizer.batch_decode(
            queries, skip_special_tokens=True
        )
        texts = [q + r for q, r in zip(prompts_text, decoded)]
        rewards, stats = reward_model.reward_with_stats(texts)
        step_stats = trainer.step(queries, list(responses), list(rewards.cpu()))
        step_stats["credal/width_mean"] = float(stats["width"].mean())
        step_stats["credal/width_p95"] = float(stats["width"].quantile(0.95))
        step_stats["credal/reward_mean"] = float(rewards.mean())
        trainer.log_stats(step_stats, batch, rewards)
    trainer.save_pretrained(str(output_dir))
