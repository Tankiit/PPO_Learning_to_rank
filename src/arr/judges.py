from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schema import RankingGroup, ScoreRecord
from .utils import hf_dtype_kwargs, read_jsonl, resolve_hf_source, stable_hash

DIRECT_SYSTEM_PROMPT = (
    "You are an impartial judge of explanation quality. Evaluate correctness, logical "
    "coherence and relevance to the supplied problem. Return only strict JSON. Do not "
    "add reasoning, commentary, markdown, or any text before or after the JSON object."
)
DIRECT_USER_TEMPLATE = """Problem:
{question}

Candidate explanation:
{candidate}

Rate the explanation from 0 (invalid) to 4 (fully correct). Return exactly:
{{"score": <number from 0 to 4>}}
End the response immediately after the closing brace."""

COT_SYSTEM_PROMPT = (
    "You are an impartial judge of explanation quality. Check factual/logical correctness, "
    "coherence and relevance. First provide one concise analysis sentence, then provide the "
    "score as strict JSON on the final line. Never put the JSON before the analysis."
)
COT_USER_TEMPLATE = """Problem:
{question}

Candidate explanation:
{candidate}

Use exactly this two-part format:
Analysis: <one concise sentence without braces>
{{"score": <number from 0 to 4>}}

The score must be between 0 and 4. Do not add anything after the closing brace."""

SCALAR_INPUT_TEMPLATE = "Problem:\n{question}\n\nExplanation:\n{candidate}"


def parse_score_json(raw_output: str, mode: str = "direct") -> float:
    """Parse the promised 0..4 score and normalise it to 0..1.

    ``direct`` accepts no prose around the object.  ``cot`` permits reasoning before
    it but requires the JSON object to be the final non-empty content.  No value is
    ever imputed when parsing fails.
    """

    stripped = raw_output.strip()
    if mode == "direct":
        candidate_json = stripped
    elif mode == "cot":
        match = re.search(r'(\{\s*"score"\s*:\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*\})\s*$', stripped)
        if match is None:
            raise ValueError("CoT output does not finish with the required score JSON")
        if not stripped[: match.start()].strip():
            raise ValueError("CoT output must contain reasoning before the final score JSON")
        candidate_json = match.group(1)
    else:
        raise ValueError(f"unknown prompt mode: {mode}")
    try:
        value = json.loads(candidate_json)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid score JSON") from exc
    if not isinstance(value, dict) or set(value) != {"score"}:
        raise ValueError('score JSON must contain exactly the key "score"')
    if isinstance(value["score"], bool) or not isinstance(value["score"], (int, float)):
        raise ValueError("score must be numeric")
    score = float(value["score"])
    if not 0.0 <= score <= 4.0:
        raise ValueError(f"score must be in 0..4, got {score}")
    return score / 4.0


def parse_listwise_json(raw_output: str, expected_count: int) -> list[float]:
    stripped = raw_output.strip()
    match = re.search(r'(\{\s*"scores"\s*:\s*\[[^\]]*\]\s*\})\s*$', stripped)
    if match is None:
        raise ValueError("listwise output must finish with a scores JSON object")
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid listwise JSON") from exc
    if not isinstance(value, dict) or set(value) != {"scores"} or not isinstance(value["scores"], list):
        raise ValueError('expected exactly {"scores": [...]}')
    if len(value["scores"]) != expected_count:
        raise ValueError(f"expected {expected_count} scores, found {len(value['scores'])}")
    scores: list[float] = []
    for item in value["scores"]:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("all listwise scores must be numeric")
        score = float(item)
        if not 0.0 <= score <= 4.0:
            raise ValueError(f"listwise score outside 0..4: {score}")
        scores.append(score / 4.0)
    return scores


class Judge(ABC):
    @abstractmethod
    def score(self, groups: Sequence[RankingGroup]) -> list[ScoreRecord]:
        """Return one auditable record for every real candidate."""


class PromptedJudge(Judge):
    def __init__(
        self,
        model_name: str,
        prompt_mode: str,
        revision: str = "main",
        seed: int = 42,
        batch_size: int = 4,
        max_new_tokens: int | None = None,
        max_input_length: int = 2048,
        cache_path: str | Path | None = None,
        device_map: str | dict[str, Any] = "auto",
        dtype: str = "bfloat16",
        local_files_only: bool = False,
    ) -> None:
        if prompt_mode not in {"direct", "cot"}:
            raise ValueError("prompt_mode must be direct or cot")
        self.model_name = model_name
        self.prompt_mode = prompt_mode
        self.requested_revision = revision
        self.seed = seed
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens or (64 if prompt_mode == "direct" else 256)
        self.max_input_length = max_input_length
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.device_map = device_map
        self.dtype = dtype
        self.local_files_only = local_files_only
        self._model: Any = None
        self._tokenizer: Any = None
        self.model_revision = revision
        template = DIRECT_USER_TEMPLATE if prompt_mode == "direct" else COT_USER_TEMPLATE
        system = DIRECT_SYSTEM_PROMPT if prompt_mode == "direct" else COT_SYSTEM_PROMPT
        self.prompt_hash = stable_hash("arr-prompt-v1", prompt_mode, system, template)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for prompted scoring") from exc
        dtype = getattr(torch, self.dtype)
        source = resolve_hf_source(
            self.model_name, self.requested_revision, self.local_files_only
        )
        self._tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            source,
            **hf_dtype_kwargs(dtype),
            device_map=self.device_map,
        )
        self._model.eval()
        self.model_revision = str(
            getattr(self._model.config, "_commit_hash", None) or self.requested_revision
        )

    def render_prompt(self, question: str, candidate: str) -> str:
        system = DIRECT_SYSTEM_PROMPT if self.prompt_mode == "direct" else COT_SYSTEM_PROMPT
        template = DIRECT_USER_TEMPLATE if self.prompt_mode == "direct" else COT_USER_TEMPLATE
        return self.render_custom_prompt(template.format(question=question, candidate=candidate), system)

    def render_custom_prompt(self, user_content: str, system_prompt: str) -> str:
        self._load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System: {system_prompt}\nUser: {messages[1]['content']}\nAssistant:"

    def generate_texts(self, prompts: Sequence[str]) -> list[tuple[str, float]]:
        self._load()
        import torch

        outputs: list[tuple[str, float]] = []
        torch.manual_seed(self.seed)
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            encoded = self._tokenizer(
                list(batch_prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
            )
            device = next(self._model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    stop_strings=["}"],
                    tokenizer=self._tokenizer,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(batch_prompts)
            prompt_length = encoded["input_ids"].shape[1]
            new_tokens = generated[:, prompt_length:]
            decoded = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend((text, elapsed_ms) for text in decoded)
        return outputs

    def _cached_records(self) -> dict[tuple[str, str, str, str, str], ScoreRecord]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        output: dict[tuple[str, str, str, str, str], ScoreRecord] = {}
        for value in read_jsonl(self.cache_path):
            record = ScoreRecord.from_dict(value)
            key = (
                record.group_id,
                record.candidate_id,
                record.model_name,
                record.model_revision,
                record.prompt_hash,
            )
            output[key] = record
        return output

    def score(self, groups: Sequence[RankingGroup]) -> list[ScoreRecord]:
        self._load()
        cached = self._cached_records()
        complete: list[ScoreRecord | None] = []
        pending: list[tuple[int, RankingGroup, Any, str]] = []
        for group in groups:
            for candidate in group.candidates:
                key = (
                    group.group_id,
                    candidate.candidate_id,
                    self.model_name,
                    self.model_revision,
                    self.prompt_hash,
                )
                old = cached.get(key)
                if old is not None and old.data_fingerprint == group.data_fingerprint:
                    complete.append(
                        ScoreRecord(
                            **{
                                **old.to_dict(),
                                "parsing_status": "cached",
                                "metadata": {
                                    **old.metadata,
                                    "cached_original_status": old.parsing_status,
                                },
                            }
                        )
                    )
                    continue
                prompt = self.render_prompt(group.question, candidate.text)
                index = len(complete)
                complete.append(None)
                pending.append((index, group, candidate, prompt))
        generated = self.generate_texts([item[3] for item in pending])
        for (index, group, candidate, _), (raw_output, inference_ms) in zip(pending, generated):
            try:
                score = parse_score_json(raw_output, self.prompt_mode)
                status = "ok"
                error = None
            except ValueError as exc:
                score = None
                status = "parse_error"
                error = str(exc)
            complete[index] = ScoreRecord(
                group_id=group.group_id,
                candidate_id=candidate.candidate_id,
                model_name=self.model_name,
                model_revision=self.model_revision,
                prompt_hash=self.prompt_hash,
                data_fingerprint=group.data_fingerprint,
                score=score,
                raw_output=raw_output,
                parsing_status=status,
                seed=self.seed,
                inference_ms=inference_ms,
                metadata={"prompt_mode": self.prompt_mode, "parse_error": error},
            )
        return [record for record in complete if record is not None]


class ScalarJudge(Judge):
    """Hugging Face/PEFT scalar head with a sigmoid output contract."""

    def __init__(
        self,
        checkpoint: str | Path,
        seed: int = 42,
        batch_size: int = 16,
        max_length: int = 512,
        device_map: str | dict[str, Any] = "auto",
        dtype: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.seed = seed
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_map = device_map
        self.dtype = dtype
        self.local_files_only = local_files_only
        self._model: Any = None
        self._tokenizer: Any = None
        self.model_name = str(self.checkpoint)
        self.model_revision = "local"
        self.prompt_hash = stable_hash("arr-scalar-input-v1", SCALAR_INPUT_TEMPLATE)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for scalar scoring") from exc
        manifest_path = self.checkpoint / "arr_model_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base_model = str(manifest.get("base_model", self.checkpoint))
        base_revision = str(manifest.get("model_revision", "main"))
        resolved_base = resolve_hf_source(base_model, base_revision, self.local_files_only)
        tokenizer_source = self.checkpoint if any(self.checkpoint.glob("tokenizer*")) else resolved_base
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), use_fast=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype = getattr(torch, self.dtype or str(manifest.get("dtype", "bfloat16")))
        adapter_config = self.checkpoint / "adapter_config.json"
        if adapter_config.exists():
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("peft is required to load this adapter checkpoint") from exc
            base = AutoModelForSequenceClassification.from_pretrained(
                resolved_base,
                num_labels=1,
                device_map=self.device_map,
                **hf_dtype_kwargs(dtype),
            )
            self._model = PeftModel.from_pretrained(base, str(self.checkpoint))
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.checkpoint), device_map=self.device_map, **hf_dtype_kwargs(dtype)
            )
        # Decoder sequence classifiers cannot score batches larger than one
        # unless the model config knows which padded positions to ignore.  PEFT
        # adapter checkpoints do not persist this base-model config mutation.
        self._model.config.pad_token_id = self._tokenizer.pad_token_id
        if hasattr(self._model, "get_base_model"):
            self._model.get_base_model().config.pad_token_id = self._tokenizer.pad_token_id
        self._model.eval()
        self.model_name = str(manifest.get("base_model", self.checkpoint))
        self.model_revision = str(manifest.get("model_revision", "local"))

    def score_texts(self, questions: Sequence[str], candidates: Sequence[str]) -> list[tuple[float, float]]:
        if len(questions) != len(candidates):
            raise ValueError("questions and candidates must have the same length")
        self._load()
        import torch

        results: list[tuple[float, float]] = []
        texts = [
            SCALAR_INPUT_TEMPLATE.format(question=question, candidate=candidate)
            for question, candidate in zip(questions, candidates)
        ]
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            device = next(self._model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = self._model(**encoded).logits.squeeze(-1)
                scores = torch.sigmoid(logits).float().cpu().tolist()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(batch)
            results.extend((float(score), elapsed_ms) for score in scores)
        return results

    def score(self, groups: Sequence[RankingGroup]) -> list[ScoreRecord]:
        flattened = [(group, candidate) for group in groups for candidate in group.candidates]
        predictions = self.score_texts(
            [group.question for group, _ in flattened],
            [candidate.text for _, candidate in flattened],
        )
        return [
            ScoreRecord(
                group_id=group.group_id,
                candidate_id=candidate.candidate_id,
                model_name=self.model_name,
                model_revision=self.model_revision,
                prompt_hash=self.prompt_hash,
                data_fingerprint=group.data_fingerprint,
                score=score,
                raw_output=f"{score:.10f}",
                parsing_status="ok",
                seed=self.seed,
                inference_ms=inference_ms,
                metadata={"checkpoint": str(self.checkpoint), "bounded_by": "sigmoid"},
            )
            for (group, candidate), (score, inference_ms) in zip(flattened, predictions)
        ]
