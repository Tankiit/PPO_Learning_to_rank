from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def _plain_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class Candidate:
    """One explanation candidate and its reference score in ``[0, 1]``."""

    candidate_id: str
    text: str
    score: float
    score_provenance: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not self.text.strip():
            raise ValueError("candidate text must be non-empty; use candidate_mask for padding")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(f"candidate score must be in [0, 1], got {self.score}")
        if not self.score_provenance.strip():
            raise ValueError("score_provenance must be explicit")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Candidate":
        return cls(
            candidate_id=str(value["candidate_id"]),
            text=str(value["text"]),
            score=float(value["score"]),
            score_provenance=str(value["score_provenance"]),
            metadata=_plain_mapping(value.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankingGroup:
    """Canonical variable-length ranking query.

    Padding is never serialised.  Batching code constructs ``candidate_mask`` from
    the real candidate counts, so an empty candidate can never be mistaken for a
    genuine example.
    """

    group_id: str
    split: str
    domain: str
    question: str
    candidates: tuple[Candidate, ...]
    data_fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not self.group_id.strip():
            raise ValueError("group_id must be non-empty")
        if not self.split.strip() or not self.domain.strip():
            raise ValueError("split and domain must be non-empty")
        if not self.question.strip():
            raise ValueError("question must be non-empty")
        if not candidates:
            raise ValueError("a RankingGroup needs at least one candidate")
        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate candidate_id in group {self.group_id}")
        if not self.data_fingerprint.strip():
            raise ValueError("data_fingerprint must be non-empty")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata))

    @property
    def candidate_mask(self) -> tuple[bool, ...]:
        return tuple(True for _ in self.candidates)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(candidate.score for candidate in self.candidates)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RankingGroup":
        return cls(
            group_id=str(value["group_id"]),
            split=str(value["split"]),
            domain=str(value["domain"]),
            question=str(value["question"]),
            candidates=tuple(Candidate.from_dict(item) for item in value["candidates"]),
            data_fingerprint=str(value["data_fingerprint"]),
            metadata=_plain_mapping(value.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "split": self.split,
            "domain": self.domain,
            "question": self.question,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidate_mask": list(self.candidate_mask),
            "data_fingerprint": self.data_fingerprint,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ScoreRecord:
    """Auditable candidate-level prediction emitted by every judge."""

    group_id: str
    candidate_id: str
    model_name: str
    model_revision: str
    prompt_hash: str
    data_fingerprint: str
    score: float | None
    raw_output: str
    parsing_status: str
    seed: int
    inference_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is not None and not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(f"predicted score must be in [0, 1], got {self.score}")
        if self.parsing_status not in {"ok", "parse_error", "model_error", "cached"}:
            raise ValueError(f"unsupported parsing_status: {self.parsing_status}")
        if self.inference_ms < 0:
            raise ValueError("inference_ms must be non-negative")
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreRecord":
        raw_score = value.get("score")
        return cls(
            group_id=str(value["group_id"]),
            candidate_id=str(value["candidate_id"]),
            model_name=str(value["model_name"]),
            model_revision=str(value.get("model_revision", "unknown")),
            prompt_hash=str(value.get("prompt_hash", "none")),
            data_fingerprint=str(value["data_fingerprint"]),
            score=None if raw_score is None else float(raw_score),
            raw_output=str(value.get("raw_output", "")),
            parsing_status=str(value.get("parsing_status", "ok")),
            seed=int(value.get("seed", 0)),
            inference_ms=float(value.get("inference_ms", 0.0)),
            metadata=_plain_mapping(value.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_disjoint_splits(split_groups: Mapping[str, Sequence[RankingGroup]]) -> None:
    seen: dict[str, str] = {}
    for split, groups in split_groups.items():
        for group in groups:
            previous = seen.get(group.group_id)
            if previous is not None:
                raise ValueError(
                    f"group {group.group_id} occurs in both {previous!r} and {split!r}"
                )
            seen[group.group_id] = split
