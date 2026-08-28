from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TypeVar

T = TypeVar("T")


def canonical_json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: Any, length: int | None = None) -> str:
    payload = "\x1f".join(canonical_json(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return digest if length is None else digest[:length]


def stable_seed(*parts: Any) -> int:
    return int(stable_hash(*parts, length=16), 16)


def deterministic_uniform(low: float, high: float, *parts: Any) -> float:
    if high < low:
        raise ValueError("high must be >= low")
    integer = int(stable_hash("uniform-v1", *parts), 16)
    fraction = integer / float((1 << 256) - 1)
    return low + (high - low) * fraction


def fingerprint_records(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            yield value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json(path: str | Path, value: Any) -> None:
    _atomic_text(Path(path), json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    lines: list[str] = []
    for value in records:
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        elif is_dataclass(value):
            value = asdict(value)
        lines.append(canonical_json(value))
    _atomic_text(Path(path), "\n".join(lines) + ("\n" if lines else ""))


def parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    keys = dotted_key.split(".")
    if not all(keys):
        raise ValueError(f"invalid configuration key: {dotted_key!r}")
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override nested key below {key!r}")
        cursor = child
    cursor[keys[-1]] = value


def resolve_hf_source(
    source: str | Path,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str:
    """Resolve a cached Hub revision to a real path when offline mode is requested."""

    path = Path(source)
    if path.exists():
        return str(path)
    if not local_files_only:
        return str(source)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to resolve a cached model") from exc
    try:
        return snapshot_download(
            repo_id=str(source),
            revision=revision,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"model {source}@{revision or 'main'} is not complete in the local Hugging Face cache"
        ) from exc


def hf_dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """Use the non-deprecated Transformers dtype keyword across v4 and v5."""

    try:
        from transformers import __version__ as transformers_version

        major = int(transformers_version.split(".", 1)[0])
    except (ImportError, ValueError):
        major = 4
    return {"dtype" if major >= 5 else "torch_dtype": dtype}
