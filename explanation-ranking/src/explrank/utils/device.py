"""Device selection for training and evaluation scripts."""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device string to :class:`torch.device`.

    * ``auto`` — cuda if available, else mps, else cpu
    * ``cuda`` / ``cpu`` / ``mps`` — used as-is when supported
    """
    spec = spec.lower().strip()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available on this system")
        return torch.device("mps")
    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({spec}) but not available")
        return torch.device(spec)
    return torch.device(spec)
