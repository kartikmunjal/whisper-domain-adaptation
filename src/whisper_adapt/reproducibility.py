"""Determinism, hashing, and run-provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def collect_provenance(
    *,
    repo_root: str | Path,
    arguments: dict[str, Any],
    input_files: list[str | Path],
    seed: int | None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "jiwer", "librosa",
                 "numpy", "pandas"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(repo, "status", "--porcelain")),
        "arguments": arguments,
        "seed": seed,
        "inputs": {
            str(Path(p)): sha256_file(p) for p in input_files if Path(p).is_file()
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "packages": packages,
        },
    }


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
