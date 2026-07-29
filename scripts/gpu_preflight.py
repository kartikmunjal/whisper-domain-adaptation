#!/usr/bin/env python3
"""Fail fast when a confirmatory run is not using a suitable CUDA device."""

from __future__ import annotations

import json
import platform
import sys

import torch


def main() -> None:
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update({
            "device": props.name,
            "vram_gib": round(props.total_memory / (1024 ** 3), 2),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        })
    print(json.dumps(report, indent=2))
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Locked GPU environment requires Python 3.11")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to start confirmatory trials")
    if torch.cuda.get_device_properties(0).total_memory < 7 * 1024 ** 3:
        raise SystemExit("At least 7 GiB VRAM is required by the locked configuration")


if __name__ == "__main__":
    main()
