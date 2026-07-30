from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reconstruct_codec_eval.py"
SPEC = importlib.util.spec_from_file_location("reconstruct_codec_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_vq_empirical_entropy_flattens_batch_and_time() -> None:
    indices = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])

    assert MODULE.empirical_entropy(indices) == pytest.approx(1.0)


def test_fsq_empirical_entropy_counts_frame_vectors() -> None:
    indices = torch.tensor(
        [
            [[0, 0], [0, 0], [1, 1], [1, 1]],
            [[0, 0], [0, 0], [1, 1], [1, 1]],
        ]
    )

    assert MODULE.empirical_entropy(indices) == pytest.approx(1.0)
