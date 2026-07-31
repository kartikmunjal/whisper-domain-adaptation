from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import numpy as np


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


def test_log_mel_distance_is_zero_for_identical_audio() -> None:
    time = np.arange(3200) / 16_000
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    assert MODULE.log_mel_l1_db(audio, audio.copy(), 16_000) == pytest.approx(0.0)


def test_log_mel_distance_detects_spectral_change() -> None:
    time = np.arange(3200) / 16_000
    low = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    high = np.sin(2 * np.pi * 1760 * time).astype(np.float32)
    assert MODULE.log_mel_l1_db(low, high, 16_000) > 1.0
