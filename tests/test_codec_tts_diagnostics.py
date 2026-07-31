from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_codec_tts_conditioning.py"
SPEC = importlib.util.spec_from_file_location("analyze_codec_tts_conditioning", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_edit_alignment_identifies_operations() -> None:
    operations = [op for op, _ in MODULE.edit_alignment([1, 2, 3], [1, 4, 3, 5])]
    assert operations.count("substitute") == 1
    assert operations.count("insert") == 1
    assert operations.count("equal") == 2


def test_token_edit_rate_is_normalized() -> None:
    assert MODULE.token_edit_rate([1, 2], [1, 2]) == 0.0
    assert MODULE.token_edit_rate([1, 2], [1, 3]) == pytest.approx(0.5)


def test_position_errors_cover_all_alignment_events() -> None:
    errors, totals = MODULE.normalized_position_errors([1, 2, 3], [1, 4, 3, 5])
    assert errors.sum() == 2
    assert totals.sum() == 4
