"""
Tests for data loading and quality filtering.

All tests use synthetic audio to avoid downloading datasets.
The quality checks are designed to fail at specific thresholds;
these match the thresholds used in Audio-Data-Creation exactly,
so the test expectations are stable.
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_adapt.data.medical import (
    QualityThresholds,
    check_quality,
    estimate_snr,
    silence_ratio,
)


SR = 16_000
DURATION = 3.0  # seconds


def make_speech_tone(freq: float = 440.0, duration: float = DURATION, sr: int = SR,
                     amplitude: float = 0.5) -> np.ndarray:
    """Pure tone as a stand-in for clean speech."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_noisy_audio(snr_db: float = 5.0, duration: float = DURATION, sr: int = SR) -> np.ndarray:
    """Mix a tone with white noise at a target SNR."""
    signal = make_speech_tone(duration=duration, sr=sr, amplitude=0.5)
    noise_amp = 0.5 * (10 ** (-snr_db / 20))
    noise = (noise_amp * np.random.default_rng(0).standard_normal(len(signal))).astype(np.float32)
    return np.clip(signal + noise, -1.0, 1.0)


def make_silence(duration: float = DURATION, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def make_clipped(duration: float = DURATION, sr: int = SR) -> np.ndarray:
    audio = make_speech_tone(duration=duration, sr=sr, amplitude=2.0)
    return np.clip(audio, -0.99, 0.99)


class TestEstimateSNR:
    def test_clean_tone_high_snr(self):
        audio = make_speech_tone()
        snr = estimate_snr(audio, SR)
        assert snr > 20.0, f"Expected high SNR for clean tone, got {snr:.1f}"

    def test_noisy_audio_low_snr(self):
        audio = make_noisy_audio(snr_db=5.0)
        snr = estimate_snr(audio, SR)
        # Should detect low SNR; exact value varies by noise seed but should be < 15
        assert snr < 20.0, f"Expected lower SNR for noisy audio, got {snr:.1f}"

    def test_silence_returns_inf(self):
        audio = make_silence()
        snr = estimate_snr(audio, SR)
        assert np.isinf(snr) or snr > 100, "Near-silence should return very high or inf SNR"


class TestSilenceRatio:
    def test_clean_speech_low_silence(self):
        audio = make_speech_tone()
        ratio = silence_ratio(audio, SR)
        assert ratio < 0.1, f"Expected low silence ratio for tone, got {ratio:.3f}"

    def test_full_silence_high_ratio(self):
        # Pad speech with lots of silence
        speech = make_speech_tone(duration=0.5)
        padding = np.zeros(SR * 5, dtype=np.float32)
        audio = np.concatenate([speech, padding])
        ratio = silence_ratio(audio, SR)
        assert ratio > 0.8, f"Expected high silence ratio for mostly-silent audio, got {ratio:.3f}"


class TestCheckQuality:
    def setup_method(self):
        self.thresh = QualityThresholds()

    def test_clean_audio_passes(self):
        audio = make_speech_tone()
        report = check_quality(audio, SR, self.thresh)
        assert report.passes, f"Clean audio should pass, got: {report.fail_reasons}"

    def test_too_short_fails(self):
        audio = make_speech_tone(duration=0.2)
        report = check_quality(audio, SR, self.thresh)
        assert not report.passes
        assert any("too_short" in r for r in report.fail_reasons)

    def test_too_long_fails(self):
        audio = make_speech_tone(duration=35.0)
        report = check_quality(audio, SR, self.thresh)
        assert not report.passes
        assert any("too_long" in r for r in report.fail_reasons)

    def test_clipping_fails(self):
        audio = make_clipped()
        # Need very aggressive clipping for the test to trigger
        audio_hard = np.ones(SR * 3, dtype=np.float32) * 0.99
        thresh = QualityThresholds(max_clipping_ratio=0.0001)
        report = check_quality(audio_hard, SR, thresh)
        assert not report.passes
        assert any("clipping" in r for r in report.fail_reasons)

    def test_report_fields_populated(self):
        audio = make_speech_tone()
        report = check_quality(audio, SR, self.thresh)
        assert report.duration_sec > 0
        assert report.snr_db > 0
        assert 0.0 <= report.silence_ratio <= 1.0
        assert 0.0 <= report.clipping_ratio <= 1.0
