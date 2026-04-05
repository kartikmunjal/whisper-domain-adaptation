"""
Medical speech dataset loading and quality filtering.

The HuggingFace dataset `speech-recognition/medical-speech-transcription` contains
~8,000 de-identified medical dictation clips covering clinical notes, discharge
summaries, and radiology reports. It's exactly the kind of content that breaks
base Whisper — dense with polysyllabic Latin-derived terms that rarely appear in
general-purpose training corpora.

Quality filtering mirrors the pipeline in Audio-Data-Creation:
  - SNR >= 15 dB  (same threshold as Common Voice curation)
  - Duration 0.5 – 30.0 s
  - Silence ratio < 0.40
  - No clipping (< 0.1% samples at |amp| >= 0.99)

This keeps ~80% of clips, consistent with Common Voice validation-split pass rates.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

TARGET_SR = 16_000  # Whisper expects 16 kHz mono


# ── Quality thresholds (shared conventions with Audio-Data-Creation) ────────

@dataclass
class QualityThresholds:
    min_duration_sec: float = 0.5
    max_duration_sec: float = 30.0
    min_snr_db: float = 15.0
    max_silence_ratio: float = 0.40
    max_clipping_ratio: float = 0.001  # 0.1%
    min_rms_db: float = -40.0


@dataclass
class QualityReport:
    passes: bool
    duration_sec: float = 0.0
    snr_db: float = 0.0
    silence_ratio: float = 0.0
    clipping_ratio: float = 0.0
    rms_db: float = 0.0
    fail_reasons: list[str] = field(default_factory=list)


# ── Reference-free SNR estimation ────────────────────────────────────────────

def estimate_snr(audio: np.ndarray, sr: int, frame_ms: float = 25.0) -> float:
    """
    Reference-free SNR estimate.

    Strategy: treat the lowest-energy decile of frames as noise floor and
    the 90th-percentile frame as signal. Works without a separate noise
    reference, which medical dictation recordings never have.

    Returns np.inf for near-silence inputs (nothing to measure).
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len // 2

    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=0)) + 1e-10

    noise_rms = np.percentile(rms_per_frame, 10)
    signal_rms = np.percentile(rms_per_frame, 90)

    if noise_rms < 1e-9:
        return np.inf

    return float(20 * np.log10(signal_rms / noise_rms))


def silence_ratio(audio: np.ndarray, sr: int, frame_ms: float = 25.0,
                  top_db: float = 30.0) -> float:
    """Fraction of frames that are more than top_db below the peak RMS."""
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len // 2

    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=0)) + 1e-10
    rms_db = 20 * np.log10(rms_per_frame / (rms_per_frame.max() + 1e-10))

    return float(np.mean(rms_db < -top_db))


def check_quality(audio: np.ndarray, sr: int,
                  thresholds: QualityThresholds) -> QualityReport:
    """Run all quality checks on a single audio array."""
    duration = len(audio) / sr
    rms = float(20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10))
    snr = estimate_snr(audio, sr)
    sil = silence_ratio(audio, sr)
    clip = float(np.mean(np.abs(audio) >= 0.99))

    fails: list[str] = []
    if duration < thresholds.min_duration_sec:
        fails.append(f"too_short ({duration:.2f}s)")
    if duration > thresholds.max_duration_sec:
        fails.append(f"too_long ({duration:.1f}s)")
    if snr < thresholds.min_snr_db:
        fails.append(f"low_snr ({snr:.1f} dB)")
    if sil > thresholds.max_silence_ratio:
        fails.append(f"too_silent ({sil:.2f})")
    if clip > thresholds.max_clipping_ratio:
        fails.append(f"clipping ({clip:.4f})")
    if rms < thresholds.min_rms_db:
        fails.append(f"quiet ({rms:.1f} dBFS)")

    return QualityReport(
        passes=len(fails) == 0,
        duration_sec=duration,
        snr_db=float(snr),
        silence_ratio=sil,
        clipping_ratio=clip,
        rms_db=rms,
        fail_reasons=fails,
    )


# ── Dataset class ─────────────────────────────────────────────────────────────

class MedicalSpeechDataset:
    """
    Wrapper around the HuggingFace medical-speech-transcription dataset.

    After loading, applies quality filtering and returns a pandas DataFrame
    manifest with columns:
      id, path, sentence, duration_sec, snr_db, silence_ratio, source

    The 'source' column is always 'real' here; it's used when mixing with
    synthesized financial samples.
    """

    HF_DATASET_ID = "speech-recognition/medical-speech-transcription"

    def __init__(
        self,
        output_dir: str | Path,
        thresholds: Optional[QualityThresholds] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or QualityThresholds()
        self.max_samples = max_samples
        self.seed = seed

    def load_and_filter(self, split: str = "train") -> pd.DataFrame:
        """
        Download the dataset, resample to 16 kHz, run quality checks,
        and write filtered WAV files to output_dir.

        Returns a manifest DataFrame ready for feature extraction.
        """
        logger.info("Loading %s split from %s", split, self.HF_DATASET_ID)
        ds = load_dataset(self.HF_DATASET_ID, split=split, trust_remote_code=True)

        if self.max_samples and len(ds) > self.max_samples:
            ds = ds.shuffle(seed=self.seed).select(range(self.max_samples))

        records = []
        n_rejected = 0
        reject_reasons: dict[str, int] = {}

        for sample in tqdm(ds, desc="Quality filtering"):
            try:
                audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
                orig_sr = sample["audio"]["sampling_rate"]

                # Resample to 16 kHz if needed
                if orig_sr != TARGET_SR:
                    audio_array = librosa.resample(
                        audio_array, orig_sr=orig_sr, target_sr=TARGET_SR
                    )

                report = check_quality(audio_array, TARGET_SR, self.thresholds)

                if not report.passes:
                    n_rejected += 1
                    for r in report.fail_reasons:
                        key = r.split("(")[0].strip()
                        reject_reasons[key] = reject_reasons.get(key, 0) + 1
                    continue

                # Write to disk and record
                sample_id = hashlib.md5(
                    sample["sentence"].encode()
                ).hexdigest()[:12]
                wav_path = self.output_dir / f"{sample_id}.wav"

                import soundfile as sf
                sf.write(str(wav_path), audio_array, TARGET_SR)

                records.append({
                    "id": sample_id,
                    "path": str(wav_path),
                    "sentence": sample["sentence"],
                    "duration_sec": report.duration_sec,
                    "snr_db": report.snr_db,
                    "silence_ratio": report.silence_ratio,
                    "source": "real",
                })

            except Exception as exc:
                logger.warning("Skipping sample: %s", exc)
                n_rejected += 1

        df = pd.DataFrame(records)
        logger.info(
            "Quality filtering: %d kept, %d rejected (%.1f%% pass rate)",
            len(df),
            n_rejected,
            100 * len(df) / max(1, len(df) + n_rejected),
        )
        if reject_reasons:
            logger.info("Rejection breakdown: %s", reject_reasons)

        return df


def load_medical_dataset(
    output_dir: str | Path,
    split: str = "train",
    max_samples: Optional[int] = None,
    thresholds: Optional[QualityThresholds] = None,
) -> pd.DataFrame:
    """Convenience wrapper around MedicalSpeechDataset."""
    loader = MedicalSpeechDataset(
        output_dir=output_dir,
        thresholds=thresholds,
        max_samples=max_samples,
    )
    return loader.load_and_filter(split=split)
