#!/usr/bin/env python3
"""Generate leakage-safe medical adapter train/validation speech."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import librosa
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.data.financial import ALL_VOICES, TARGET_SR, SynthesisConfig, _synthesize_one
from whisper_adapt.data.medical import QualityThresholds, check_quality
from whisper_adapt.evaluation.wer import load_domain_vocab, normalize_text
from whisper_adapt.reproducibility import sha256_file

TEMPLATES = (
    "The clinical note documents {term}.",
    "The patient was evaluated for {term}.",
    "The differential diagnosis includes {term}.",
    "The physician discussed treatment for {term}.",
    "The medical record contains the term {term}.",
    "Follow-up was recommended because of {term}.",
    "The assessment mentions {term}.",
    "The care team reviewed findings related to {term}.",
)
SPLIT_VOICES = {"train": ALL_VOICES[:8], "validation": ALL_VOICES[8:10]}
SPLIT_TEMPLATE_IDS = {"train": tuple(range(6)), "validation": (6, 7)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/medical_research")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    terms = sorted(load_domain_vocab(root / "configs" / "medical_terms.txt"))
    planned = []
    for split, template_ids in SPLIT_TEMPLATE_IDS.items():
        voices = SPLIT_VOICES[split]
        counter = 0
        for template_id in template_ids:
            for term in terms:
                planned.append({
                    "split": split,
                    "sentence": TEMPLATES[template_id].format(term=term),
                    "term": term,
                    "is_domain": True,
                    "template_family": f"medical_{template_id:02d}",
                    "voice": voices[counter % len(voices)],
                })
                counter += 1
    if args.dry_run:
        frame = pd.DataFrame(planned)
        print(json.dumps({
            "n_planned": len(frame),
            "split_counts": frame.groupby("split").size().to_dict(),
        }, indent=2))
        return

    cfg = SynthesisConfig()
    thresholds = QualityThresholds(
        min_snr_db=cfg.min_snr_db,
        max_silence_ratio=cfg.max_silence_ratio,
        min_duration_sec=cfg.min_duration_sec,
        max_duration_sec=cfg.max_duration_sec,
    )
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    failures = []
    for row in planned:
        identity = f"{row['sentence']}|{row['voice']}"
        sample_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        wav_path = wav_dir / f"{sample_id}.wav"
        if not wav_path.exists() and not asyncio.run(
            _synthesize_one(row["sentence"], row["voice"], wav_path)
        ):
            failures.append({"id": sample_id, "reason": "synthesis_failed"})
            continue
        audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        quality = check_quality(audio, TARGET_SR, thresholds)
        if not quality.passes:
            failures.append({"id": sample_id, "reason": quality.fail_reasons})
            continue
        accepted.append({
            **row,
            "id": sample_id,
            "path": str(wav_path.relative_to(root)),
            "duration_sec": quality.duration_sec,
            "snr_db": quality.snr_db,
            "silence_ratio": quality.silence_ratio,
            "source": "edge-tts",
            "audio_sha256": sha256_file(wav_path),
            "transcript_sha256": hashlib.sha256(
                normalize_text(row["sentence"]).encode()
            ).hexdigest(),
        })
    if failures:
        raise RuntimeError(f"Refusing partial medical dataset: {failures[:5]}")
    frame = pd.DataFrame(accepted)
    if set(frame[frame.split == "train"].voice) & set(
        frame[frame.split == "validation"].voice
    ):
        raise RuntimeError("Voice leakage between medical train and validation")
    if set(frame[frame.split == "train"].template_family) & set(
        frame[frame.split == "validation"].template_family
    ):
        raise RuntimeError("Template leakage between medical train and validation")
    for split, part in frame.groupby("split"):
        part.reset_index(drop=True).to_parquet(
            output / f"{split}_manifest.parquet", index=False
        )
    report = {
        "schema_version": 1,
        "generator": "scripts/prepare_medical_research_data.py",
        "n_samples": len(frame),
        "split_counts": frame.groupby("split").size().to_dict(),
        "voice_assignments": SPLIT_VOICES,
        "template_assignments": SPLIT_TEMPLATE_IDS,
        "training_data_warning": (
            "Adapter training and validation are synthetic Edge-TTS. "
            "Confirmatory evaluation is a separate real-audio dataset."
        ),
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
