#!/usr/bin/env python3
"""Build preregistered voice- and template-disjoint financial TTS splits."""

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

from whisper_adapt.data.financial import (  # noqa: E402
    ALL_VOICES,
    CONTEXT_TEMPLATES,
    TARGET_SR,
    SynthesisConfig,
    _synthesize_one,
)
from whisper_adapt.data.medical import QualityThresholds, check_quality  # noqa: E402
from whisper_adapt.evaluation.wer import normalize_text  # noqa: E402
from whisper_adapt.evaluation.wer import load_domain_vocab  # noqa: E402
from whisper_adapt.reproducibility import sha256_file  # noqa: E402

SPLIT_VOICES = {
    "train": ALL_VOICES[:8],
    "validation": ALL_VOICES[8:10],
    "test": ALL_VOICES[10:],
}
SPLIT_TEMPLATE_IDS = {
    "train": tuple(range(0, 6)),
    "validation": (6, 7),
    "test": (8, 9),
}
COMMON_STEMS = [
    "Thank you for joining our quarterly conference call.",
    "Management will now discuss the results for the period.",
    "Demand remained resilient across our major customer segments.",
    "We continued to invest in product quality and customer service.",
    "The team executed well despite a changing environment.",
    "We will now open the call for questions from analysts.",
    "Our strategy remains focused on disciplined long-term execution.",
    "Customer engagement improved throughout the reporting period.",
    "We appreciate the dedication of our employees around the world.",
    "The board reviewed the operating plan during its recent meeting.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/financial_research")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _records_to_generate(terms: list[str]) -> list[dict]:
    rows = []
    for split, template_ids in SPLIT_TEMPLATE_IDS.items():
        voices = SPLIT_VOICES[split]
        counter = 0
        for template_id in template_ids:
            template = CONTEXT_TEMPLATES[template_id]
            for term in terms:
                rows.append({
                    "split": split,
                    "sentence": template.format(term=term),
                    "term": term,
                    "is_domain": True,
                    "template_family": f"domain_{template_id:02d}",
                    "voice": voices[counter % len(voices)],
                })
                counter += 1
        for common_id in template_ids:
            for repeat in range(4):
                rows.append({
                    "split": split,
                    "sentence": COMMON_STEMS[common_id],
                    "term": None,
                    "is_domain": False,
                    "template_family": f"common_{common_id:02d}",
                    "voice": voices[counter % len(voices)],
                })
                counter += 1
    return rows


def _assert_disjoint(df: pd.DataFrame) -> None:
    for column in ("voice", "template_family"):
        sets = {s: set(part[column]) for s, part in df.groupby("split")}
        names = sorted(sets)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                overlap = sets[left] & sets[right]
                if overlap:
                    raise RuntimeError(f"{column} leakage {left}/{right}: {overlap}")
    for split, part in df.groupby("split"):
        if not part["is_domain"].any() or part["is_domain"].all():
            raise RuntimeError(f"{split} must contain domain and common controls")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output_dir).resolve()
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    cfg = SynthesisConfig()
    thresholds = QualityThresholds(
        min_snr_db=cfg.min_snr_db,
        max_silence_ratio=cfg.max_silence_ratio,
        min_duration_sec=cfg.min_duration_sec,
        max_duration_sec=cfg.max_duration_sec,
    )
    accepted = []
    failures = []
    planned = _records_to_generate(
        sorted(load_domain_vocab(root / "configs" / "financial_terms.txt"))
    )
    if args.dry_run:
        print(json.dumps({"n_planned": len(planned), "by_split": pd.DataFrame(planned)
                         .groupby("split").size().to_dict()}, indent=2))
        return
    for row in planned:
        identity = f"{row['sentence']}|{row['voice']}"
        sample_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        wav_path = wav_dir / f"{sample_id}.wav"
        if not wav_path.exists() and not asyncio.run(
            _synthesize_one(row["sentence"], row["voice"], wav_path)
        ):
            failures.append({"id": sample_id, "reason": "synthesis_failed"})
            continue
        try:
            audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        except Exception as exc:
            failures.append({"id": sample_id, "reason": f"load_failed:{exc}"})
            continue
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
    df = pd.DataFrame(accepted)
    if failures:
        raise RuntimeError(
            f"{len(failures)} generation/quality failures; refusing partial dataset. "
            f"First failures: {failures[:5]}"
        )
    _assert_disjoint(df)
    output.mkdir(parents=True, exist_ok=True)
    for split, part in df.groupby("split"):
        part.reset_index(drop=True).to_parquet(
            output / f"{split}_manifest.parquet", index=False
        )
    report = {
        "schema_version": 1,
        "generator": "scripts/prepare_financial_research_data.py",
        "n_samples": len(df),
        "split_counts": df.groupby("split").size().to_dict(),
        "voice_assignments": SPLIT_VOICES,
        "template_assignments": SPLIT_TEMPLATE_IDS,
        "tts_only_external_validity_warning": (
            "All audio is Edge-TTS. Results are optimistic and do not establish "
            "performance on real earnings-call audio."
        ),
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
