#!/usr/bin/env python3
"""Materialize Corti med-dictate as a hashed evaluation-only manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.evaluation.wer import normalize_text
from whisper_adapt.reproducibility import sha256_file

DATASET_ID = "corti/med-dictate"
DATASET_CONFIGS = ("en", "de", "fr")
LICENSE = "Corti ASR Evaluation Dataset Licence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/med_dictate_eval")
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for language in DATASET_CONFIGS:
        dataset = load_dataset(
            DATASET_ID, language, split="test", revision=args.revision
        )
        dataset = dataset.cast_column("audio", Audio(decode=False))
        for index, sample in enumerate(dataset):
            audio = sample["audio"]
            reference = sample["transcription"]
            sample_id = hashlib.sha256(
                f"{DATASET_ID}|{language}|{args.revision}|{index}".encode()
            ).hexdigest()[:16]
            wav_path = wav_dir / f"{sample_id}.wav"
            if audio.get("bytes") is not None:
                samples, sample_rate = sf.read(io.BytesIO(audio["bytes"]))
            else:
                samples, sample_rate = sf.read(audio["path"])
            sf.write(wav_path, samples, sample_rate, subtype="PCM_16")
            rows.append({
                "id": sample_id,
                "path": str(wav_path.relative_to(root)),
                "sentence": reference,
                "language": language,
                "source_index": index,
                "source_dataset": DATASET_ID,
                "source_config": language,
                "source_revision": args.revision,
                "license": LICENSE,
                "evaluation_only": True,
                "audio_sha256": sha256_file(wav_path),
                "transcript_sha256": hashlib.sha256(
                    normalize_text(reference).encode()
                ).hexdigest(),
            })
    if len(rows) != 40:
        raise RuntimeError(f"Expected 40 evaluation rows, found {len(rows)}")
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "eval_manifest.parquet", index=False)
    frame[frame.language == "en"].reset_index(drop=True).to_parquet(
        output / "eval_en_manifest.parquet", index=False
    )
    report = {
        "schema_version": 1,
        "source_dataset": DATASET_ID,
        "source_configs": list(DATASET_CONFIGS),
        "source_revision": args.revision,
        "license": LICENSE,
        "n_eval": len(frame),
        "language_counts": frame.groupby("language").size().to_dict(),
        "evaluation_only": True,
        "prohibited_uses": [
            "codec training",
            "ASR fine-tuning",
            "checkpoint selection",
            "normalization tuning",
        ],
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
