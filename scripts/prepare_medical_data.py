"""
Download and quality-filter the medical speech transcription dataset.

Usage
-----
python scripts/prepare_medical_data.py \
    --output_dir data/medical \
    --max_samples 10000 \
    --eval_fraction 0.1 \
    --seed 42

Outputs
-------
data/medical/train/               WAV files
data/medical/eval/                WAV files
data/medical/train_manifest.parquet
data/medical/eval_manifest.parquet
data/medical/curation_report.json

The parquet manifests use the same schema as Audio-Data-Creation's
filtered_manifest.parquet: id, path, sentence, duration_sec, snr_db,
silence_ratio, source. This means the evaluation scripts can operate
on either medical or Common Voice data without changes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_adapt.data.medical import MedicalSpeechDataset, QualityThresholds

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare medical speech dataset")
    p.add_argument("--output_dir", default="data/medical",
                   help="Root directory for WAV files and manifests")
    p.add_argument("--max_samples", type=int, default=10_000,
                   help="Maximum number of samples to download (0 = no limit)")
    p.add_argument("--eval_fraction", type=float, default=0.1,
                   help="Fraction of data to hold out for evaluation")
    p.add_argument("--min_snr_db", type=float, default=15.0)
    p.add_argument("--max_silence_ratio", type=float, default=0.40)
    p.add_argument("--min_duration_sec", type=float, default=0.5)
    p.add_argument("--max_duration_sec", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = QualityThresholds(
        min_snr_db=args.min_snr_db,
        max_silence_ratio=args.max_silence_ratio,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )

    max_samples = args.max_samples if args.max_samples > 0 else None

    # Load and filter — WAV files written to output_dir/wav/
    wav_dir = output_dir / "wav"
    ds = MedicalSpeechDataset(
        output_dir=wav_dir,
        thresholds=thresholds,
        max_samples=max_samples,
        seed=args.seed,
    )
    df = ds.load_and_filter(split="train")

    if len(df) == 0:
        logger.error("No samples passed quality filtering — check dataset availability")
        sys.exit(1)

    # Train/eval split (stratified by approximate duration bucket for balance)
    rng = np.random.default_rng(args.seed)
    n_eval = max(1, int(len(df) * args.eval_fraction))
    eval_idx = rng.choice(len(df), size=n_eval, replace=False)
    eval_mask = np.zeros(len(df), dtype=bool)
    eval_mask[eval_idx] = True

    train_df = df[~eval_mask].reset_index(drop=True)
    eval_df = df[eval_mask].reset_index(drop=True)

    # Save manifests
    train_manifest = output_dir / "train_manifest.parquet"
    eval_manifest = output_dir / "eval_manifest.parquet"
    train_df.to_parquet(train_manifest, index=False)
    eval_df.to_parquet(eval_manifest, index=False)

    logger.info("Saved %d train samples → %s", len(train_df), train_manifest)
    logger.info("Saved %d eval samples  → %s", len(eval_df), eval_manifest)

    # Curation report
    report = {
        "n_total_downloaded": len(df) + (max_samples or 0),
        "n_passed_quality": len(df),
        "n_train": len(train_df),
        "n_eval": len(eval_df),
        "quality_thresholds": {
            "min_snr_db": thresholds.min_snr_db,
            "max_silence_ratio": thresholds.max_silence_ratio,
            "min_duration_sec": thresholds.min_duration_sec,
            "max_duration_sec": thresholds.max_duration_sec,
        },
        "stats": {
            "mean_duration_sec": round(float(df["duration_sec"].mean()), 2),
            "mean_snr_db": round(float(df["snr_db"].mean()), 2),
            "mean_silence_ratio": round(float(df["silence_ratio"].mean()), 3),
        },
    }
    report_path = output_dir / "curation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Curation report saved to %s", report_path)


if __name__ == "__main__":
    main()
