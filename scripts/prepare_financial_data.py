"""
Synthesize financial speech training data using Edge-TTS.

This script uses the same TTS voice catalog as the rlhf-and-reward-modelling-alt
project's Extension 11 (TTS RLHF pipeline). Running this before fine-tuning
creates a synthetic financial speech corpus targeting earnings-call terminology
that base Whisper handles poorly.

Why TTS for financial domain?
  Financial earnings call recordings are hard to obtain at scale with clean
  aligned transcripts. SEC filings are text-only. The few available audio
  datasets (e.g., FER dataset) have licensing constraints. TTS synthesis with
  diverse voices + quality gating is the pragmatic alternative.

  The ablation on Audio-Data-Creation showed 50% synthetic mix is optimal;
  for pure financial domain where *all* training data is synthetic, we compensate
  by using more voice diversity (all 14 voices) and more context variety
  (10 sentence templates per term) to prevent prosodic overfitting.

Usage
-----
python scripts/prepare_financial_data.py \
    --output_dir data/financial_synth \
    --eval_fraction 0.15 \
    --dry_run          # skip TTS, just show what would be generated

Outputs
-------
data/financial_synth/           WAV files
data/financial_synth/train_manifest.parquet
data/financial_synth/eval_manifest.parquet
data/financial_synth/synthesis_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_adapt.data.financial import FinancialSpeechDataset, SynthesisConfig

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize financial speech training data")
    p.add_argument("--output_dir", default="data/financial_synth")
    p.add_argument("--eval_fraction", type=float, default=0.15)
    p.add_argument("--min_snr_db", type=float, default=20.0)
    p.add_argument("--max_silence_ratio", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="Skip TTS synthesis; print what would be generated")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SynthesisConfig(
        min_snr_db=args.min_snr_db,
        max_silence_ratio=args.max_silence_ratio,
    )

    ds = FinancialSpeechDataset(output_dir=output_dir, cfg=cfg)
    df = ds.synthesize(dry_run=args.dry_run)

    if args.dry_run or len(df) == 0:
        if args.dry_run:
            logger.info("[dry_run] No files written. Use without --dry_run to synthesize.")
        else:
            logger.error("No samples synthesized — check edge-tts installation.")
            sys.exit(1)
        return

    # Train/eval split
    rng = np.random.default_rng(args.seed)
    n_eval = max(1, int(len(df) * args.eval_fraction))
    eval_idx = rng.choice(len(df), size=n_eval, replace=False)
    eval_mask = np.zeros(len(df), dtype=bool)
    eval_mask[eval_idx] = True

    train_df = df[~eval_mask].reset_index(drop=True)
    eval_df = df[eval_mask].reset_index(drop=True)

    train_df.to_parquet(output_dir / "train_manifest.parquet", index=False)
    eval_df.to_parquet(output_dir / "eval_manifest.parquet", index=False)

    logger.info("Saved %d train / %d eval samples", len(train_df), len(eval_df))

    # Voice diversity report
    voice_counts = df["voice"].value_counts().to_dict()
    unique_terms = df["term"].nunique() if "term" in df.columns else 0

    report = {
        "n_synthesized": len(df),
        "n_train": len(train_df),
        "n_eval": len(eval_df),
        "unique_terms_covered": unique_terms,
        "voice_distribution": voice_counts,
        "quality_thresholds": {
            "min_snr_db": cfg.min_snr_db,
            "max_silence_ratio": cfg.max_silence_ratio,
        },
    }
    with open(output_dir / "synthesis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", output_dir / "synthesis_report.json")


if __name__ == "__main__":
    main()
