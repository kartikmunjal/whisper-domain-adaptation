"""
Import an Audio-Data-Creation manifest and prepare it for Whisper fine-tuning.

This is the entry point for the forward direction of the feedback loop:

    Audio-Data-Creation curation run
        → outputs/filtered_manifest.parquet
            → this script
                → data/{domain}/train_manifest.parquet
                → data/{domain}/eval_manifest.parquet
                    → run_finetune.py

The manifest schema is shared between both projects (id, path, sentence,
duration_sec, snr_db, silence_ratio, source), so no format conversion is needed —
only domain filtering, optional oversampling, and train/eval splitting.

Usage
-----
python scripts/import_from_curation.py \
    --manifest ../Audio-Data-Creation/outputs/filtered_manifest.parquet \
    --domain_vocab configs/medical_terms.txt \
    --output_dir data/medical_from_curation \
    --eval_fraction 0.1 \
    --domain_oversample 2.0

python scripts/import_from_curation.py \
    --manifest ../Audio-Data-Creation/outputs/filtered_manifest.parquet \
    --domain_vocab configs/financial_terms.txt \
    --output_dir data/financial_from_curation \
    --source_filter synthetic     # import only TTS-synthesized samples

After running, use the manifests with:
    python scripts/run_finetune.py \
        --config configs/medical_finetune.yaml \
        --train_manifest data/medical_from_curation/train_manifest.parquet \
        --eval_manifest data/medical_from_curation/eval_manifest.parquet
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_adapt.data.curation_bridge import (
    load_from_curation,
    combine_for_training,
    manifest_stats,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Import Audio-Data-Creation manifest for Whisper fine-tuning"
    )
    p.add_argument("--manifest", required=True,
                   help="Path to filtered_manifest.parquet from Audio-Data-Creation")
    p.add_argument("--domain_vocab", default=None,
                   help="Domain vocabulary file (e.g. configs/medical_terms.txt). "
                        "If omitted, all samples are treated as training data.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write train/eval manifests and import report")
    p.add_argument("--eval_fraction", type=float, default=0.1,
                   help="Fraction to hold out for evaluation (default: 0.1)")
    p.add_argument("--domain_oversample", type=float, default=2.0,
                   help="How many times to repeat domain samples relative to natural "
                        "proportion. 1.0 = no oversampling, 2.0 = 2× (default).")
    p.add_argument("--min_snr_db", type=float, default=None,
                   help="Optional secondary SNR gate (Audio-Data-Creation already "
                        "filtered at 15 dB; use this for a stricter threshold)")
    p.add_argument("--max_duration_sec", type=float, default=25.0,
                   help="Max clip duration for Whisper (default 25s; AC allows 30s)")
    p.add_argument("--source_filter", choices=["real", "synthetic"], default=None,
                   help="Import only 'real' or 'synthetic' samples")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and split
    domain_df, general_df = load_from_curation(
        manifest_path=args.manifest,
        domain_vocab_file=args.domain_vocab,
        min_snr_db=args.min_snr_db,
        max_duration_sec=args.max_duration_sec,
        source_filter=args.source_filter,
    )

    # Combine with oversampling for training
    train_pool = combine_for_training(
        domain_df=domain_df,
        general_df=general_df,
        domain_oversample=args.domain_oversample,
        seed=args.seed,
    )

    if len(train_pool) == 0:
        logger.error("No samples after filtering — check manifest path and filters.")
        sys.exit(1)

    # Train / eval split — eval taken from domain samples only if possible
    # (we want to evaluate on domain-heavy utterances)
    rng = np.random.default_rng(args.seed)
    if len(domain_df) >= 20:
        n_eval = max(10, int(len(domain_df) * args.eval_fraction))
        eval_idx = rng.choice(len(domain_df), size=n_eval, replace=False)
        eval_df = domain_df.iloc[eval_idx].reset_index(drop=True)
        eval_ids = set(eval_df["id"])
        train_df = train_pool[~train_pool["id"].isin(eval_ids)].reset_index(drop=True)
    else:
        # Not enough domain samples — split the full pool
        n_eval = max(1, int(len(train_pool) * args.eval_fraction))
        eval_idx = rng.choice(len(train_pool), size=n_eval, replace=False)
        eval_mask = np.zeros(len(train_pool), dtype=bool)
        eval_mask[eval_idx] = True
        eval_df = train_pool[eval_mask].reset_index(drop=True)
        train_df = train_pool[~eval_mask].reset_index(drop=True)

    # Save
    train_manifest = output_dir / "train_manifest.parquet"
    eval_manifest = output_dir / "eval_manifest.parquet"
    train_df.to_parquet(train_manifest, index=False)
    eval_df.to_parquet(eval_manifest, index=False)

    logger.info("Train manifest: %d samples → %s", len(train_df), train_manifest)
    logger.info("Eval manifest:  %d samples → %s", len(eval_df), eval_manifest)

    # Import report (compatible with Audio-Data-Creation curation_report.json format)
    report = {
        "source_manifest": str(args.manifest),
        "domain_vocab": args.domain_vocab,
        "filters_applied": {
            "min_snr_db": args.min_snr_db,
            "max_duration_sec": args.max_duration_sec,
            "source_filter": args.source_filter,
        },
        "domain_oversample": args.domain_oversample,
        "splits": {
            "n_domain_utterances": len(domain_df),
            "n_general_utterances": len(general_df),
            "n_train": len(train_df),
            "n_eval": len(eval_df),
        },
        "train_stats": manifest_stats(train_df),
        "eval_stats": manifest_stats(eval_df),
    }
    report_path = output_dir / "import_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Import report: %s", report_path)

    logger.info("")
    logger.info("Next steps:")
    logger.info("  python scripts/run_finetune.py \\")
    logger.info("      --config configs/medical_finetune.yaml \\")
    logger.info("      --train_manifest %s \\", train_manifest)
    logger.info("      --eval_manifest %s", eval_manifest)


if __name__ == "__main__":
    main()
