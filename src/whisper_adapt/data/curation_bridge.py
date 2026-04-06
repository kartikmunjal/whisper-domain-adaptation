"""
Bridge between Audio-Data-Creation and whisper-domain-adaptation.

The two projects share an identical parquet manifest schema:
  id, path, sentence, duration_sec, snr_db, silence_ratio, source

This is intentional — Audio-Data-Creation's filtered_manifest.parquet can be
fed directly into Whisper fine-tuning without any format conversion. The bridge
adds one thing the curation pipeline doesn't provide: domain term filtering,
which splits the manifest into domain-heavy vs. general utterances so we can
track the right WER metrics.

The feedback loop this enables
--------------------------------
  1. Audio-Data-Creation curates a raw corpus:
       python scripts/run_pipeline.py → data/filtered_manifest.parquet

  2. This bridge ingests that manifest:
       python scripts/import_from_curation.py \
           --manifest ../Audio-Data-Creation/outputs/filtered_manifest.parquet \
           --domain_vocab configs/medical_terms.txt \
           --output_dir data/medical

  3. Fine-tune Whisper on the curated data:
       python scripts/run_finetune.py --config configs/medical_finetune.yaml ...

  4. Export the fine-tuned model back to Audio-Data-Creation's evaluator:
       The evaluator.py in Audio-Data-Creation accepts fine_tuned_model_path,
       which routes transcription through the domain-adapted model instead of
       base Whisper — giving accurate WER numbers for medical/financial speech.

  5. Audio-Data-Creation can now use the fine-tuned model to evaluate the
     quality of *new* curation runs on domain data, closing the loop:
       python scripts/evaluate_with_domain_model.py \
           --manifest outputs/filtered_manifest.parquet \
           --model_path ../whisper-domain-adaptation/checkpoints/medical/adapter
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Schema constants (shared with Audio-Data-Creation) ────────────────────────

REQUIRED_COLS = {"id", "path", "sentence"}
OPTIONAL_COLS = {"duration_sec", "snr_db", "silence_ratio", "source",
                 "gender", "age", "accent", "speaker_id"}


def load_from_curation(
    manifest_path: str | Path,
    domain_vocab_file: Optional[str | Path] = None,
    min_snr_db: Optional[float] = None,
    max_duration_sec: Optional[float] = None,
    source_filter: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load an Audio-Data-Creation manifest and split into domain / general subsets.

    The split is used downstream to track domain-specific WER separately from
    general speech WER. Both subsets are valid training data; the split only
    matters for evaluation.

    Parameters
    ----------
    manifest_path    : path to filtered_manifest.parquet from Audio-Data-Creation
    domain_vocab_file: optional path to domain terms file (e.g. configs/medical_terms.txt)
                       If None, returns (full_df, empty_df) — no split
    min_snr_db       : optional secondary SNR gate (AC already filtered, but you
                       might want a stricter threshold for fine-tuning)
    max_duration_sec : optional duration cap (AC allows up to 30s; Whisper prefers ≤25s)
    source_filter    : "real", "synthetic", or None for both

    Returns
    -------
    (domain_df, general_df) — domain contains utterances with any domain term;
                               general contains the rest
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_parquet(path)
    _validate_schema(df, path)

    logger.info("Loaded %d samples from %s", len(df), path)

    # Optional secondary filters
    if min_snr_db is not None and "snr_db" in df.columns:
        before = len(df)
        df = df[df["snr_db"] >= min_snr_db]
        logger.info("SNR gate (>= %.1f dB): %d → %d", min_snr_db, before, len(df))

    if max_duration_sec is not None and "duration_sec" in df.columns:
        before = len(df)
        df = df[df["duration_sec"] <= max_duration_sec]
        logger.info("Duration gate (<= %.1fs): %d → %d", max_duration_sec, before, len(df))

    if source_filter is not None and "source" in df.columns:
        before = len(df)
        df = df[df["source"] == source_filter]
        logger.info("Source filter ('%s'): %d → %d", source_filter, before, len(df))

    if domain_vocab_file is None:
        return df.reset_index(drop=True), pd.DataFrame(columns=df.columns)

    # Domain/general split
    vocab = _load_vocab(domain_vocab_file)
    is_domain = df["sentence"].apply(
        lambda s: _contains_any(str(s), vocab)
    )

    domain_df = df[is_domain].reset_index(drop=True)
    general_df = df[~is_domain].reset_index(drop=True)

    logger.info(
        "Domain split: %d domain utterances / %d general (%.1f%% domain)",
        len(domain_df), len(general_df),
        100 * len(domain_df) / max(1, len(df)),
    )

    return domain_df, general_df


def combine_for_training(
    domain_df: pd.DataFrame,
    general_df: pd.DataFrame,
    domain_oversample: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Combine domain and general splits into a single training manifest.

    Domain samples are oversampled (default 2×) to increase domain term
    exposure during fine-tuning, which is especially important when domain
    utterances are a minority of the corpus.

    Parameters
    ----------
    domain_df        : domain subset from load_from_curation()
    general_df       : general subset
    domain_oversample: how many times to repeat domain samples relative to
                       their natural proportion (1.0 = no oversampling)
    seed             : reproducibility for random sampling

    Returns
    -------
    Combined DataFrame, shuffled, ready for run_finetune.py
    """
    if len(domain_df) == 0:
        return general_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if domain_oversample > 1.0:
        n_extra = int(len(domain_df) * (domain_oversample - 1.0))
        extra = domain_df.sample(n=n_extra, replace=True, random_state=seed)
        domain_df = pd.concat([domain_df, extra], ignore_index=True)
        logger.info("Oversampled domain set: %d → %d samples (%.1f×)",
                    len(domain_df) - n_extra, len(domain_df), domain_oversample)

    combined = pd.concat([domain_df, general_df], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def manifest_stats(df: pd.DataFrame) -> dict:
    """
    Return the same statistics summary that Audio-Data-Creation's
    curation_report.json uses — so reports are directly comparable.
    """
    stats: dict = {"n_samples": len(df)}

    if "duration_sec" in df.columns:
        stats["mean_duration_sec"] = round(float(df["duration_sec"].mean()), 2)
        stats["total_hours"] = round(float(df["duration_sec"].sum() / 3600), 2)

    if "snr_db" in df.columns:
        stats["mean_snr_db"] = round(float(df["snr_db"].mean()), 2)

    if "silence_ratio" in df.columns:
        stats["mean_silence_ratio"] = round(float(df["silence_ratio"].mean()), 3)

    if "source" in df.columns:
        stats["source_counts"] = df["source"].value_counts().to_dict()

    for col in ("gender", "accent", "age"):
        if col in df.columns:
            stats[f"{col}_dist"] = df[col].value_counts(normalize=True).round(3).to_dict()

    return stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_schema(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest {path} is missing required columns: {missing}. "
            "Expected Audio-Data-Creation filtered_manifest.parquet format."
        )


def _load_vocab(vocab_file: str | Path) -> set[str]:
    with open(vocab_file) as f:
        terms = set()
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.add(line.lower())
    return terms


def _contains_any(text: str, vocab: set[str]) -> bool:
    text_lower = text.lower()
    return any(term in text_lower for term in vocab)
