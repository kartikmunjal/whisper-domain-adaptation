"""
Evaluate base Whisper WER on domain-specific eval sets.

Run this before fine-tuning to establish baseline numbers.
The output JSON becomes the "before" in the before/after comparison.

Usage
-----
python scripts/evaluate_baseline.py \
    --eval_manifest data/medical/eval_manifest.parquet \
    --domain_vocab configs/medical_terms.txt \
    --model_id openai/whisper-small \
    --output experiments/results/medical/baseline_wer.json

python scripts/evaluate_baseline.py \
    --eval_manifest data/financial_synth/eval_manifest.parquet \
    --domain_vocab configs/financial_terms.txt \
    --model_id openai/whisper-small \
    --output experiments/results/financial/baseline_wer.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_adapt.evaluation.wer import DomainWERAnalyzer, load_domain_vocab
from whisper_adapt.evaluation.oov_analysis import OOVAnalyzer

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline Whisper WER evaluation")
    p.add_argument("--eval_manifest", required=True)
    p.add_argument("--domain_vocab", required=True,
                   help="Path to domain terms file (medical_terms.txt or financial_terms.txt)")
    p.add_argument("--model_id", default="openai/whisper-small")
    p.add_argument("--output", default=None,
                   help="Path to save JSON report (default: experiments/results/baseline_wer.json)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps (auto-detected if not specified)")
    return p.parse_args()


@torch.no_grad()
def transcribe_batch(
    audio_paths: list[str],
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
) -> list[str]:
    """Transcribe a batch of audio files and return predicted strings."""
    input_features_list = []
    for path in audio_paths:
        audio, _ = librosa.load(path, sr=16_000, mono=True)
        feats = processor.feature_extractor(
            audio, sampling_rate=16_000, return_tensors="pt"
        ).input_features
        input_features_list.append(feats)

    input_features = torch.cat(input_features_list, dim=0).to(device)

    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        ),
    )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Load model
    logger.info("Loading %s", args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id).to(device)
    model.eval()
    processor = WhisperProcessor.from_pretrained(args.model_id)

    # Load eval data
    df = pd.read_parquet(args.eval_manifest)
    logger.info("Evaluating on %d samples", len(df))

    # Transcribe in batches
    all_hypotheses = []
    for i in tqdm(range(0, len(df), args.batch_size), desc="Transcribing"):
        batch_df = df.iloc[i:i + args.batch_size]
        paths = batch_df["path"].tolist()
        try:
            hyps = transcribe_batch(paths, model, processor, device)
            all_hypotheses.extend(hyps)
        except Exception as e:
            logger.warning("Batch %d failed: %s — padding with empty strings", i, e)
            all_hypotheses.extend([""] * len(paths))

    references = df["sentence"].tolist()
    assert len(references) == len(all_hypotheses)

    # Domain WER breakdown
    domain_vocab = load_domain_vocab(args.domain_vocab)
    analyzer = DomainWERAnalyzer(domain_vocab)
    report = analyzer.analyze(references, all_hypotheses)

    # OOV analysis
    oov_analyzer = OOVAnalyzer(list(domain_vocab))
    oov_report = oov_analyzer.analyze(references, all_hypotheses)

    result = {
        "model": args.model_id,
        "eval_manifest": args.eval_manifest,
        "n_samples": len(references),
        "wer": {
            "overall": report.wer_overall,
            "domain_terms": report.wer_domain_terms,
            "common_terms": report.wer_common_terms,
            "n_domain_utterances": report.n_domain,
            "n_common_utterances": report.n_common,
        },
        "oov": {
            "worst_terms": oov_report.worst_terms,
            "best_terms": oov_report.best_terms,
            "per_term_summary": oov_report.summary[
                ["term", "n_occurrences", "term_recall", "wer"]
            ].to_dict(orient="records"),
        },
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("Baseline WER results (%s)", args.model_id)
    logger.info("  Overall WER:       %.1f%%", report.wer_overall * 100)
    logger.info("  Domain terms WER:  %.1f%%", report.wer_domain_terms * 100)
    logger.info("  Common terms WER:  %.1f%%", report.wer_common_terms * 100)
    logger.info("  Hardest terms: %s", oov_report.worst_terms[:3])
    logger.info("=" * 60)

    # Save
    output_path = args.output or "experiments/results/baseline_wer.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
