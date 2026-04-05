"""
Evaluate fine-tuned Whisper on domain eval sets and compare to baseline.

Usage
-----
python scripts/evaluate_finetuned.py \
    --adapter_path checkpoints/medical/adapter \
    --base_model openai/whisper-small \
    --eval_manifest data/medical/eval_manifest.parquet \
    --domain_vocab configs/medical_terms.txt \
    --baseline_report experiments/results/medical/baseline_wer.json \
    --output experiments/results/medical/finetuned_wer.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_adapt.evaluation.wer import DomainWERAnalyzer, load_domain_vocab
from whisper_adapt.evaluation.oov_analysis import OOVAnalyzer
from whisper_adapt.models.whisper_lora import load_finetuned
from evaluate_baseline import transcribe_batch

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper")
    p.add_argument("--adapter_path", required=True,
                   help="Path to saved LoRA adapter (from run_finetune.py)")
    p.add_argument("--base_model", default="openai/whisper-small")
    p.add_argument("--eval_manifest", required=True)
    p.add_argument("--domain_vocab", required=True)
    p.add_argument("--baseline_report", default=None,
                   help="Optional: baseline JSON for side-by-side comparison")
    p.add_argument("--output", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default=None)
    return p.parse_args()


def print_comparison(baseline: dict, finetuned_result: dict) -> None:
    """Print a formatted before/after comparison table."""
    bw = baseline.get("wer", {})
    fw = finetuned_result.get("wer", {})

    def pct(v):
        return f"{v*100:.1f}%" if isinstance(v, float) else "N/A"

    def delta(b, f):
        if isinstance(b, float) and isinstance(f, float):
            d = (f - b) * 100
            sign = "↓" if d < 0 else "↑"
            return f"{sign}{abs(d):.1f}pp"
        return "—"

    logger.info("=" * 70)
    logger.info("%-30s  %-12s  %-12s  %-10s", "Metric", "Baseline", "Fine-tuned", "Δ")
    logger.info("-" * 70)
    logger.info("%-30s  %-12s  %-12s  %-10s",
                "Overall WER",
                pct(bw.get("overall")), pct(fw.get("overall")),
                delta(bw.get("overall"), fw.get("overall")))
    logger.info("%-30s  %-12s  %-12s  %-10s",
                "Domain terms WER",
                pct(bw.get("domain_terms")), pct(fw.get("domain_terms")),
                delta(bw.get("domain_terms"), fw.get("domain_terms")))
    logger.info("%-30s  %-12s  %-12s  %-10s",
                "Common terms WER (regression)",
                pct(bw.get("common_terms")), pct(fw.get("common_terms")),
                delta(bw.get("common_terms"), fw.get("common_terms")))
    logger.info("=" * 70)


@torch.no_grad()
def main() -> None:
    args = parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Load fine-tuned model (LoRA merged)
    logger.info("Loading fine-tuned model from %s", args.adapter_path)
    model = load_finetuned(args.base_model, args.adapter_path).to(device)
    model.eval()
    processor = WhisperProcessor.from_pretrained(args.adapter_path)

    df = pd.read_parquet(args.eval_manifest)
    logger.info("Evaluating on %d samples", len(df))

    all_hypotheses = []
    for i in tqdm(range(0, len(df), args.batch_size), desc="Transcribing"):
        batch_df = df.iloc[i:i + args.batch_size]
        paths = batch_df["path"].tolist()
        try:
            hyps = transcribe_batch(paths, model, processor, device)
            all_hypotheses.extend(hyps)
        except Exception as e:
            logger.warning("Batch %d failed: %s", i, e)
            all_hypotheses.extend([""] * len(paths))

    references = df["sentence"].tolist()

    domain_vocab = load_domain_vocab(args.domain_vocab)
    analyzer = DomainWERAnalyzer(domain_vocab)
    report = analyzer.analyze(references, all_hypotheses)

    oov_analyzer = OOVAnalyzer(list(domain_vocab))
    oov_report = oov_analyzer.analyze(references, all_hypotheses)

    result = {
        "model": args.base_model,
        "adapter_path": args.adapter_path,
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

    # Comparison
    if args.baseline_report and Path(args.baseline_report).exists():
        with open(args.baseline_report) as f:
            baseline = json.load(f)
        print_comparison(baseline, result)
    else:
        logger.info("Overall WER:       %.1f%%", report.wer_overall * 100)
        logger.info("Domain terms WER:  %.1f%%", report.wer_domain_terms * 100)
        logger.info("Common terms WER:  %.1f%%", report.wer_common_terms * 100)

    # Save
    output_path = args.output or "experiments/results/finetuned_wer.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
