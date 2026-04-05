"""
Ablation study: effect of training data size and synthetic mix ratio on WER.

Two ablations:
  1. Data scaling: train on 10%, 25%, 50%, 75%, 100% of medical data
     → shows how quickly domain WER improves with more data
  2. Synthetic mix (financial): 0%, 25%, 50%, 75%, 100% synthetic
     → mirrors the ablation in Audio-Data-Creation's run_ablations.py

Usage
-----
# Data scaling ablation (medical)
python scripts/run_ablations.py \
    --ablation data_scaling \
    --train_manifest data/medical/train_manifest.parquet \
    --eval_manifest data/medical/eval_manifest.parquet \
    --domain_vocab configs/medical_terms.txt \
    --output_dir experiments/results/ablations/data_scaling

# Synthetic mix ablation (financial)
python scripts/run_ablations.py \
    --ablation synthetic_mix \
    --real_manifest data/medical/train_manifest.parquet \
    --synthetic_manifest data/financial_synth/train_manifest.parquet \
    --eval_manifest data/financial_synth/eval_manifest.parquet \
    --domain_vocab configs/financial_terms.txt \
    --output_dir experiments/results/ablations/synthetic_mix
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_SCALING_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
SYNTHETIC_MIX_RATIOS = [0.0, 0.25, 0.50, 0.75, 1.0]


def run_one_experiment(
    train_manifest: str,
    eval_manifest: str,
    domain_vocab_file: str,
    output_dir: Path,
    label: str,
    seed: int = 42,
) -> dict:
    """
    Fine-tune one model configuration and return WER results.

    This function orchestrates prepare → finetune → evaluate in a single call
    so the ablation loop stays clean.
    """
    from transformers import WhisperProcessor
    from whisper_adapt.data.feature_extraction import WhisperFeatureExtractor
    from whisper_adapt.models.whisper_lora import build_whisper_lora, LoRAConfig
    from whisper_adapt.training.finetune import FinetuneConfig, run_finetune
    from whisper_adapt.evaluation.wer import DomainWERAnalyzer, load_domain_vocab

    import librosa
    from datasets import Dataset
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/whisper-small"

    # Quick training config for ablations (fewer steps)
    ft_cfg = FinetuneConfig(
        output_dir=str(output_dir / label),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        eval_steps=200,
        save_steps=200,
        warmup_steps=100,
        early_stopping_patience=3,
        report_to=["none"],
    )

    extractor = WhisperFeatureExtractor(model_id=model_id)
    processor = WhisperProcessor.from_pretrained(model_id)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

    def load_ds(path):
        df = pd.read_parquet(path)
        records = []
        for _, row in df.iterrows():
            try:
                audio, _ = librosa.load(row["path"], sr=16_000, mono=True)
                p = extractor(audio, row["sentence"])
                records.append({
                    "input_features": p["input_features"].numpy().tolist(),
                    "labels": p["labels"].numpy().tolist(),
                })
            except Exception:
                pass
        return Dataset.from_list(records)

    lora_cfg = LoRAConfig(r=16)  # smaller rank for faster ablation runs
    model = build_whisper_lora(model_id=model_id, lora_cfg=lora_cfg)
    train_ds = load_ds(train_manifest)
    eval_ds = load_ds(eval_manifest)

    run_finetune(model, processor, train_ds, eval_ds, ft_cfg)

    # Evaluate
    from whisper_adapt.models.whisper_lora import load_finetuned
    finetuned = load_finetuned(model_id, str(output_dir / label / "adapter")).to(device)
    finetuned.eval()

    eval_df = pd.read_parquet(eval_manifest)
    hyps = []
    with torch.no_grad():
        for _, row in eval_df.iterrows():
            try:
                audio, _ = librosa.load(row["path"], sr=16_000, mono=True)
                feats = processor.feature_extractor(
                    audio, sampling_rate=16_000, return_tensors="pt"
                ).input_features.to(device)
                ids = finetuned.generate(
                    feats,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                )
                hyps.append(processor.decode(ids[0], skip_special_tokens=True))
            except Exception:
                hyps.append("")

    domain_vocab = load_domain_vocab(domain_vocab_file)
    analyzer = DomainWERAnalyzer(domain_vocab)
    report = analyzer.analyze(eval_df["sentence"].tolist(), hyps)

    return {
        "label": label,
        "wer_overall": report.wer_overall,
        "wer_domain_terms": report.wer_domain_terms,
        "wer_common_terms": report.wer_common_terms,
        "n_train": len(pd.read_parquet(train_manifest)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation studies for domain adaptation")
    p.add_argument("--ablation", choices=["data_scaling", "synthetic_mix"], required=True)
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--real_manifest", default=None)
    p.add_argument("--synthetic_manifest", default=None)
    p.add_argument("--eval_manifest", required=True)
    p.add_argument("--domain_vocab", required=True)
    p.add_argument("--output_dir", default="experiments/results/ablations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if args.ablation == "data_scaling":
        assert args.train_manifest, "--train_manifest required for data_scaling"
        df = pd.read_parquet(args.train_manifest)
        rng = np.random.default_rng(args.seed)

        for frac in DATA_SCALING_FRACTIONS:
            n = max(10, int(len(df) * frac))
            subset_idx = rng.choice(len(df), size=n, replace=False)
            subset_path = output_dir / f"subset_{int(frac*100)}.parquet"
            df.iloc[subset_idx].to_parquet(subset_path, index=False)

            label = f"data_{int(frac*100)}pct"
            logger.info("Running %s (n=%d)...", label, n)
            res = run_one_experiment(
                str(subset_path), args.eval_manifest,
                args.domain_vocab, output_dir / "runs", label, args.seed,
            )
            results.append(res)
            logger.info("  WER domain: %.1f%%", res["wer_domain_terms"] * 100)

    elif args.ablation == "synthetic_mix":
        assert args.real_manifest and args.synthetic_manifest, \
            "--real_manifest and --synthetic_manifest required for synthetic_mix"
        real_df = pd.read_parquet(args.real_manifest)
        synth_df = pd.read_parquet(args.synthetic_manifest)
        n_total = len(real_df)
        rng = np.random.default_rng(args.seed)

        for ratio in SYNTHETIC_MIX_RATIOS:
            n_synth = min(int(n_total * ratio), len(synth_df))
            n_real = n_total - n_synth

            real_sub = real_df.sample(n=min(n_real, len(real_df)), random_state=args.seed)
            synth_sub = synth_df.sample(n=n_synth, random_state=args.seed) if n_synth > 0 else pd.DataFrame()
            mixed = pd.concat([real_sub, synth_sub], ignore_index=True)

            label = f"mix_{int(ratio*100)}pct_synth"
            mix_path = output_dir / f"{label}.parquet"
            mixed.to_parquet(mix_path, index=False)

            logger.info("Running %s (%d real + %d synth)...", label, len(real_sub), n_synth)
            res = run_one_experiment(
                str(mix_path), args.eval_manifest,
                args.domain_vocab, output_dir / "runs", label, args.seed,
            )
            results.append(res)
            logger.info("  WER domain: %.1f%%", res["wer_domain_terms"] * 100)

    # Save results
    out_path = output_dir / f"{args.ablation}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Print summary
    logger.info("\nSummary:")
    for r in results:
        logger.info("  %-30s  domain WER: %.1f%%  overall WER: %.1f%%",
                    r["label"], r["wer_domain_terms"] * 100, r["wer_overall"] * 100)


if __name__ == "__main__":
    main()
