"""
Fine-tune Whisper on domain-specific data.

Usage
-----
# Medical domain
python scripts/run_finetune.py \
    --config configs/medical_finetune.yaml \
    --train_manifest data/medical/train_manifest.parquet \
    --eval_manifest data/medical/eval_manifest.parquet

# Financial domain
python scripts/run_finetune.py \
    --config configs/financial_finetune.yaml \
    --train_manifest data/financial_synth/train_manifest.parquet \
    --eval_manifest data/financial_synth/eval_manifest.parquet

# Override hyperparameters
python scripts/run_finetune.py \
    --config configs/medical_finetune.yaml \
    --train_manifest data/medical/train_manifest.parquet \
    --eval_manifest data/medical/eval_manifest.parquet \
    --learning_rate 5e-5 \
    --num_epochs 5
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_adapt.data.feature_extraction import WhisperFeatureExtractor, prepare_batch
from whisper_adapt.models.whisper_lora import LoRAConfig, build_whisper_lora
from whisper_adapt.training.finetune import FinetuneConfig, run_finetune
from transformers import WhisperProcessor

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Whisper on domain speech")
    p.add_argument("--config", required=True,
                   help="Path to YAML config (medical_finetune.yaml or financial_finetune.yaml)")
    p.add_argument("--train_manifest", required=True,
                   help="Parquet manifest from prepare_*_data.py")
    p.add_argument("--eval_manifest", required=True)
    p.add_argument("--output_dir", default=None,
                   help="Override output_dir from config")
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=None)
    return p.parse_args()


def load_audio_dataset(manifest_path: str, extractor: WhisperFeatureExtractor) -> Dataset:
    """
    Load a parquet manifest and convert to HuggingFace Dataset with
    pre-extracted log-mel features.

    Memory note: features are extracted on-the-fly to avoid loading all
    audio into RAM simultaneously.
    """
    df = pd.read_parquet(manifest_path)
    logger.info("Loaded manifest: %d samples from %s", len(df), manifest_path)

    import librosa
    import numpy as np

    records = []
    for _, row in df.iterrows():
        try:
            audio, _ = librosa.load(row["path"], sr=16_000, mono=True)
            processed = extractor(audio, row["sentence"])
            records.append({
                "input_features": processed["input_features"].numpy().tolist(),
                "labels": processed["labels"].numpy().tolist(),
            })
        except Exception as e:
            logger.debug("Skipping %s: %s", row["path"], e)

    logger.info("Feature extraction complete: %d/%d samples", len(records), len(df))
    return Dataset.from_list(records)


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model"]["base_model"]
    lora_params = cfg.get("lora", {})
    train_params = cfg.get("training", {})

    # CLI overrides
    if args.output_dir:
        train_params["output_dir"] = args.output_dir
    if args.num_epochs:
        train_params["num_train_epochs"] = args.num_epochs
    if args.learning_rate:
        train_params["learning_rate"] = args.learning_rate
    if args.batch_size:
        train_params["per_device_train_batch_size"] = args.batch_size
    if args.lora_r:
        lora_params["r"] = args.lora_r

    # Build model and processor
    lora_cfg = LoRAConfig(
        r=lora_params.get("r", 32),
        lora_alpha=lora_params.get("lora_alpha", 64),
        lora_dropout=lora_params.get("lora_dropout", 0.05),
        target_modules=lora_params.get("target_modules",
                                        ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]),
    )

    logger.info("Loading base model: %s", model_id)
    model = build_whisper_lora(model_id=model_id, lora_cfg=lora_cfg)
    processor = WhisperProcessor.from_pretrained(model_id)
    processor.tokenizer.set_prefix_tokens(
        language=cfg["model"].get("language", "en"),
        task=cfg["model"].get("task", "transcribe"),
    )

    extractor = WhisperFeatureExtractor(
        model_id=model_id,
        language=cfg["model"].get("language", "en"),
        task=cfg["model"].get("task", "transcribe"),
    )

    # Load datasets
    logger.info("Preparing training data...")
    train_ds = load_audio_dataset(args.train_manifest, extractor)
    eval_ds = load_audio_dataset(args.eval_manifest, extractor)

    # Training config
    ft_cfg = FinetuneConfig(
        output_dir=train_params.get("output_dir", "checkpoints/domain"),
        num_train_epochs=train_params.get("num_train_epochs", 3),
        per_device_train_batch_size=train_params.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=train_params.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=train_params.get("gradient_accumulation_steps", 2),
        learning_rate=float(train_params.get("learning_rate", 1e-4)),
        warmup_steps=train_params.get("warmup_steps", 500),
        eval_steps=train_params.get("eval_steps", 500),
        save_steps=train_params.get("save_steps", 500),
        logging_steps=train_params.get("logging_steps", 25),
        fp16=train_params.get("fp16", True),
        gradient_checkpointing=train_params.get("gradient_checkpointing", True),
        early_stopping_patience=train_params.get("early_stopping_patience", 5),
        push_to_hub=train_params.get("push_to_hub", False),
    )

    logger.info("Starting fine-tuning...")
    trainer = run_finetune(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        cfg=ft_cfg,
    )

    logger.info("Fine-tuning complete. Best model saved to %s/adapter", ft_cfg.output_dir)
    logger.info("Run evaluate_finetuned.py to compute WER on domain terms.")


if __name__ == "__main__":
    main()
