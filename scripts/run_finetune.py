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
import hashlib
import json
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
from whisper_adapt.reproducibility import collect_provenance, seed_everything
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
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--acoustic_augmentation", action="store_true")
    return p.parse_args()


def shoebox_rir(rng, sample_rate: int = 16_000):
    """Deterministic order-2 image-source RIR for a sampled shoebox room."""
    import numpy as np
    room = rng.uniform([3.0, 3.0, 2.4], [10.0, 8.0, 4.0]); source = rng.uniform([0.5]*3, room-0.5); microphone = rng.uniform([0.5]*3, room-0.5); rt60=float(rng.uniform(0.2,0.8))
    volume=float(np.prod(room)); area=float(2*(room[0]*room[1]+room[0]*room[2]+room[1]*room[2])); absorption=float(np.clip(0.161*volume/(area*rt60),0.05,0.95)); reflection=np.sqrt(1-absorption)
    length=int(np.ceil((rt60+0.1)*sample_rate)); rir=np.zeros(length,dtype=np.float32)
    for nx in range(-2,3):
        for ny in range(-2,3):
            for nz in range(-2,3):
                order=abs(nx)+abs(ny)+abs(nz); image=2*np.array([nx,ny,nz])*room+((-1.0)**np.array([nx,ny,nz]))*source; distance=float(np.linalg.norm(image-microphone)); delay=int(round(distance/343.0*sample_rate))
                if delay<length: rir[delay]+=reflection**order/max(distance,0.1)
    direct=float(np.max(np.abs(rir))); return rir/max(direct,1e-8)


def augment_financial_audio(audio, seed: int, key: str, sample_rate: int = 16_000):
    """Apply the locked augmentation recipe with clip-stable randomness."""
    import numpy as np
    from scipy.signal import fftconvolve
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8],"little")); result=np.asarray(audio,dtype=np.float32)
    if rng.random()<0.5: result=fftconvolve(result,shoebox_rir(rng,sample_rate),mode="full").astype(np.float32)
    if rng.random()<0.5:
        snr_db=float(rng.uniform(5,25)); signal_rms=float(np.sqrt(np.mean(result**2))+1e-8); noise=rng.standard_normal(len(result)).astype(np.float32); noise/=float(np.sqrt(np.mean(noise**2))+1e-8); result=result+noise*signal_rms/(10**(snr_db/20))
    if rng.random()<0.5: result=result*float(10**(rng.uniform(-6,6)/20))
    if rng.random()<0.5: result=np.pad(result,(int(rng.uniform(0,.5)*sample_rate),int(rng.uniform(0,.5)*sample_rate)))
    peak=float(np.max(np.abs(result))) if len(result) else 0.0; return (result/max(peak/.99,1.0)).astype(np.float32)


def load_audio_dataset(manifest_path: str, extractor: WhisperFeatureExtractor, *, augmentation_seed: int | None = None) -> Dataset:
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
    repo_root = Path(__file__).resolve().parents[1]
    for _, row in df.iterrows():
        try:
            path = Path(row["path"])
            if not path.is_absolute():
                path = repo_root / path
            audio, _ = librosa.load(path, sr=16_000, mono=True)
            if augmentation_seed is not None:
                audio = augment_financial_audio(audio, augmentation_seed, str(row.get("id", row["path"])))
            processed = extractor(audio, row["sentence"])
            records.append({
                "input_features": processed["input_features"].numpy().tolist(),
                "attention_mask": processed["attention_mask"].numpy().tolist(),
                "labels": processed["labels"].numpy().tolist(),
            })
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed for {row['path']}: {e}") from e

    logger.info("Feature extraction complete: %d/%d samples", len(records), len(df))
    return Dataset.from_list(records)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

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

    output_path = Path(train_params.get("output_dir", "checkpoints/domain"))
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(
            f"Refusing non-empty output directory: {output_path}. "
            "Archive the failed/completed run or choose a new directory."
        )

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
    train_ds = load_audio_dataset(args.train_manifest, extractor, augmentation_seed=args.seed if args.acoustic_augmentation else None)
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
        dataloader_num_workers=train_params.get("dataloader_num_workers", 0),
        seed=args.seed,
    )

    logger.info("Starting fine-tuning...")
    trainer = run_finetune(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        cfg=ft_cfg,
    )

    provenance = collect_provenance(
        repo_root=Path(__file__).resolve().parents[1],
        arguments=vars(args),
        input_files=[args.config, args.train_manifest, args.eval_manifest],
        seed=args.seed,
    )
    output_dir = Path(ft_cfg.output_dir)
    (output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str)
    )

    logger.info("Fine-tuning complete. Best model saved to %s/adapter", ft_cfg.output_dir)
    logger.info("Run evaluate_finetuned.py to compute WER on domain terms.")


if __name__ == "__main__":
    main()
