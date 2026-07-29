#!/usr/bin/env python3
"""One real-audio CUDA forward/backward gate for the locked training stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.data.feature_extraction import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
    WhisperFeatureExtractor,
)
from whisper_adapt.models.whisper_lora import (  # noqa: E402
    LoRAConfig,
    build_whisper_lora,
)
from whisper_adapt.reproducibility import seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="data/financial_research/train_manifest.parquet"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    seed_everything(11)
    row = pd.read_parquet(args.manifest).iloc[0]
    path = Path(row["path"])
    audio, _ = librosa.load(path, sr=16_000, mono=True)
    extractor = WhisperFeatureExtractor(model_id="openai/whisper-small")
    feature = extractor(audio, row["sentence"])
    batch = DataCollatorSpeechSeq2SeqWithPadding(extractor.processor)([feature])
    model = build_whisper_lora(
        model_id="openai/whisper-small", lora_cfg=LoRAConfig(r=32)
    ).to("cuda")
    model.train()
    batch = {key: value.to("cuda") for key, value in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke loss: {loss.item()}")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Missing or non-finite LoRA gradients")
    print({
        "loss": round(float(loss.detach().cpu()), 6),
        "trainable_grad_tensors": len(gradients),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
    })


if __name__ == "__main__":
    main()
