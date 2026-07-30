#!/usr/bin/env python3
"""Strict CUDA forward/backward smoke test for codec-token TTS."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from whisper_adapt.models.codec_tts import CodecTTSConfig, CodecTokenTTS


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)
    torch.use_deterministic_algorithms(True, warn_only=False)
    cfg = CodecTTSConfig(
        codebook_size=64,
        d_model=64,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=128,
        max_text_tokens=32,
        max_audio_tokens=64,
    )
    model = CodecTokenTTS(cfg).cuda()
    text = torch.randint(1, 100, (2, 16), device="cuda")
    decoder = torch.randint(0, 64, (2, 32), device="cuda")
    labels = torch.randint(0, 64, (2, 32), device="cuda")
    loss = F.cross_entropy(model(text, decoder).flatten(0, 1), labels.flatten())
    loss.backward()
    gradients = sum(p.grad is not None for p in model.parameters())
    if not gradients:
        raise RuntimeError("No codec-TTS gradients")
    print({"loss": float(loss.detach()), "gradient_tensors": gradients})


if __name__ == "__main__":
    main()
