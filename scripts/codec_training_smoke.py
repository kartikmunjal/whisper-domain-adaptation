#!/usr/bin/env python3
"""Strict GPU forward/backward and inference smoke test for both codecs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from whisper_adapt.models.audio_codec import AudioCodecConfig, AudioVQVAE


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for confirmatory codec runs")
    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device("cuda")
    audio = torch.randn(2, 16_000, device=device)
    for quantizer, kwargs in (
        ("vq", {"codebook_size": 256}),
        ("fsq", {"fsq_levels": (4, 4, 4, 4)}),
    ):
        cfg = AudioCodecConfig(hidden_dim=32, latent_dim=16, **kwargs)
        model = AudioVQVAE(cfg, quantizer=quantizer).to(device)
        output = model(audio)
        output["loss"].backward()
        gradients = sum(
            parameter.grad is not None for parameter in model.parameters()
        )
        if gradients == 0:
            raise RuntimeError(f"No gradients for {quantizer}")
        reconstruction = model.reconstruct(audio)
        if reconstruction.shape != (2, 1, 16_000):
            raise RuntimeError(
                f"Bad {quantizer} reconstruction shape: {reconstruction.shape}"
            )
        print({
            "quantizer": quantizer,
            "loss": float(output["loss"].detach()),
            "gradient_tensors": gradients,
            "frame_rate_hz": cfg.frame_rate_hz,
            "nominal_bitrate_bps": model.nominal_bitrate_bps,
        })


if __name__ == "__main__":
    main()
