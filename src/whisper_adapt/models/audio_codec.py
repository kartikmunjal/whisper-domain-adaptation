"""Small neural audio codec components: VQ-VAE and FSQ.

This is intentionally compact rather than production-grade SoundStream. The
goal is to make the codec mechanics explicit: encoder, quantizer, commitment
loss, straight-through estimator, and decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class AudioCodecConfig:
    sample_rate: int = 16_000
    channels: int = 1
    hidden_dim: int = 128
    latent_dim: int = 64
    codebook_size: int = 256
    commitment_cost: float = 0.25
    fsq_levels: Tuple[int, ...] = (8, 8, 8, 8)


class WaveformEncoder(nn.Module):
    """Convolutional encoder from waveform to frame-level latents."""

    def __init__(self, cfg: AudioCodecConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cfg.channels, cfg.hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.hidden_dim, cfg.latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return self.net(audio).transpose(1, 2)


class WaveformDecoder(nn.Module):
    """Transpose-convolution decoder from quantized latents to waveform."""

    def __init__(self, cfg: AudioCodecConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cfg.latent_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.hidden_dim, cfg.channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, latents: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        audio = self.net(latents.transpose(1, 2))
        if target_length is not None:
            audio = audio[..., :target_length]
            if audio.shape[-1] < target_length:
                audio = F.pad(audio, (0, target_length - audio.shape[-1]))
        return audio


class VectorQuantizer(nn.Module):
    """VQ-VAE codebook with straight-through gradients."""

    def __init__(self, codebook_size: int, latent_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flat = latents.reshape(-1, self.latent_dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(latents)

        codebook_loss = F.mse_loss(quantized, latents.detach())
        commitment_loss = F.mse_loss(latents, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized_st = latents + (quantized - latents).detach()
        one_hot = F.one_hot(indices, self.codebook_size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        return quantized_st, {
            "quantizer_loss": loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "indices": indices.view(latents.shape[:2]),
        }


class FiniteScalarQuantizer(nn.Module):
    """Finite Scalar Quantization baseline from "VQ-VAE Made Simple".

    FSQ removes the learned codebook: each latent dimension is bounded, rounded
    onto a fixed scalar grid, then passed through a straight-through estimator.
    """

    def __init__(self, levels: Tuple[int, ...]):
        super().__init__()
        self.levels = torch.tensor(levels, dtype=torch.float32)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        levels = self.levels.to(latents.device)
        if latents.shape[-1] % len(levels) != 0:
            raise ValueError("latent_dim must be divisible by number of FSQ levels")

        repeats = latents.shape[-1] // len(levels)
        levels = levels.repeat_interleave(repeats).view(1, 1, -1)
        bounded = torch.tanh(latents)
        scaled = bounded * ((levels - 1) / 2)
        rounded = torch.round(scaled)
        quantized = rounded / ((levels - 1) / 2).clamp_min(1)
        quantized_st = latents + (quantized - latents).detach()

        return quantized_st, {
            "quantizer_loss": torch.zeros((), device=latents.device, dtype=latents.dtype),
            "indices": (rounded + (levels - 1) / 2).long(),
        }


class AudioVQVAE(nn.Module):
    """End-to-end audio VQ-VAE/FSQ codec."""

    def __init__(self, cfg: AudioCodecConfig | None = None, quantizer: str = "vq"):
        super().__init__()
        self.cfg = cfg or AudioCodecConfig()
        self.encoder = WaveformEncoder(self.cfg)
        self.decoder = WaveformDecoder(self.cfg)
        if quantizer == "vq":
            self.quantizer = VectorQuantizer(
                self.cfg.codebook_size,
                self.cfg.latent_dim,
                self.cfg.commitment_cost,
            )
        elif quantizer == "fsq":
            self.quantizer = FiniteScalarQuantizer(self.cfg.fsq_levels)
        else:
            raise ValueError("quantizer must be 'vq' or 'fsq'")
        self.quantizer_name = quantizer

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_length = audio.shape[-1]
        latents = self.encoder(audio)
        quantized, q_info = self.quantizer(latents)
        recon = self.decoder(quantized, target_length=target_length)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        reconstruction_loss = F.l1_loss(recon, audio)
        total_loss = reconstruction_loss + q_info["quantizer_loss"]
        return {
            "reconstruction": recon,
            "latents": latents,
            "quantized": quantized,
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            **q_info,
        }


def codec_rate_hz(sample_rate: int, num_strides: int = 2) -> float:
    """Frame rate after the default stride-2 convolution stack."""
    return sample_rate / (2 ** num_strides)
