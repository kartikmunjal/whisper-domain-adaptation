"""Small neural audio codec components: VQ-VAE and FSQ.

This is intentionally compact rather than production-grade SoundStream. The
goal is to make the codec mechanics explicit: encoder, quantizer, commitment
loss, straight-through estimator, and decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
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
    strides: Tuple[int, ...] = (4, 4, 4, 5)
    spectral_loss_weight: float = 1.0

    @property
    def hop_length(self) -> int:
        return prod(self.strides)

    @property
    def frame_rate_hz(self) -> float:
        return self.sample_rate / self.hop_length


class WaveformEncoder(nn.Module):
    """Convolutional encoder from waveform to frame-level latents."""

    def __init__(self, cfg: AudioCodecConfig):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(cfg.channels, cfg.hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
        ]
        for stride in cfg.strides:
            layers.extend([
                nn.Conv1d(
                    cfg.hidden_dim,
                    cfg.hidden_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2,
                ),
                nn.GELU(),
            ])
        layers.append(nn.Conv1d(cfg.hidden_dim, cfg.latent_dim, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return self.net(audio).transpose(1, 2)


class WaveformDecoder(nn.Module):
    """Transpose-convolution decoder from quantized latents to waveform."""

    def __init__(self, cfg: AudioCodecConfig):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(cfg.latent_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for stride in reversed(cfg.strides):
            layers.extend([
                nn.ConvTranspose1d(
                    cfg.hidden_dim,
                    cfg.hidden_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2,
                ),
                nn.GELU(),
            ])
        layers.extend([
            nn.Conv1d(cfg.hidden_dim, cfg.channels, kernel_size=7, padding=3),
            nn.Tanh(),
        ])
        self.net = nn.Sequential(*layers)

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
            self.quantizer_in = nn.Identity()
            self.quantizer_out = nn.Identity()
            self.quantizer = VectorQuantizer(
                self.cfg.codebook_size,
                self.cfg.latent_dim,
                self.cfg.commitment_cost,
            )
        elif quantizer == "fsq":
            fsq_dim = len(self.cfg.fsq_levels)
            self.quantizer_in = nn.Linear(self.cfg.latent_dim, fsq_dim)
            self.quantizer_out = nn.Linear(fsq_dim, self.cfg.latent_dim)
            self.quantizer = FiniteScalarQuantizer(self.cfg.fsq_levels)
        else:
            raise ValueError("quantizer must be 'vq' or 'fsq'")
        self.quantizer_name = quantizer

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode waveform into continuous frame-level latents."""
        return self.encoder(audio)

    def quantize(self, latents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize continuous latents and return decoder-space latents plus codes."""
        bottleneck = self.quantizer_in(latents)
        quantized, info = self.quantizer(bottleneck)
        return self.quantizer_out(quantized), info

    def decode(
        self, quantized: torch.Tensor, target_length: int | None = None
    ) -> torch.Tensor:
        """Decode quantized latents into waveform."""
        return self.decoder(quantized, target_length=target_length)

    @torch.inference_mode()
    def reconstruct(self, audio: torch.Tensor) -> torch.Tensor:
        """Inference-only encode → quantize → decode waveform reconstruction."""
        was_training = self.training
        self.eval()
        target_length = audio.shape[-1]
        quantized, _ = self.quantize(self.encode(audio))
        reconstruction = self.decode(quantized, target_length=target_length)
        if was_training:
            self.train()
        return reconstruction

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_length = audio.shape[-1]
        latents = self.encode(audio)
        quantized, q_info = self.quantize(latents)
        recon = self.decode(quantized, target_length=target_length)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        waveform_loss = F.l1_loss(recon, audio)
        spectral_loss = multi_resolution_stft_loss(recon, audio)
        reconstruction_loss = waveform_loss + self.cfg.spectral_loss_weight * spectral_loss
        total_loss = reconstruction_loss + q_info["quantizer_loss"]
        return {
            "reconstruction": recon,
            "latents": latents,
            "quantized": quantized,
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "waveform_loss": waveform_loss,
            "spectral_loss": spectral_loss,
            **q_info,
        }

    @property
    def nominal_bits_per_frame(self) -> float:
        if self.quantizer_name == "vq":
            return float(self.cfg.codebook_size.bit_length() - 1)
        return float(sum(torch.log2(torch.tensor(self.cfg.fsq_levels)).tolist()))

    @property
    def nominal_bitrate_bps(self) -> float:
        return self.cfg.frame_rate_hz * self.nominal_bits_per_frame

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, map_location: str | torch.device = "cpu"
    ) -> "AudioVQVAE":
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        config = dict(payload["config"])
        for key in ("fsq_levels", "strides"):
            if key in config:
                config[key] = tuple(config[key])
        model = cls(AudioCodecConfig(**config), quantizer=payload["quantizer"])
        model.load_state_dict(payload["state_dict"])
        return model


def codec_rate_hz(
    sample_rate: int, strides: Tuple[int, ...] = (4, 4, 4, 5)
) -> float:
    """Frame rate after the configured convolutional downsampling stack."""
    return sample_rate / prod(strides)


def multi_resolution_stft_loss(
    reconstruction: torch.Tensor,
    reference: torch.Tensor,
    fft_sizes: Tuple[int, ...] = (256, 512, 1024),
) -> torch.Tensor:
    """Log-magnitude STFT loss that prevents low-energy silence solutions."""
    if reference.dim() == 2:
        reference = reference.unsqueeze(1)
    reconstruction = reconstruction.reshape(-1, reconstruction.shape[-1])
    reference = reference.reshape(-1, reference.shape[-1])
    losses = []
    for fft_size in fft_sizes:
        hop = fft_size // 4
        window = torch.hann_window(
            fft_size, device=reference.device, dtype=reference.dtype
        )
        estimated = torch.stft(
            reconstruction,
            n_fft=fft_size,
            hop_length=hop,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        target = torch.stft(
            reference,
            n_fft=fft_size,
            hop_length=hop,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        losses.append(
            F.l1_loss(torch.log1p(estimated), torch.log1p(target))
        )
    return torch.stack(losses).mean()
