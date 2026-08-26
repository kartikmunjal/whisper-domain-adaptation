"""Matched-backbone continuous baseline for the discrete codec experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from whisper_adapt.models.audio_codec import (
    AudioCodecConfig,
    WaveformDecoder,
    WaveformEncoder,
    multi_resolution_stft_loss,
)


@dataclass
class ContinuousCodecConfig:
    audio: AudioCodecConfig
    bottleneck_dim: int = 8
    kl_weight: float = 1e-4
    quantization_bits: int = 6
    quantization_clip: float = 3.0

    def __post_init__(self) -> None:
        if self.bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be positive")
        if self.kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        if not 1 <= self.quantization_bits <= 16:
            raise ValueError("quantization_bits must be in [1, 16]")
        if self.quantization_clip <= 0:
            raise ValueError("quantization_clip must be positive")


def uniform_quantize(
    values: torch.Tensor, bits: int, clip_value: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric fixed-range quantization with a straight-through estimate."""
    if not 1 <= bits <= 16:
        raise ValueError("bits must be in [1, 16]")
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")
    levels = 2**bits
    clipped = values.clamp(-clip_value, clip_value)
    indices = torch.round((clipped + clip_value) * (levels - 1) / (2 * clip_value)).long()
    restored = indices.to(values.dtype) * (2 * clip_value) / (levels - 1) - clip_value
    straight_through = values + (restored - values).detach()
    saturation = values.detach().abs().ge(clip_value).float().mean()
    return straight_through, indices, saturation


class ContinuousAudioVAE(nn.Module):
    """Gaussian VAE using the exact waveform backbone of ``AudioVQVAE``."""

    def __init__(self, config: ContinuousCodecConfig):
        super().__init__()
        self.config = config
        self.cfg = config.audio
        self.encoder = WaveformEncoder(self.cfg)
        self.posterior = nn.Linear(self.cfg.latent_dim, 2 * config.bottleneck_dim)
        self.decoder_projection = nn.Linear(config.bottleneck_dim, self.cfg.latent_dim)
        self.decoder = WaveformDecoder(self.cfg)

    def encode_distribution(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = self.posterior(self.encoder(audio)).chunk(2, dim=-1)
        return mean, log_variance.clamp(-12.0, 8.0)

    @staticmethod
    def sample(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def decode(self, latents: torch.Tensor, target_length: int) -> torch.Tensor:
        return self.decoder(self.decoder_projection(latents), target_length)

    @torch.inference_mode()
    def reconstruct(self, audio: torch.Tensor, quantized: bool = False) -> torch.Tensor:
        was_training = self.training
        self.eval()
        mean, _ = self.encode_distribution(audio)
        if quantized:
            mean, _, _ = uniform_quantize(
                mean, self.config.quantization_bits, self.config.quantization_clip
            )
        output = self.decode(mean, audio.shape[-1])
        if was_training:
            self.train()
        return output

    @torch.inference_mode()
    def reconstruct_chunked(
        self,
        audio: torch.Tensor,
        *,
        quantization_bits: int | None = None,
        chunk_samples: int = 160_000,
        overlap_samples: int = 16_000,
    ) -> torch.Tensor:
        """Deterministically reconstruct long audio with linear crossfades."""
        if audio.dim() != 2 or audio.shape[0] != 1:
            raise ValueError("chunked reconstruction requires [1, samples] audio")
        if chunk_samples <= overlap_samples or overlap_samples < 0:
            raise ValueError("require chunk_samples > overlap_samples >= 0")
        total = audio.shape[-1]
        if total <= chunk_samples:
            mean, _ = self.encode_distribution(audio)
            if quantization_bits is not None:
                mean, _, _ = uniform_quantize(
                    mean, quantization_bits, self.config.quantization_clip
                )
            return self.decode(mean, total)
        step = chunk_samples - overlap_samples
        output = torch.zeros(1, 1, total, device=audio.device, dtype=audio.dtype)
        weight = torch.zeros_like(output)
        for start in range(0, total, step):
            end = min(start + chunk_samples, total)
            segment = audio[..., start:end]
            mean, _ = self.encode_distribution(segment)
            if quantization_bits is not None:
                mean, _, _ = uniform_quantize(
                    mean, quantization_bits, self.config.quantization_clip
                )
            decoded = self.decode(mean, end - start)
            window = torch.ones(end - start, device=audio.device, dtype=audio.dtype)
            fade = min(overlap_samples, end - start)
            if start > 0 and fade:
                window[:fade] = torch.linspace(0, 1, fade, device=audio.device)
            if end < total and fade:
                window[-fade:] = torch.linspace(1, 0, fade, device=audio.device)
            output[..., start:end] += decoded * window
            weight[..., start:end] += window
            if end == total:
                break
        return output / weight.clamp_min(torch.finfo(output.dtype).eps)

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = audio.unsqueeze(1) if audio.dim() == 2 else audio
        mean, log_variance = self.encode_distribution(audio)
        latents = self.sample(mean, log_variance) if self.training else mean
        reconstruction = self.decode(latents, target.shape[-1])
        waveform_loss = F.l1_loss(reconstruction, target)
        spectral_loss = multi_resolution_stft_loss(reconstruction, target)
        kl_per_frame = -0.5 * (
            1.0 + log_variance - mean.square() - log_variance.exp()
        ).sum(dim=-1).mean()
        reconstruction_loss = waveform_loss + self.cfg.spectral_loss_weight * spectral_loss
        return {
            "loss": reconstruction_loss + self.config.kl_weight * kl_per_frame,
            "reconstruction": reconstruction,
            "reconstruction_loss": reconstruction_loss,
            "waveform_loss": waveform_loss,
            "spectral_loss": spectral_loss,
            "kl_per_frame": kl_per_frame,
            "posterior_mean": mean,
            "posterior_log_variance": log_variance,
        }

    @property
    def fixed_width_bitrate_bps(self) -> float:
        return (
            self.cfg.frame_rate_hz
            * self.config.bottleneck_dim
            * self.config.quantization_bits
        )

    def save_checkpoint(self, path: str | Path, **metadata: object) -> None:
        payload = {
            "schema_version": 1,
            "config": {
                **asdict(self.config),
                "audio": asdict(self.config.audio),
            },
            "state_dict": self.state_dict(),
            "metadata": metadata,
        }
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "ContinuousAudioVAE":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = dict(payload["config"])
        audio = dict(config.pop("audio"))
        audio["fsq_levels"] = tuple(audio["fsq_levels"])
        audio["strides"] = tuple(audio["strides"])
        model = cls(ContinuousCodecConfig(audio=AudioCodecConfig(**audio), **config))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model
