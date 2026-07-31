"""Small neural audio codec components: VQ-VAE and FSQ.

This is intentionally compact rather than production-grade SoundStream. The
goal is to make the codec mechanics explicit while retaining the safeguards
needed for a meaningful comparison: EMA VQ updates, dead-code replacement,
FSQ range normalization, and auditable usage statistics.
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
    vq_ema_decay: float = 0.99
    vq_ema_epsilon: float = 1e-5
    vq_dead_code_batches: int = 100
    fsq_levels: Tuple[int, ...] = (8, 8, 8, 8)
    fsq_input_scale: float = 1.0
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
    """VQ codebook updated by EMA, with deterministic dead-code accounting."""

    def __init__(
        self,
        codebook_size: int,
        latent_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        dead_code_batches: int = 100,
    ):
        super().__init__()
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if dead_code_batches < 1:
            raise ValueError("dead_code_batches must be positive")
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.dead_code_batches = dead_code_batches
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.embedding.weight.requires_grad_(False)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embedding_sum", self.embedding.weight.detach().clone())
        self.register_buffer(
            "batches_since_use",
            torch.full((codebook_size,), dead_code_batches, dtype=torch.long),
        )
        self.register_buffer("total_resets", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _ema_update(
        self, flat: torch.Tensor, indices: torch.Tensor, one_hot: torch.Tensor
    ) -> int:
        counts = one_hot.sum(dim=0)
        embedding_sum = one_hot.t() @ flat
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            counts, alpha=1.0 - self.ema_decay
        )
        self.ema_embedding_sum.mul_(self.ema_decay).add_(
            embedding_sum, alpha=1.0 - self.ema_decay
        )
        self.batches_since_use.add_(1)
        self.batches_since_use[counts.gt(0)] = 0

        dead = self.batches_since_use.ge(self.dead_code_batches)
        n_dead = int(dead.sum())
        if n_dead and flat.shape[0]:
            # torch RNG is seeded by the training entrypoint, making replacement
            # reproducible while drawing actual encoder outputs.
            samples = flat[
                torch.randint(flat.shape[0], (n_dead,), device=flat.device)
            ]
            self.embedding.weight.data[dead] = samples
            self.ema_embedding_sum[dead] = samples
            self.ema_cluster_size[dead] = 1.0
            self.batches_since_use[dead] = 0
            self.total_resets.add_(n_dead)

        normalizer = self.ema_cluster_size.sum()
        smoothed = (
            (self.ema_cluster_size + self.ema_epsilon)
            / (normalizer + self.codebook_size * self.ema_epsilon)
            * normalizer
        ).clamp_min(self.ema_epsilon)
        self.embedding.weight.data.copy_(
            self.ema_embedding_sum / smoothed.unsqueeze(1)
        )
        return n_dead

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flat = latents.reshape(-1, self.latent_dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(latents)

        commitment_loss = F.mse_loss(latents, quantized.detach())
        loss = self.commitment_cost * commitment_loss

        quantized_st = latents + (quantized - latents).detach()
        one_hot = F.one_hot(indices, self.codebook_size).float()
        resets = self._ema_update(flat.detach(), indices, one_hot) if self.training else 0
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        usage_histogram = one_hot.sum(dim=0)

        return quantized_st, {
            "quantizer_loss": loss,
            "codebook_loss": torch.zeros_like(commitment_loss),
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "usage_histogram": usage_histogram,
            "dead_code_fraction": usage_histogram.eq(0).float().mean(),
            "dead_codes_reset": torch.tensor(
                resets, device=latents.device, dtype=torch.long
            ),
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
        indices = (rounded + (levels - 1) / 2).long()
        flat_indices = indices.reshape(-1, indices.shape[-1])
        unique_vectors = torch.unique(flat_indices, dim=0).shape[0]

        return quantized_st, {
            "quantizer_loss": torch.zeros((), device=latents.device, dtype=latents.dtype),
            "indices": indices,
            "pre_quantization_min": latents.detach().amin(),
            "pre_quantization_max": latents.detach().amax(),
            "pre_quantization_rms": latents.detach().square().mean().sqrt(),
            "bounded_saturation_fraction": bounded.detach().abs().gt(0.95).float().mean(),
            "unique_code_vectors": torch.tensor(
                unique_vectors, device=latents.device, dtype=torch.long
            ),
        }


class FSQRangeNormalizer(nn.Module):
    """Normalize each projected FSQ dimension using running encoder statistics."""

    def __init__(
        self,
        dimensions: int,
        target_std: float = 1.0,
        epsilon: float = 1e-8,
        momentum: float = 0.1,
    ):
        super().__init__()
        if target_std <= 0:
            raise ValueError("target_std must be positive")
        self.target_std = target_std
        self.epsilon = epsilon
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(dimensions))
        self.register_buffer("running_variance", torch.ones(dimensions))
        self.register_buffer("batches_tracked", torch.zeros((), dtype=torch.long))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.training:
            reduce_dimensions = tuple(range(values.dim() - 1))
            mean = values.mean(dim=reduce_dimensions)
            variance = values.var(dim=reduce_dimensions, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_variance.lerp_(variance.detach(), self.momentum)
                self.batches_tracked.add_(1)
        else:
            mean = self.running_mean
            variance = self.running_variance
        return (
            (values - mean) / variance.clamp_min(self.epsilon).sqrt()
            * self.target_std
        )


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
                self.cfg.vq_ema_decay,
                self.cfg.vq_ema_epsilon,
                self.cfg.vq_dead_code_batches,
            )
            self.fsq_normalizer = nn.Identity()
        elif quantizer == "fsq":
            fsq_dim = len(self.cfg.fsq_levels)
            self.quantizer_in = nn.Linear(self.cfg.latent_dim, fsq_dim)
            self.quantizer_out = nn.Linear(fsq_dim, self.cfg.latent_dim)
            self.fsq_normalizer = FSQRangeNormalizer(
                fsq_dim, self.cfg.fsq_input_scale
            )
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
        bottleneck = self.fsq_normalizer(bottleneck)
        quantized, info = self.quantizer(bottleneck)
        return self.quantizer_out(quantized), info

    def decode(
        self, quantized: torch.Tensor, target_length: int | None = None
    ) -> torch.Tensor:
        """Decode quantized latents into waveform."""
        return self.decoder(quantized, target_length=target_length)

    @torch.inference_mode()
    def decode_vq_indices(
        self, indices: torch.Tensor, target_length: int | None = None
    ) -> torch.Tensor:
        """Decode discrete VQ indices produced by a text-to-codec model."""
        if self.quantizer_name != "vq":
            raise RuntimeError("decode_vq_indices is only defined for VQ checkpoints")
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        if indices.numel() == 0:
            raise ValueError("At least one VQ index is required")
        if indices.min() < 0 or indices.max() >= self.cfg.codebook_size:
            raise ValueError("VQ index is outside the codec codebook")
        if target_length is None:
            target_length = indices.shape[1] * prod(self.cfg.strides)
        quantized = self.quantizer.embedding(indices)
        return self.decode(quantized, target_length=target_length)

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

    @torch.inference_mode()
    def reconstruct_chunked(
        self,
        audio: torch.Tensor,
        chunk_samples: int = 160_000,
        overlap_samples: int = 16_000,
    ) -> torch.Tensor:
        """Reconstruct arbitrarily long audio with deterministic overlap-add."""
        if overlap_samples >= chunk_samples:
            raise ValueError("overlap_samples must be smaller than chunk_samples")
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        total = audio.shape[-1]
        if total <= chunk_samples:
            return self.reconstruct(audio)
        step = chunk_samples - overlap_samples
        output = torch.zeros(
            (audio.shape[0], 1, total), device=audio.device, dtype=audio.dtype
        )
        weights = torch.zeros_like(output)
        for start in range(0, total, step):
            end = min(start + chunk_samples, total)
            chunk = audio[..., start:end]
            reconstructed = self.reconstruct(chunk)
            window = torch.ones(
                reconstructed.shape[-1],
                device=audio.device,
                dtype=audio.dtype,
            )
            fade = min(overlap_samples, reconstructed.shape[-1] // 2)
            if start > 0 and fade:
                window[:fade] = torch.linspace(
                    0, 1, fade, device=audio.device, dtype=audio.dtype
                )
            if end < total and fade:
                window[-fade:] = torch.linspace(
                    1, 0, fade, device=audio.device, dtype=audio.dtype
                )
            output[..., start:end] += reconstructed * window
            weights[..., start:end] += window
            if end == total:
                break
        return output / weights.clamp_min(1e-8)

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
        incompatible = model.load_state_dict(payload["state_dict"], strict=False)
        allowed_missing = {
            "quantizer.ema_cluster_size",
            "quantizer.ema_embedding_sum",
            "quantizer.batches_since_use",
            "quantizer.total_resets",
            "fsq_normalizer.running_mean",
            "fsq_normalizer.running_variance",
            "fsq_normalizer.batches_tracked",
        }
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint is incompatible: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
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
