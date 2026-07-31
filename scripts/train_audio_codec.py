#!/usr/bin/env python3
"""Train a small VQ-VAE/FSQ audio codec on local WAV files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from whisper_adapt.models.audio_codec import AudioCodecConfig, AudioVQVAE
from whisper_adapt.reproducibility import collect_provenance, sha256_file


class ManifestAudioDataset(Dataset):
    def __init__(self, manifest: Path, sample_rate: int, clip_seconds: float):
        frame = pd.read_parquet(manifest)
        self.paths = [Path(path) for path in frame["path"].tolist()]
        if not self.paths:
            raise FileNotFoundError(f"No audio rows found in {manifest}")
        missing = [str(path) for path in self.paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing manifest audio, first path: {missing[0]}")
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * clip_seconds)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        audio, _ = librosa.load(self.paths[idx], sr=self.sample_rate, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=35)
        if len(audio) < self.num_samples:
            audio = librosa.util.fix_length(audio, size=self.num_samples)
        else:
            max_start = len(audio) - self.num_samples
            start = random.randint(0, max_start) if max_start else 0
            audio = audio[start : start + self.num_samples]
        return torch.tensor(audio, dtype=torch.float32)


def code_histogram(
    indices: torch.Tensor, quantizer: str, codebook_size: int, fsq_levels: tuple[int, ...]
) -> torch.Tensor:
    """Count actual discrete symbols, using joint mixed-radix FSQ indices."""
    indices = indices.detach().cpu().long()
    if quantizer == "vq":
        symbols = indices.reshape(-1)
        size = codebook_size
    else:
        flat = indices.reshape(-1, indices.shape[-1])
        levels = torch.tensor(fsq_levels, dtype=torch.long)
        multipliers = torch.cumprod(
            torch.cat([torch.ones(1, dtype=torch.long), levels[:-1]]), dim=0
        )
        symbols = (flat * multipliers).sum(dim=1)
        size = int(np.prod(fsq_levels))
    return torch.bincount(symbols, minlength=size)


def usage_report(histogram: torch.Tensor) -> dict:
    total = int(histogram.sum())
    probabilities = histogram.double() / max(total, 1)
    nonzero = probabilities.gt(0)
    entropy = float(
        -(probabilities[nonzero] * probabilities[nonzero].log2()).sum()
    )
    return {
        "n_codes": int(histogram.numel()),
        "n_used_codes": int(histogram.gt(0).sum()),
        "dead_code_fraction": float(histogram.eq(0).double().mean()),
        "entropy_bits_per_frame": entropy,
        "usage_histogram": histogram.tolist(),
    }


def train(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    cfg = AudioCodecConfig(
        sample_rate=args.sample_rate,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        commitment_cost=args.commitment_cost,
        vq_ema_decay=args.vq_ema_decay,
        vq_ema_epsilon=args.vq_ema_epsilon,
        vq_dead_code_batches=args.vq_dead_code_batches,
        fsq_levels=tuple(args.fsq_levels),
        fsq_input_scale=args.fsq_input_scale,
        strides=tuple(args.strides),
    )
    model = AudioVQVAE(cfg, quantizer=args.quantizer).to(device)
    dataset = ManifestAudioDataset(
        Path(args.train_manifest), args.sample_rate, args.clip_seconds
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    diagnostics = []
    for epoch in range(1, args.epochs + 1):
        totals = {
            "loss": 0.0,
            "waveform_loss": 0.0,
            "spectral_loss": 0.0,
            "quantizer_loss": 0.0,
        }
        perplexities = []
        epoch_histogram = None
        dead_resets = 0
        fsq_ranges = []
        for audio in loader:
            audio = audio.to(device)
            out = model(audio)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(out[key].detach().cpu())
            if "perplexity" in out:
                perplexities.append(float(out["perplexity"].detach().cpu()))
            batch_histogram = code_histogram(
                out["indices"], args.quantizer, args.codebook_size, tuple(args.fsq_levels)
            )
            epoch_histogram = (
                batch_histogram
                if epoch_histogram is None
                else epoch_histogram + batch_histogram
            )
            dead_resets += int(
                out.get("dead_codes_reset", torch.tensor(0)).detach().cpu()
            )
            if args.quantizer == "fsq":
                fsq_ranges.append({
                    "minimum": float(out["pre_quantization_min"].detach().cpu()),
                    "maximum": float(out["pre_quantization_max"].detach().cpu()),
                    "rms": float(out["pre_quantization_rms"].detach().cpu()),
                    "saturation_fraction": float(
                        out["bounded_saturation_fraction"].detach().cpu()
                    ),
                })

        denom = max(len(loader), 1)
        epoch_diagnostics = {
            "epoch": epoch,
            **usage_report(epoch_histogram),
            "dead_codes_reset": dead_resets,
        }
        if fsq_ranges:
            epoch_diagnostics["pre_quantization_range"] = {
                key: float(np.mean([row[key] for row in fsq_ranges]))
                for key in fsq_ranges[0]
            }
        diagnostics.append(epoch_diagnostics)
        print(
            f"epoch={epoch} "
            f"loss={totals['loss'] / denom:.4f} "
            f"wave={totals['waveform_loss'] / denom:.4f} "
            f"stft={totals['spectral_loss'] / denom:.4f} "
            f"quant={totals['quantizer_loss'] / denom:.4f} "
            f"perplexity={np.mean(perplexities) if perplexities else float('nan'):.2f} "
            f"used={epoch_diagnostics['n_used_codes']}/{epoch_diagnostics['n_codes']} "
            f"dead={epoch_diagnostics['dead_code_fraction']:.3f} "
            f"entropy={epoch_diagnostics['entropy_bits_per_frame']:.3f} "
            f"resets={dead_resets}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "codec.pt"
    torch.save({"config": cfg.__dict__, "quantizer": args.quantizer, "state_dict": model.state_dict()}, checkpoint)
    (output_dir / "run.json").write_text(json.dumps({
        "schema_version": 1,
        "seed": args.seed,
        "quantizer": args.quantizer,
        "nominal_bits_per_frame": model.nominal_bits_per_frame,
        "frame_rate_hz": cfg.frame_rate_hz,
        "nominal_bitrate_bps": model.nominal_bitrate_bps,
        "n_train_clips": len(dataset),
        "epochs": args.epochs,
        "training_diagnostics": diagnostics,
        "config": cfg.__dict__,
        "checkpoint_sha256": sha256_file(checkpoint),
        "provenance": collect_provenance(
            repo_root=root,
            arguments=vars(args),
            input_files=[args.train_manifest],
            seed=args.seed,
        ),
    }, indent=2), encoding="utf-8")
    print(f"Saved codec checkpoint to {checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact audio VQ-VAE or FSQ codec")
    parser.add_argument(
        "--train_manifest",
        default="data/financial_research/train_manifest.parquet",
    )
    parser.add_argument("--output_dir", default="checkpoints/audio_codec")
    parser.add_argument("--quantizer", choices=["vq", "fsq"], default="vq")
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--clip_seconds", type=float, default=2.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--vq_ema_decay", type=float, default=0.99)
    parser.add_argument("--vq_ema_epsilon", type=float, default=1e-5)
    parser.add_argument("--vq_dead_code_batches", type=int, default=100)
    parser.add_argument("--fsq_levels", type=int, nargs="+", default=[4, 4, 4, 4])
    parser.add_argument("--fsq_input_scale", type=float, default=1.0)
    parser.add_argument("--strides", type=int, nargs="+", default=[4, 4, 4, 5])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
