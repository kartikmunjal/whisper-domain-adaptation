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


def train(args: argparse.Namespace) -> None:
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
        fsq_levels=tuple(args.fsq_levels),
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

    for epoch in range(1, args.epochs + 1):
        totals = {
            "loss": 0.0,
            "waveform_loss": 0.0,
            "spectral_loss": 0.0,
            "quantizer_loss": 0.0,
        }
        perplexities = []
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

        denom = max(len(loader), 1)
        print(
            f"epoch={epoch} "
            f"loss={totals['loss'] / denom:.4f} "
            f"wave={totals['waveform_loss'] / denom:.4f} "
            f"stft={totals['spectral_loss'] / denom:.4f} "
            f"quant={totals['quantizer_loss'] / denom:.4f} "
            f"perplexity={np.mean(perplexities) if perplexities else float('nan'):.2f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "quantizer": args.quantizer, "state_dict": model.state_dict()}, output_dir / "codec.pt")
    (output_dir / "run.json").write_text(json.dumps({
        "seed": args.seed,
        "quantizer": args.quantizer,
        "nominal_bits_per_frame": model.nominal_bits_per_frame,
        "frame_rate_hz": cfg.frame_rate_hz,
        "nominal_bitrate_bps": model.nominal_bitrate_bps,
        "n_train_clips": len(dataset),
        "epochs": args.epochs,
        "config": cfg.__dict__,
    }, indent=2))
    print(f"Saved codec checkpoint to {output_dir / 'codec.pt'}")


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
    parser.add_argument("--fsq_levels", type=int, nargs="+", default=[4, 4, 4, 4])
    parser.add_argument("--strides", type=int, nargs="+", default=[4, 4, 4, 5])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
