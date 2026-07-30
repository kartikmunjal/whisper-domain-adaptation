#!/usr/bin/env python3
"""Train a small VQ-VAE/FSQ audio codec on local WAV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import librosa
import torch
from torch.utils.data import DataLoader, Dataset

from whisper_adapt.models.audio_codec import AudioCodecConfig, AudioVQVAE


class WavFolderDataset(Dataset):
    def __init__(self, audio_dir: Path, sample_rate: int, clip_seconds: float):
        self.paths = sorted(audio_dir.glob("*.wav"))
        if not self.paths:
            raise FileNotFoundError(f"No .wav files found in {audio_dir}")
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * clip_seconds)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        audio, _ = librosa.load(self.paths[idx], sr=self.sample_rate, mono=True)
        if len(audio) < self.num_samples:
            audio = librosa.util.fix_length(audio, size=self.num_samples)
        else:
            audio = audio[: self.num_samples]
        return torch.tensor(audio, dtype=torch.float32)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    cfg = AudioCodecConfig(
        sample_rate=args.sample_rate,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        commitment_cost=args.commitment_cost,
    )
    model = AudioVQVAE(cfg, quantizer=args.quantizer).to(device)
    dataset = WavFolderDataset(Path(args.audio_dir), args.sample_rate, args.clip_seconds)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "quantizer_loss": 0.0}
        for audio in loader:
            audio = audio.to(device)
            out = model(audio)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(out[key].detach().cpu())

        denom = max(len(loader), 1)
        print(
            f"epoch={epoch} "
            f"loss={totals['loss'] / denom:.4f} "
            f"recon={totals['reconstruction_loss'] / denom:.4f} "
            f"quant={totals['quantizer_loss'] / denom:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "quantizer": args.quantizer, "state_dict": model.state_dict()}, output_dir / "codec.pt")
    print(f"Saved codec checkpoint to {output_dir / 'codec.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact audio VQ-VAE or FSQ codec")
    parser.add_argument("--audio_dir", default="data/financial_synth_eval")
    parser.add_argument("--output_dir", default="checkpoints/audio_codec")
    parser.add_argument("--quantizer", choices=["vq", "fsq"], default="vq")
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--clip_seconds", type=float, default=2.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
