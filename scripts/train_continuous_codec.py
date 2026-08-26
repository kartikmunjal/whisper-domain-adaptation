#!/usr/bin/env python3
"""Train the preregistered matched-backbone continuous audio VAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_audio_codec import ManifestAudioDataset
from whisper_adapt.continuous_codec import ContinuousAudioVAE, ContinuousCodecConfig
from whisper_adapt.models.audio_codec import AudioCodecConfig
from whisper_adapt.reproducibility import collect_provenance, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default="data/financial_research/train_manifest.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-latent-dim", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--strides", type=int, nargs="+", default=[4, 4, 4, 5])
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--quantization-bits", type=int, default=6)
    parser.add_argument("--quantization-clip", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(args.device)
    config = ContinuousCodecConfig(
        audio=AudioCodecConfig(
            sample_rate=args.sample_rate,
            hidden_dim=args.hidden_dim,
            latent_dim=args.encoder_latent_dim,
            strides=tuple(args.strides),
        ),
        bottleneck_dim=args.bottleneck_dim,
        kl_weight=args.kl_weight,
        quantization_bits=args.quantization_bits,
        quantization_clip=args.quantization_clip,
    )
    model = ContinuousAudioVAE(config).to(device)
    dataset = ManifestAudioDataset(
        root / args.train_manifest, args.sample_rate, args.clip_seconds
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        totals = {key: 0.0 for key in ("loss", "reconstruction_loss", "kl_per_frame")}
        model.train()
        for audio in loader:
            output = model(audio.to(device))
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(output[key].detach().cpu())
        row = {"epoch": epoch, **{key: value / len(loader) for key, value in totals.items()}}
        history.append(row)
        print(json.dumps(row))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "continuous_codec.pt"
    model.save_checkpoint(checkpoint, seed=args.seed)
    report = {
        "schema_version": 1,
        "seed": args.seed,
        "n_train_clips": len(dataset),
        "fixed_width_bitrate_bps": model.fixed_width_bitrate_bps,
        "history": history,
        "checkpoint_sha256": sha256_file(checkpoint),
        "provenance": collect_provenance(
            repo_root=root,
            arguments=vars(args),
            input_files=[root / args.train_manifest],
            seed=args.seed,
        ),
    }
    (output_dir / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
