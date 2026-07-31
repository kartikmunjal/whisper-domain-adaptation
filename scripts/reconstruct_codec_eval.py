#!/usr/bin/env python3
"""Reconstruct a real-audio manifest and compute clip-level codec metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.models.audio_codec import AudioVQVAE
from whisper_adapt.reproducibility import collect_provenance, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    reference -= reference.mean()
    estimate -= estimate.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12)
    target = scale * reference
    noise = estimate - target
    return float(10 * np.log10((np.dot(target, target) + 1e-12) /
                               (np.dot(noise, noise) + 1e-12)))


def log_mel_l1_db(
    reference: np.ndarray, estimate: np.ndarray, sample_rate: int
) -> float:
    """Phase-insensitive mean absolute log-mel distance in decibels."""
    length = min(len(reference), len(estimate))
    if length < 400:
        raise ValueError("log-mel distance requires at least 400 aligned samples")
    kwargs = {
        "sr": sample_rate,
        "n_fft": 400,
        "hop_length": 160,
        "win_length": 400,
        "n_mels": 80,
        "fmin": 20,
        "fmax": min(7600, sample_rate / 2),
        "power": 2.0,
        "center": False,
    }
    reference_mel = librosa.feature.melspectrogram(y=reference[:length], **kwargs)
    estimate_mel = librosa.feature.melspectrogram(y=estimate[:length], **kwargs)
    reference_db = 10.0 * np.log10(np.maximum(reference_mel, 1e-10))
    estimate_db = 10.0 * np.log10(np.maximum(estimate_mel, 1e-10))
    return float(np.mean(np.abs(reference_db - estimate_db)))


def bootstrap_mean(values: np.ndarray, n: int) -> list[float]:
    rng = np.random.default_rng(20260729)
    means = np.empty(n)
    for index in range(n):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def empirical_entropy(indices: torch.Tensor) -> float:
    """Return entropy per codec frame for VQ scalars or FSQ code vectors.

    VQ emits ``[batch, frames]`` scalar indices, while FSQ emits
    ``[batch, frames, dimensions]`` vectors.  Flattening a two-dimensional
    VQ tensor into one row would incorrectly count an entire utterance as a
    single symbol and therefore report zero bitrate.
    """
    array = indices.detach().cpu().numpy()
    if array.ndim <= 2:
        _, counts = np.unique(array.reshape(-1), return_counts=True)
    else:
        flattened = array.reshape(-1, array.shape[-1])
        _, counts = np.unique(flattened, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = (root / args.checkpoint).resolve()
    output = (root / args.output_dir).resolve()
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = AudioVQVAE.from_checkpoint(checkpoint, map_location=device).to(device)
    model.eval()
    frame = pd.read_parquet(root / args.eval_manifest)
    rows = []
    for row in frame.to_dict("records"):
        reference, _ = librosa.load(root / row["path"], sr=model.cfg.sample_rate, mono=True)
        tensor = torch.tensor(reference, dtype=torch.float32, device=device).unsqueeze(0)
        reconstruction = model.reconstruct_chunked(tensor)[0, 0].cpu().numpy()
        wav_path = wav_dir / f"{row['id']}.wav"
        sf.write(wav_path, reconstruction, model.cfg.sample_rate, subtype="PCM_16")

        entropy_bits = []
        chunk = 160_000
        for start in range(0, len(reference), chunk):
            latent = model.encode(tensor[..., start:start + chunk])
            _, info = model.quantize(latent)
            entropy_bits.append(empirical_entropy(info["indices"]))
        rows.append({
            **row,
            "original_path": row["path"],
            "path": str(wav_path.relative_to(root)),
            "codec_checkpoint": str(checkpoint.relative_to(root)),
            "codec_sha256": sha256_file(checkpoint),
            "reconstructed_sha256": sha256_file(wav_path),
            "si_sdr_db": si_sdr(reference, reconstruction),
            "log_mel_l1_db": log_mel_l1_db(
                reference, reconstruction, model.cfg.sample_rate
            ),
            "empirical_bits_per_frame": float(np.mean(entropy_bits)),
        })
    result = pd.DataFrame(rows)
    result.to_parquet(output / "reconstructed_manifest.parquet", index=False)
    values = result.si_sdr_db.to_numpy()
    mel_values = result.log_mel_l1_db.to_numpy()
    empirical_bps = result.empirical_bits_per_frame.mean() * model.cfg.frame_rate_hz
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "quantizer": model.quantizer_name,
        "n_clips": len(result),
        "nominal_bitrate_bps": model.nominal_bitrate_bps,
        "empirical_bitrate_bps": float(empirical_bps),
        "si_sdr_db": {
            "mean": float(values.mean()),
            "clip_bootstrap_95_ci": bootstrap_mean(values, args.bootstrap_resamples),
            "clip_values": values.tolist(),
        },
        "log_mel_l1_db": {
            "mean": float(mel_values.mean()),
            "clip_bootstrap_95_ci": bootstrap_mean(
                mel_values, args.bootstrap_resamples
            ),
            "clip_values": mel_values.tolist(),
            "lower_is_better": True,
            "configuration": {
                "n_mels": 80,
                "window_samples": 400,
                "hop_samples": 160,
                "fmin_hz": 20,
                "fmax_hz": min(7600, model.cfg.sample_rate / 2),
            },
        },
        "provenance": collect_provenance(
            repo_root=root,
            arguments=vars(args),
            input_files=[checkpoint, root / args.eval_manifest],
            seed=20260729,
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
