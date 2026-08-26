#!/usr/bin/env python3
"""Reconstruct held-out audio and report continuous-codec signal outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reconstruct_codec_eval import bootstrap_mean, log_mel_l1_db, si_sdr
from whisper_adapt.continuous_codec import ContinuousAudioVAE, uniform_quantize
from whisper_adapt.reproducibility import collect_provenance, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--quantization-bits", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    parser.add_argument("--metadata-header-bits", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = (root / args.checkpoint).resolve()
    output_dir = (root / args.output_dir).resolve()
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = ContinuousAudioVAE.from_checkpoint(checkpoint, map_location=device).to(device).eval()
    frame = pd.read_parquet(root / args.eval_manifest)
    rows = []
    conditions = [("posterior_mean", None)] + [
        (f"quantized_{bits}bit", bits) for bits in args.quantization_bits
    ]
    condition_rows = {name: [] for name, _ in conditions}
    with torch.inference_mode():
        for row_number, row in enumerate(frame.to_dict("records")):
            reference, _ = librosa.load(root / row["path"], sr=model.cfg.sample_rate, mono=True)
            audio = torch.tensor(reference, dtype=torch.float32, device=device).unsqueeze(0)
            identifier = str(row.get("id", row_number))
            chunk_samples = round(args.chunk_seconds * model.cfg.sample_rate)
            kl_sum = 0.0; frame_count = 0; saturation_by_bits = {bits: [] for bits in args.quantization_bits}
            for start in range(0, len(reference), chunk_samples):
                mean, log_variance = model.encode_distribution(audio[..., start:start + chunk_samples])
                per_frame = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(-1)
                kl_sum += float(per_frame.sum().cpu()); frame_count += per_frame.numel()
                for bits in args.quantization_bits:
                    _, _, sat = uniform_quantize(mean, bits, model.config.quantization_clip)
                    saturation_by_bits[bits].append((float(sat.cpu()), mean.numel()))
            kl_value = kl_sum / frame_count
            duration = len(reference) / model.cfg.sample_rate
            for condition, bits in conditions:
                condition_wav_dir = wav_dir / condition
                condition_wav_dir.mkdir(parents=True, exist_ok=True)
                reconstruction = model.reconstruct_chunked(
                    audio,
                    quantization_bits=bits,
                    chunk_samples=chunk_samples,
                    overlap_samples=round(args.overlap_seconds * model.cfg.sample_rate),
                )[0, 0].cpu().numpy()
                wav_path = condition_wav_dir / f"{identifier}.wav"
                sf.write(wav_path, reconstruction, model.cfg.sample_rate, subtype="PCM_16")
                saturation = 0.0
                payload_bps = None
                effective_bps = None
                if bits is not None:
                    saturation = sum(v*n for v,n in saturation_by_bits[bits]) / sum(n for _,n in saturation_by_bits[bits])
                    payload_bps = model.cfg.frame_rate_hz * model.config.bottleneck_dim * bits
                    effective_bps = payload_bps + args.metadata_header_bits / duration
                condition_rows[condition].append({
                    **row,
                    "original_path": row["path"],
                    "path": str(wav_path.relative_to(root)),
                    "condition": condition,
                    "quantization_bits": bits,
                    "payload_bitrate_bps": payload_bps,
                    "effective_bitrate_bps": effective_bps,
                    "si_sdr_db": si_sdr(reference, reconstruction),
                    "log_mel_l1_db": log_mel_l1_db(reference, reconstruction, model.cfg.sample_rate),
                    "posterior_kl_per_frame": kl_value,
                    "latent_saturation_fraction": saturation,
                })
    reports = {}
    for condition, bits in conditions:
        result = pd.DataFrame(condition_rows[condition])
        condition_dir = output_dir / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(condition_dir / "reconstructed_manifest.parquet", index=False)
        si_sdr_values = result.si_sdr_db.to_numpy()
        mel_values = result.log_mel_l1_db.to_numpy()
        reports[condition] = {
            "quantization_bits": bits,
            "payload_bitrate_bps": None if bits is None else float(result.payload_bitrate_bps.mean()),
            "effective_bitrate_bps_mean": None if bits is None else float(result.effective_bitrate_bps.mean()),
            "si_sdr_db": {"mean": float(si_sdr_values.mean()), "clip_bootstrap_95_ci": bootstrap_mean(si_sdr_values, args.bootstrap_resamples)},
            "log_mel_l1_db": {"mean": float(mel_values.mean()), "clip_bootstrap_95_ci": bootstrap_mean(mel_values, args.bootstrap_resamples)},
            "posterior_kl_per_frame": float(result.posterior_kl_per_frame.mean()),
            "latent_saturation_fraction": float(result.latent_saturation_fraction.mean()),
        }
    report = {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(checkpoint),
        "n_clips": len(result),
        "conditions": reports,
        "bitrate_assumption": f"fixed-width payload plus {args.metadata_header_bits}-bit per-clip header; container overhead excluded",
        "provenance": collect_provenance(
            repo_root=root,
            arguments=vars(args),
            input_files=[checkpoint, root / args.eval_manifest],
            seed=20260729,
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
