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
    with torch.inference_mode():
        for row_number, row in enumerate(frame.to_dict("records")):
            reference, _ = librosa.load(root / row["path"], sr=model.cfg.sample_rate, mono=True)
            audio = torch.tensor(reference, dtype=torch.float32, device=device).unsqueeze(0)
            mean, _ = model.encode_distribution(audio)
            quantized, _, saturation = uniform_quantize(
                mean, model.config.quantization_bits, model.config.quantization_clip
            )
            reconstruction = model.decode(quantized, len(reference))[0, 0].cpu().numpy()
            identifier = str(row.get("id", row_number))
            wav_path = wav_dir / f"{identifier}.wav"
            sf.write(wav_path, reconstruction, model.cfg.sample_rate, subtype="PCM_16")
            rows.append({
                **row,
                "original_path": row["path"],
                "path": str(wav_path.relative_to(root)),
                "si_sdr_db": si_sdr(reference, reconstruction),
                "log_mel_l1_db": log_mel_l1_db(reference, reconstruction, model.cfg.sample_rate),
                "latent_saturation_fraction": float(saturation.cpu()),
            })
    result = pd.DataFrame(rows)
    result.to_parquet(output_dir / "reconstructed_manifest.parquet", index=False)
    si_sdr_values = result.si_sdr_db.to_numpy()
    mel_values = result.log_mel_l1_db.to_numpy()
    report = {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(checkpoint),
        "n_clips": len(result),
        "fixed_width_bitrate_bps": model.fixed_width_bitrate_bps,
        "bitrate_assumption": "fixed-width codes; container/framing overhead excluded",
        "si_sdr_db": {"mean": float(si_sdr_values.mean()), "clip_bootstrap_95_ci": bootstrap_mean(si_sdr_values, args.bootstrap_resamples)},
        "log_mel_l1_db": {"mean": float(mel_values.mean()), "clip_bootstrap_95_ci": bootstrap_mean(mel_values, args.bootstrap_resamples)},
        "latent_saturation_fraction": float(result.latent_saturation_fraction.mean()),
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
