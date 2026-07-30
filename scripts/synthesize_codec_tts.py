#!/usr/bin/env python3
"""Generate held-out waveforms with a trained text-to-codec-token model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.models.audio_codec import AudioVQVAE
from whisper_adapt.models.codec_tts import CodecTokenTTS, encode_text_bytes
from whisper_adapt.reproducibility import collect_provenance, sha256_file


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(len(reference), len(estimate))
    reference = reference[:length].astype(np.float64)
    estimate = estimate[:length].astype(np.float64)
    reference -= reference.mean()
    estimate -= estimate.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12)
    target = scale * reference
    noise = estimate - target
    return float(10 * np.log10(
        (np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)
    ))


def bootstrap_mean_ci(values: list[float], n: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260729)
    means = np.array([
        rng.choice(array, len(array), replace=True).mean() for _ in range(n)
    ])
    return np.quantile(means, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--codec-checkpoint",
        default="checkpoints/codec_rate_grid/vq_400bps/seed_11/codec.pt",
    )
    parser.add_argument("--manifest", default="data/financial_research/test_manifest.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    device = torch.device(args.device)
    tts = CodecTokenTTS.from_checkpoint(root / args.tts_checkpoint, device).to(device).eval()
    codec = AudioVQVAE.from_checkpoint(root / args.codec_checkpoint, device).to(device).eval()
    output = root / args.output_dir
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(root / args.manifest)
    targets = pd.read_parquet(root / "data/codec_tts_tokens/test.parquet")
    target_lengths = dict(zip(targets.id.astype(str), targets.n_codec_tokens))
    rows = []
    records = frame.to_dict("records")
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        encoded = [
            encode_text_bytes(row["sentence"], tts.config.max_text_tokens)
            for row in batch
        ]
        text_ids = torch.zeros(
            len(batch), max(map(len, encoded)), dtype=torch.long, device=device
        )
        for index, ids in enumerate(encoded):
            text_ids[index, :len(ids)] = torch.tensor(ids, device=device)
        generated_batch = tts.generate(
            text_ids, max_new_tokens=args.max_new_tokens
        )
        for row, generated in zip(batch, generated_batch):
            eos = generated.eq(tts.config.audio_eos_id).nonzero()
            if len(eos):
                generated = generated[: int(eos[0])]
            generated = generated[generated.lt(tts.config.codebook_size)]
            if not len(generated):
                raise RuntimeError(f"Model generated no codec tokens for {row['id']}")
            waveform = codec.decode_vq_indices(generated)[0, 0].cpu().numpy()
            reference, _ = librosa.load(
                root / row["path"], sr=codec.cfg.sample_rate, mono=True
            )
            wav_path = wav_dir / f"{row['id']}.wav"
            sf.write(wav_path, waveform, codec.cfg.sample_rate, subtype="PCM_16")
            rows.append({
                **row,
                "edge_tts_path": row["path"],
                "path": str(wav_path.relative_to(root)),
                "n_generated_codec_tokens": len(generated),
                "n_reference_codec_tokens": int(target_lengths[str(row["id"])]),
                "absolute_sequence_length_error": abs(
                    len(generated) - int(target_lengths[str(row["id"])])
                ),
                "si_sdr_db": si_sdr(reference, waveform),
                "terminated_with_eos": bool(len(eos)),
            })
    generated_manifest = output / "generated_manifest.parquet"
    pd.DataFrame(rows).to_parquet(generated_manifest, index=False)
    report = {
        "schema_version": 1,
        "tts_checkpoint": args.tts_checkpoint,
        "tts_sha256": sha256_file(root / args.tts_checkpoint),
        "codec_checkpoint": args.codec_checkpoint,
        "codec_sha256": sha256_file(root / args.codec_checkpoint),
        "source_manifest": args.manifest,
        "source_manifest_sha256": sha256_file(root / args.manifest),
        "n_samples": len(rows),
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "eos_rate": sum(row["terminated_with_eos"] for row in rows) / len(rows),
        "sequence_length_error": {
            "mean_absolute_tokens": float(np.mean([
                row["absolute_sequence_length_error"] for row in rows
            ])),
            "clip_bootstrap_95_ci": bootstrap_mean_ci([
                row["absolute_sequence_length_error"] for row in rows
            ]),
        },
        "si_sdr_db": {
            "mean": float(np.mean([row["si_sdr_db"] for row in rows])),
            "clip_bootstrap_95_ci": bootstrap_mean_ci([
                row["si_sdr_db"] for row in rows
            ]),
        },
        "provenance": collect_provenance(
            repo_root=root,
            arguments=vars(args),
            input_files=[
                root / args.tts_checkpoint,
                root / args.codec_checkpoint,
                root / args.manifest,
                root / "data/codec_tts_tokens/test.parquet",
            ],
            seed=args.seed,
        ),
    }
    (output / "generation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
