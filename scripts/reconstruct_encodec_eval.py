#!/usr/bin/env python3
"""Reconstruct the locked medical set with a pinned pretrained EnCodec model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.reproducibility import collect_provenance, sha256_file

from reconstruct_codec_eval import bootstrap_mean, log_mel_l1_db, si_sdr


MODEL_ID = "facebook/encodec_24khz"
MODEL_REVISION = "c1dbe2ae3f1de713481a3b3e7c47f357092ee040"
TARGET_BANDWIDTH_KBPS = 1.5
BOOTSTRAP_SEED = 20260801


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-manifest", default="data/med_dictate_eval/eval_en_manifest.parquet"
    )
    parser.add_argument(
        "--output-dir", default="experiments/results/codec_medical_encodec"
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--bandwidth-kbps", type=float, default=TARGET_BANDWIDTH_KBPS)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.1)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("cannot compute entropy from empty codes")
    probabilities = np.asarray(list(counts.values()), dtype=float) / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def merge_code_counts(
    totals: list[Counter[int]], audio_codes: torch.Tensor
) -> None:
    """Accumulate `[frames, batch, codebooks, time]` EnCodec codes."""
    codes = audio_codes.detach().cpu()
    if codes.ndim != 4 or codes.shape[1] != 1:
        raise ValueError(f"unexpected EnCodec code shape: {tuple(codes.shape)}")
    flattened = codes.permute(2, 0, 1, 3).reshape(codes.shape[2], -1)
    if len(totals) != flattened.shape[0]:
        raise ValueError("codebook-count container does not match encoded codes")
    for index, values in enumerate(flattened):
        totals[index].update(int(value) for value in values.tolist())


def crossfade_reconstruction(
    model,
    audio: np.ndarray,
    model_sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
    bandwidth_kbps: float,
    device: torch.device,
) -> tuple[np.ndarray, list[Counter[int]]]:
    chunk = int(round(chunk_seconds * model_sample_rate))
    overlap = int(round(overlap_seconds * model_sample_rate))
    if chunk <= 0 or overlap < 0 or overlap >= chunk:
        raise ValueError("require 0 <= overlap < chunk")
    step = chunk - overlap
    output = np.zeros(len(audio), dtype=np.float64)
    weights = np.zeros(len(audio), dtype=np.float64)
    code_counts: list[Counter[int]] | None = None

    for start in range(0, len(audio), step):
        end = min(start + chunk, len(audio))
        values = torch.as_tensor(
            audio[start:end], dtype=torch.float32, device=device
        ).view(1, 1, -1)
        padding_mask = torch.ones(
            (1, values.shape[-1]), dtype=torch.bool, device=device
        )
        with torch.inference_mode():
            encoded = model.encode(
                values,
                padding_mask=padding_mask,
                bandwidth=bandwidth_kbps,
                return_dict=True,
            )
            decoded = model.decode(
                encoded.audio_codes,
                encoded.audio_scales,
                padding_mask=padding_mask,
                last_frame_pad_length=encoded.last_frame_pad_length,
                return_dict=True,
            ).audio_values[0, 0, : end - start]
        reconstruction = decoded.detach().cpu().numpy().astype(np.float64)
        if code_counts is None:
            code_counts = [Counter() for _ in range(encoded.audio_codes.shape[2])]
        merge_code_counts(code_counts, encoded.audio_codes)

        window = np.ones(len(reconstruction), dtype=np.float64)
        fade = min(overlap, len(reconstruction) // 2)
        if fade and start > 0:
            window[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False)
        if fade and end < len(audio):
            window[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False)
        output[start:end] += reconstruction * window
        weights[start:end] += window
        if end == len(audio):
            break
    if code_counts is None or np.any(weights == 0):
        raise RuntimeError("incomplete EnCodec reconstruction")
    return (output / weights).astype(np.float32), code_counts


def codebook_report(
    counts: list[Counter[int]], codebook_size: int, frame_rate: float
) -> dict:
    entropies = [entropy_from_counts(item) for item in counts]
    rows = []
    for index, (counter, entropy) in enumerate(zip(counts, entropies)):
        rows.append(
            {
                "index": index,
                "unique_codes": len(counter),
                "unique_fraction": len(counter) / codebook_size,
                "entropy_bits_per_frame": entropy,
                "entropy_fraction_of_fixed_width": entropy / math.log2(codebook_size),
                "perplexity": 2.0**entropy,
                "n_observations": sum(counter.values()),
            }
        )
    nominal = len(counts) * math.log2(codebook_size) * frame_rate
    empirical = sum(entropies) * frame_rate
    return {
        "n_codebooks": len(counts),
        "codebook_size": codebook_size,
        "frame_rate_hz": frame_rate,
        "nominal_fixed_width_bps": nominal,
        "empirical_entropy_bps": empirical,
        "entropy_utilization": empirical / nominal,
        "codebooks": rows,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = (root / args.eval_manifest).resolve()
    output = (root / args.output_dir).resolve()
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from transformers import EncodecModel

    weight_path = Path(
        hf_hub_download(
            args.model_id, "model.safetensors", revision=args.revision
        )
    )
    config_path = Path(
        hf_hub_download(args.model_id, "config.json", revision=args.revision)
    )
    device = torch.device(args.device)
    model = EncodecModel.from_pretrained(
        args.model_id, revision=args.revision
    ).to(device)
    model.eval()
    model_sample_rate = int(model.config.sampling_rate)
    codebook_size = int(model.config.codebook_size)
    frame_rate = float(model.config.frame_rate)

    frame = pd.read_parquet(manifest_path)
    rows = []
    pooled_counts: list[Counter[int]] | None = None
    for row_index, row in enumerate(frame.to_dict("records")):
        reference, sample_rate = librosa.load(
            root / row["path"], sr=None, mono=True
        )
        model_audio = librosa.resample(
            reference, orig_sr=sample_rate, target_sr=model_sample_rate
        )
        model_reconstruction, clip_counts = crossfade_reconstruction(
            model,
            model_audio,
            model_sample_rate,
            args.chunk_seconds,
            args.overlap_seconds,
            args.bandwidth_kbps,
            device,
        )
        if pooled_counts is None:
            pooled_counts = [Counter() for _ in clip_counts]
        for pooled, clip in zip(pooled_counts, clip_counts):
            pooled.update(clip)
        reconstruction = librosa.resample(
            model_reconstruction,
            orig_sr=model_sample_rate,
            target_sr=sample_rate,
        )
        if len(reconstruction) < len(reference):
            reconstruction = np.pad(reconstruction, (0, len(reference) - len(reconstruction)))
        reconstruction = reconstruction[: len(reference)]
        wav_path = wav_dir / f"{row_index:04d}_{row['id']}.wav"
        sf.write(wav_path, reconstruction, sample_rate, subtype="PCM_16")
        clip_codes = codebook_report(clip_counts, codebook_size, frame_rate)
        rows.append(
            {
                **row,
                "original_path": row["path"],
                "path": str(wav_path.relative_to(root)),
                "reconstructed_sha256": sha256_file(wav_path),
                "si_sdr_db": si_sdr(reference, reconstruction),
                "log_mel_l1_db": log_mel_l1_db(reference, reconstruction, sample_rate),
                "empirical_bitrate_bps": clip_codes["empirical_entropy_bps"],
            }
        )
        print(f"reconstructed {row_index + 1}/{len(frame)}: {row['id']}", flush=True)

    if pooled_counts is None:
        raise RuntimeError("empty evaluation manifest")
    result = pd.DataFrame(rows)
    manifest_output = output / "reconstructed_manifest.parquet"
    result.to_parquet(manifest_output, index=False)
    si_sdr_values = result.si_sdr_db.to_numpy(dtype=float)
    mel_values = result.log_mel_l1_db.to_numpy(dtype=float)
    bitrate_values = result.empirical_bitrate_bps.to_numpy(dtype=float)
    utilization = codebook_report(pooled_counts, codebook_size, frame_rate)
    nominal_requested = args.bandwidth_kbps * 1000.0
    if not math.isclose(
        utilization["nominal_fixed_width_bps"], nominal_requested, rel_tol=0.01
    ):
        raise RuntimeError("encoded code shape does not match requested bandwidth")

    report = {
        "schema_version": 1,
        "condition": "encodec_24khz_1.5kbps",
        "model_id": args.model_id,
        "revision": args.revision,
        "upstream_project": "https://github.com/facebookresearch/encodec",
        "upstream_paper": "https://arxiv.org/abs/2210.13438",
        "weight_file": weight_path.name,
        "weight_sha256": sha256_file(weight_path),
        "config_sha256": sha256_file(config_path),
        "n_clips": len(result),
        "sample_rate_hz": model_sample_rate,
        "target_bandwidth_kbps": args.bandwidth_kbps,
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "utilization": utilization,
        "empirical_bitrate_bps": {
            "pooled": utilization["empirical_entropy_bps"],
            "clip_mean": float(bitrate_values.mean()),
            "clip_bootstrap_95_ci": bootstrap_mean(
                bitrate_values, args.bootstrap_resamples
            ),
            "clip_values": bitrate_values.tolist(),
        },
        "si_sdr_db": {
            "mean": float(si_sdr_values.mean()),
            "clip_bootstrap_95_ci": bootstrap_mean(
                si_sdr_values, args.bootstrap_resamples
            ),
            "clip_values": si_sdr_values.tolist(),
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
                "fmax_hz": 7600,
            },
        },
        "reconstructed_manifest_sha256": sha256_file(manifest_output),
        "provenance": collect_provenance(
            repo_root=root,
            arguments=vars(args),
            input_files=[manifest_path, weight_path, config_path],
            seed=BOOTSTRAP_SEED,
        ),
    }
    (output / "signal_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
