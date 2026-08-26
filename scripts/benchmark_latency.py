#!/usr/bin/env python3
"""Benchmark codec stages using preregistered, device-synchronized timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import librosa
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.latency_benchmark import benchmark_callable, hardware_metadata
from whisper_adapt.models.audio_codec import AudioVQVAE
from whisper_adapt.reproducibility import collect_provenance, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-ms", type=int, nargs="+", default=[200, 500, 1000])
    parser.add_argument("--eval-manifest", default="data/med_dictate_eval/eval_en_manifest.parquet")
    parser.add_argument("--include-full-clips", action="store_true")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = (root / args.checkpoint).resolve()
    device = torch.device(args.device)
    model = AudioVQVAE.from_checkpoint(checkpoint, map_location=device).to(device).eval()
    frame = pd.read_parquet(root / args.eval_manifest)
    clips = []
    for row in frame.to_dict("records"):
        values, _ = librosa.load(root / row["path"], sr=model.cfg.sample_rate, mono=True)
        clips.append((str(row["id"]), torch.tensor(values, device=device).unsqueeze(0)))
    conditions = []
    with torch.inference_mode():
        labels = [(f"{value}ms", value) for value in args.chunk_ms]
        if args.include_full_clips: labels.append(("full_clip", None))
        for clip_id, full_audio in clips:
            for label, chunk_ms in labels:
                if chunk_ms is None:
                    audio = full_audio
                else:
                    samples = round(model.cfg.sample_rate * chunk_ms / 1000)
                    audio = torch.nn.functional.pad(full_audio[..., :samples], (0, max(0, samples-full_audio.shape[-1])))
                duration = audio.shape[-1] / model.cfg.sample_rate
                latent = model.encode(audio); quantized, _ = model.quantize(latent)
                stages = {"encode":lambda:model.encode(audio),"decode":lambda:model.decode(quantized,target_length=audio.shape[-1]),"end_to_end":lambda:model.reconstruct(audio)}
                for stage,function in stages.items():
                    result=benchmark_callable(function,device=device,audio_duration_seconds=duration,warmup_iterations=args.warmups,timed_iterations=args.iterations)
                    conditions.append({"clip_id":clip_id,"stage":stage,"condition":label,**result.to_dict()})
    report = {
        "schema_version": 1,
        "benchmark_scope": "batch-model chunk simulation; not production streaming",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "hardware": hardware_metadata(device),
        "n_clips": len(clips),
        "conditions": conditions,
        "provenance": collect_provenance(
            repo_root=root, arguments=vars(args), input_files=[checkpoint], seed=None
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
