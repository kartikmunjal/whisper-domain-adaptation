#!/usr/bin/env python3
"""Benchmark codec stages using preregistered, device-synchronized timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.latency_benchmark import benchmark_callable, hardware_metadata
from whisper_adapt.models.audio_codec import AudioVQVAE
from whisper_adapt.reproducibility import collect_provenance, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-ms", type=int, nargs="+", default=[200, 500, 1000])
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
    conditions = []
    with torch.inference_mode():
        for chunk_ms in args.chunk_ms:
            duration = chunk_ms / 1000
            samples = round(model.cfg.sample_rate * duration)
            audio = torch.zeros(1, samples, device=device)
            latent = model.encode(audio)
            quantized, _ = model.quantize(latent)
            stages = {
                "encode": lambda: model.encode(audio),
                "decode": lambda: model.decode(quantized, target_length=samples),
                "end_to_end": lambda: model.reconstruct(audio),
            }
            for stage, function in stages.items():
                result = benchmark_callable(
                    function,
                    device=device,
                    audio_duration_seconds=duration,
                    warmup_iterations=args.warmups,
                    timed_iterations=args.iterations,
                )
                conditions.append({"stage": stage, "chunk_ms": chunk_ms, **result.to_dict()})
    report = {
        "schema_version": 1,
        "benchmark_scope": "batch-model chunk simulation; not production streaming",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "hardware": hardware_metadata(device),
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
