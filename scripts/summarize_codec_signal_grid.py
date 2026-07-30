#!/usr/bin/env python3
"""Aggregate five-seed codec signal results and select WER representatives blind."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from whisper_adapt.models.audio_codec import AudioVQVAE

SEEDS = (11, 22, 33, 44, 55)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n: int) -> list[float]:
    means = np.array([
        rng.choice(values, len(values), replace=True).mean() for _ in range(n)
    ])
    return np.quantile(means, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results/codec_medical")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/codec_rate_grid"
    )
    parser.add_argument(
        "--output", default="experiments/results/codec_medical/signal_summary.json"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    root = Path(args.results_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    rng = np.random.default_rng(20260729)
    cells = []
    for rate in (300, 400, 500):
        for quantizer in ("vq", "fsq"):
            reports = []
            training_runs = []
            for seed in SEEDS:
                path = root / f"{quantizer}_{rate}bps" / f"seed_{seed}" / "report.json"
                reports.append(json.loads(path.read_text()))
                run_path = (
                    checkpoint_root
                    / f"{quantizer}_{rate}bps"
                    / f"seed_{seed}"
                    / "run.json"
                )
                training_runs.append(json.loads(run_path.read_text()))
            batch_sizes = {
                run["provenance"]["arguments"]["batch_size"] for run in training_runs
            }
            epochs = {run["epochs"] for run in training_runs}
            train_counts = {run["n_train_clips"] for run in training_runs}
            if len(batch_sizes) != 1 or len(epochs) != 1 or len(train_counts) != 1:
                raise RuntimeError(
                    f"Training-budget mismatch within {quantizer}_{rate}bps"
                )
            checkpoint = (
                checkpoint_root
                / f"{quantizer}_{rate}bps"
                / f"seed_{SEEDS[0]}"
                / "codec.pt"
            )
            model = AudioVQVAE.from_checkpoint(checkpoint, map_location="cpu")
            trainable_parameters = sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            )
            batch_size = next(iter(batch_sizes))
            n_epochs = next(iter(epochs))
            n_train = next(iter(train_counts))
            optimizer_steps = math.ceil(n_train / batch_size) * n_epochs
            values = np.array([report["si_sdr_db"]["mean"] for report in reports])
            median = float(np.median(values))
            selected_index = min(
                range(len(SEEDS)),
                key=lambda index: (abs(values[index] - median), SEEDS[index]),
            )
            empirical = np.array([report["empirical_bitrate_bps"] for report in reports])
            cells.append({
                "quantizer": quantizer,
                "nominal_bitrate_bps": rate,
                "n_trials": len(SEEDS),
                "seeds": list(SEEDS),
                "trainable_parameters": trainable_parameters,
                "optimizer_steps_per_trial": optimizer_steps,
                "training_budget": {
                    "n_train_clips": n_train,
                    "batch_size": batch_size,
                    "epochs": n_epochs,
                },
                "si_sdr_db": {
                    "mean": float(values.mean()),
                    "trial_values": values.tolist(),
                    "trial_bootstrap_95_ci": bootstrap_mean(
                        values, rng, args.bootstrap_resamples
                    ),
                },
                "empirical_bitrate_bps": {
                    "mean": float(empirical.mean()),
                    "trial_values": empirical.tolist(),
                    "trial_bootstrap_95_ci": bootstrap_mean(
                        empirical, rng, args.bootstrap_resamples
                    ),
                },
                "wer_representative_seed": SEEDS[selected_index],
                "wer_selection_rule": (
                    "closest mean SI-SDR to the five-seed cell median; selected "
                    "before and without ASR WER"
                ),
            })
    summary = {
        "schema_version": 1,
        "n_trials_per_cell": 5,
        "cells": cells,
        "selection_uses_wer": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
