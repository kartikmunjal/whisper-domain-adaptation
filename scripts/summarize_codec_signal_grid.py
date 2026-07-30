#!/usr/bin/env python3
"""Aggregate five-seed codec signal results and select WER representatives blind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
        "--output", default="experiments/results/codec_medical/signal_summary.json"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    root = Path(args.results_dir)
    rng = np.random.default_rng(20260729)
    cells = []
    for rate in (300, 400, 500):
        for quantizer in ("vq", "fsq"):
            reports = []
            for seed in SEEDS:
                path = root / f"{quantizer}_{rate}bps" / f"seed_{seed}" / "report.json"
                reports.append(json.loads(path.read_text()))
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
