#!/usr/bin/env python3
"""Compute original-to-codec ΔWER and generate the rate-distortion plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from whisper_adapt.evaluation.wer import (
    DomainWERAnalyzer,
    load_domain_vocab,
    paired_bootstrap_difference_ci,
)

SEEDS = (11, 22, 33, 44, 55)
METRICS = ("overall", "domain_terms", "common_terms")


def paired_metric(left: dict, right: dict, metric: str, analyzer, seed: int, n: int):
    if [x["id"] for x in left["predictions"]] != [x["id"] for x in right["predictions"]]:
        raise RuntimeError("Original and codec prediction IDs are not paired")
    references = [x["reference"] for x in left["predictions"]]
    mask = [analyzer._contains_domain_term(ref) for ref in references]
    keep = (
        [True] * len(mask) if metric == "overall"
        else mask if metric == "domain_terms"
        else [not value for value in mask]
    )
    refs = [x for x, flag in zip(references, keep) if flag]
    left_hyps = [
        x["hypothesis"] for x, flag in zip(left["predictions"], keep) if flag
    ]
    right_hyps = [
        x["hypothesis"] for x, flag in zip(right["predictions"], keep) if flag
    ]
    return paired_bootstrap_difference_ci(
        refs, left_hyps, right_hyps, n_resamples=n, seed=seed
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results/codec_medical")
    parser.add_argument("--domain-vocab", default="configs/medical_terms.txt")
    parser.add_argument(
        "--output", default="experiments/results/codec_medical/wer_summary.json"
    )
    parser.add_argument(
        "--plot-output", default="experiments/results/codec_medical/rate_distortion_wer.png"
    )
    parser.add_argument(
        "--markdown-output", default="experiments/results/codec_medical/wer_summary.md"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    root = Path(args.results_dir)
    signal = json.loads((root / "signal_summary.json").read_text())
    analyzer = DomainWERAnalyzer(load_domain_vocab(args.domain_vocab))
    original_base = json.loads((root / "asr" / "original" / "baseline.json").read_text())
    original_adapted = {
        seed: json.loads((root / "asr" / "original" / f"seed_{seed}.json").read_text())
        for seed in SEEDS
    }
    rng = np.random.default_rng(20260729)
    cells = []
    for cell in signal["cells"]:
        name = f"{cell['quantizer']}_{cell['nominal_bitrate_bps']}bps"
        representative_index = cell["seeds"].index(cell["wer_representative_seed"])
        representative_empirical_bitrate = cell["empirical_bitrate_bps"][
            "trial_values"
        ][representative_index]
        codec_base = json.loads((root / "asr" / name / "baseline.json").read_text())
        codec_adapted = {
            seed: json.loads((root / "asr" / name / f"seed_{seed}.json").read_text())
            for seed in SEEDS
        }
        metrics = {}
        for metric in METRICS:
            base = paired_metric(
                original_base, codec_base, metric, analyzer, 20260729,
                args.bootstrap_resamples,
            )
            adapted_trials = [
                paired_metric(
                    original_adapted[seed], codec_adapted[seed], metric,
                    analyzer, seed, args.bootstrap_resamples,
                )
                for seed in SEEDS
            ]
            values = np.asarray([item["estimate"] for item in adapted_trials])
            means = np.asarray([
                rng.choice(values, len(values), replace=True).mean()
                for _ in range(args.bootstrap_resamples)
            ])
            metrics[metric] = {
                "base_whisper": {"n_trials": 1, **base},
                "adapted_whisper": {
                    "n_trials": 5,
                    "trial_values": values.tolist(),
                    "mean_delta_wer": float(values.mean()),
                    "trial_bootstrap_95_ci": np.quantile(
                        means, [0.025, 0.975]
                    ).tolist(),
                    "per_trial_clip_bootstrap": adapted_trials,
                },
            }
        cells.append({
            **cell,
            "condition": name,
            "wer_representative_empirical_bitrate_bps": (
                representative_empirical_bitrate
            ),
            "delta_wer": metrics,
        })
    summary = {
        "schema_version": 1,
        "delta_definition": "codec reconstruction WER minus original-audio WER",
        "cells": cells,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for quantizer, color in (("vq", "#1f77b4"), ("fsq", "#d62728")):
        subset = [cell for cell in cells if cell["quantizer"] == quantizer]
        rates = [
            cell["wer_representative_empirical_bitrate_bps"] for cell in subset
        ]
        base = [cell["delta_wer"]["overall"]["base_whisper"]["estimate"] * 100 for cell in subset]
        base_low = [
            (cell["delta_wer"]["overall"]["base_whisper"]["estimate"]
             - cell["delta_wer"]["overall"]["base_whisper"]["ci_low"]) * 100
            for cell in subset
        ]
        base_high = [
            (cell["delta_wer"]["overall"]["base_whisper"]["ci_high"]
             - cell["delta_wer"]["overall"]["base_whisper"]["estimate"]) * 100
            for cell in subset
        ]
        adapted = [
            cell["delta_wer"]["overall"]["adapted_whisper"]["mean_delta_wer"] * 100
            for cell in subset
        ]
        adapted_low = [
            (cell["delta_wer"]["overall"]["adapted_whisper"]["mean_delta_wer"]
             - cell["delta_wer"]["overall"]["adapted_whisper"]["trial_bootstrap_95_ci"][0]) * 100
            for cell in subset
        ]
        adapted_high = [
            (cell["delta_wer"]["overall"]["adapted_whisper"]["trial_bootstrap_95_ci"][1]
             - cell["delta_wer"]["overall"]["adapted_whisper"]["mean_delta_wer"]) * 100
            for cell in subset
        ]
        axes[0].errorbar(
            rates, base, yerr=[base_low, base_high], marker="o",
            capsize=3, label=quantizer.upper(), color=color,
        )
        axes[1].errorbar(
            rates, adapted, yerr=[adapted_low, adapted_high], marker="o",
            capsize=3, label=quantizer.upper(), color=color,
        )
    for axis, title in zip(axes, ("Frozen Whisper-small", "Medical LoRA mean, N=5")):
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Measured entropy rate of WER representative (bps)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("ΔWER after codec (percentage points)")
    figure.tight_layout()
    plot = Path(args.plot_output)
    plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot, dpi=180)
    plt.close(figure)

    labels = {
        "overall": "Overall",
        "domain_terms": "Domain",
        "common_terms": "Common",
    }
    lines = [
        "# Medical WER after codec reconstruction",
        "",
        "| Codec | Nominal rate | Representative measured rate | SI-SDR, five-seed mean (95% trial CI) | Split | Base ΔWER (95% clip CI) | Adapted mean ΔWER (95% trial CI) | N_trials |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for cell in cells:
        signal_ci = cell["si_sdr_db"]["trial_bootstrap_95_ci"]
        for metric in METRICS:
            base = cell["delta_wer"][metric]["base_whisper"]
            adapted = cell["delta_wer"][metric]["adapted_whisper"]
            adapted_ci = adapted["trial_bootstrap_95_ci"]
            lines.append(
                f"| {cell['quantizer'].upper()} | "
                f"{cell['nominal_bitrate_bps']:.0f} bps | "
                f"{cell['wer_representative_empirical_bitrate_bps']:.2f} bps | "
                f"{cell['si_sdr_db']['mean']:.2f} dB "
                f"({signal_ci[0]:.2f}–{signal_ci[1]:.2f}) | "
                f"{labels[metric]} | "
                f"{base['estimate'] * 100:+.2f} pp "
                f"({base['ci_low'] * 100:+.2f}–{base['ci_high'] * 100:+.2f}) | "
                f"{adapted['mean_delta_wer'] * 100:+.2f} pp "
                f"({adapted_ci[0] * 100:+.2f}–{adapted_ci[1] * 100:+.2f}) | "
                f"{adapted['n_trials']} |"
            )
    lines.extend([
        "",
        "ΔWER is reconstructed-audio WER minus paired original-audio WER.",
        "WER representatives were selected from SI-SDR before ASR evaluation; no model or seed was selected using WER.",
        "Nominal rate is the configured upper-bound design point. Measured entropy rate exposes codebook collapse and is used on the plot x-axis.",
        "",
        "Generated by `scripts/summarize_codec_wer.py`.",
        "",
    ])
    markdown = Path(args.markdown_output)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
