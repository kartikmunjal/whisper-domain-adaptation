#!/usr/bin/env python3
"""Compare the pinned EnCodec anchor with the closest custom codec cells."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.evaluation.wer import (
    DomainWERAnalyzer,
    load_domain_vocab,
    paired_bootstrap_difference_ci,
)


SEEDS = (11, 22, 33, 44, 55)
METRICS = ("overall", "domain_terms", "common_terms")
BOOTSTRAP_SEED = 20260801
N_BOOTSTRAP = 10_000


def trial_mean_ci(values, n_resamples=N_BOOTSTRAP):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(n_resamples)]
    )
    return [float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()]


def paired_ratio_ci(numerators, denominators, n_resamples=N_BOOTSTRAP):
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if len(numerators) != len(denominators) or np.any(denominators <= 0):
        raise ValueError("paired WER ratios require positive aligned trials")
    return trial_mean_ci(numerators / denominators, n_resamples)


def paired_metric(left, right, metric, analyzer, seed, n_resamples):
    if [row["id"] for row in left["predictions"]] != [
        row["id"] for row in right["predictions"]
    ]:
        raise RuntimeError("original and reconstructed prediction IDs are not paired")
    references = [row["reference"] for row in left["predictions"]]
    domain_mask = [analyzer._contains_domain_term(text) for text in references]
    keep = (
        [True] * len(references)
        if metric == "overall"
        else domain_mask
        if metric == "domain_terms"
        else [not value for value in domain_mask]
    )
    refs = [text for text, flag in zip(references, keep) if flag]
    left_hypotheses = [
        row["hypothesis"] for row, flag in zip(left["predictions"], keep) if flag
    ]
    right_hypotheses = [
        row["hypothesis"] for row, flag in zip(right["predictions"], keep) if flag
    ]
    return paired_bootstrap_difference_ci(
        refs,
        left_hypotheses,
        right_hypotheses,
        n_resamples=n_resamples,
        seed=seed,
    )


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encodec-dir", default="experiments/results/codec_medical_encodec"
    )
    parser.add_argument(
        "--custom-dir", default="experiments/results/codec_medical_corrective"
    )
    parser.add_argument("--domain-vocab", default="configs/medical_terms.txt")
    parser.add_argument(
        "--output", default="experiments/results/codec_medical_encodec/summary.json"
    )
    parser.add_argument(
        "--markdown-output",
        default="experiments/results/codec_medical_encodec/REPORT.md",
    )
    parser.add_argument(
        "--plot-output",
        default="experiments/results/codec_medical_encodec/rate_distortion_external.png",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    encodec_root = Path(args.encodec_dir)
    custom_root = Path(args.custom_dir)
    signal = json.loads((encodec_root / "signal_report.json").read_text())
    custom_signal = json.loads((custom_root / "signal_summary.json").read_text())
    custom_wer = json.loads((custom_root / "wer_summary.json").read_text())
    analyzer = DomainWERAnalyzer(load_domain_vocab(args.domain_vocab))

    original_base = json.loads(
        (encodec_root / "asr" / "original" / "baseline.json").read_text()
    )
    original_adapted = {
        seed: json.loads(
            (encodec_root / "asr" / "original" / f"seed_{seed}.json").read_text()
        )
        for seed in SEEDS
    }
    encodec_base = json.loads(
        (encodec_root / "asr" / "encodec_1.5kbps" / "baseline.json").read_text()
    )
    encodec_adapted = {
        seed: json.loads(
            (encodec_root / "asr" / "encodec_1.5kbps" / f"seed_{seed}.json").read_text()
        )
        for seed in SEEDS
    }

    encodec_metrics = {}
    for metric in METRICS:
        base_delta = paired_metric(
            original_base,
            encodec_base,
            metric,
            analyzer,
            BOOTSTRAP_SEED,
            args.bootstrap_resamples,
        )
        adapted_deltas = [
            paired_metric(
                original_adapted[seed],
                encodec_adapted[seed],
                metric,
                analyzer,
                seed,
                args.bootstrap_resamples,
            )
            for seed in SEEDS
        ]
        absolute_trials = [encodec_adapted[seed]["wer"][metric] for seed in SEEDS]
        delta_trials = [row["estimate"] for row in adapted_deltas]
        encodec_metrics[metric] = {
            "base_whisper": {
                "absolute_wer": {
                    "estimate": encodec_base["wer"][metric],
                    "clip_bootstrap_95_ci": [
                        encodec_base["uncertainty"][metric]["ci_low"],
                        encodec_base["uncertainty"][metric]["ci_high"],
                    ],
                    "n_resamples": encodec_base["uncertainty"][metric][
                        "n_resamples"
                    ],
                },
                "delta_wer": base_delta,
            },
            "adapted_whisper": {
                "n_trials": len(SEEDS),
                "absolute_wer_trial_values": absolute_trials,
                "absolute_wer_mean_ci95": trial_mean_ci(
                    absolute_trials, args.bootstrap_resamples
                ),
                "delta_wer_trial_values": delta_trials,
                "delta_wer_mean_ci95": trial_mean_ci(
                    delta_trials, args.bootstrap_resamples
                ),
                "per_trial_clip_bootstrap": adapted_deltas,
            },
        }

    custom_cells = {}
    comparisons = {}
    for quantizer in ("vq", "fsq"):
        condition = f"{quantizer}_500bps"
        signal_cell = next(
            row
            for row in custom_signal["cells"]
            if row["quantizer"] == quantizer and row["nominal_bitrate_bps"] == 500
        )
        wer_cell = next(row for row in custom_wer["cells"] if row["condition"] == condition)
        custom_adapted = {
            seed: json.loads(
                (custom_root / "asr" / condition / f"seed_{seed}.json").read_text()
            )
            for seed in SEEDS
        }
        custom_base = json.loads(
            (custom_root / "asr" / condition / "baseline.json").read_text()
        )
        custom_cells[condition] = {
            "quantizer": quantizer,
            "nominal_bitrate_bps": 500,
            "empirical_bitrate_bps": signal_cell["empirical_bitrate_bps"],
            "entropy_utilization": {
                "mean_ci95": trial_mean_ci(
                    np.asarray(signal_cell["empirical_bitrate_bps"]["trial_values"])
                    / 500.0,
                    args.bootstrap_resamples,
                ),
                "definition": "empirical entropy rate divided by nominal fixed-width rate",
            },
            "si_sdr_db": signal_cell["si_sdr_db"],
            "log_mel_l1_db": signal_cell["log_mel_l1_db"],
            "wer": {},
        }
        comparisons[condition] = {}
        for metric in METRICS:
            absolute_trials = [custom_adapted[seed]["wer"][metric] for seed in SEEDS]
            encodec_trials = encodec_metrics[metric]["adapted_whisper"][
                "absolute_wer_trial_values"
            ]
            custom_cells[condition]["wer"][metric] = {
                "base_absolute_wer": {
                    "estimate": custom_base["wer"][metric],
                    "clip_bootstrap_95_ci": [
                        custom_base["uncertainty"][metric]["ci_low"],
                        custom_base["uncertainty"][metric]["ci_high"],
                    ],
                    "n_resamples": custom_base["uncertainty"][metric][
                        "n_resamples"
                    ],
                },
                "adapted_absolute_wer_trial_values": absolute_trials,
                "adapted_absolute_wer_mean_ci95": trial_mean_ci(
                    absolute_trials, args.bootstrap_resamples
                ),
                "base_delta_wer": wer_cell["delta_wer"][metric]["base_whisper"],
                "adapted_delta_wer": wer_cell["delta_wer"][metric][
                    "adapted_whisper"
                ],
            }
            comparisons[condition][metric] = {
                "custom_minus_encodec_absolute_wer_mean_ci95": trial_mean_ci(
                    np.asarray(absolute_trials) - np.asarray(encodec_trials),
                    args.bootstrap_resamples,
                ),
                "custom_over_encodec_absolute_wer_ratio_mean_ci95": paired_ratio_ci(
                    absolute_trials, encodec_trials, args.bootstrap_resamples
                ),
            }

    summary = {
        "schema_version": 1,
        "protocol": {
            "n_clips": signal["n_clips"],
            "adapter_seeds": list(SEEDS),
            "bootstrap_resamples": args.bootstrap_resamples,
            "delta_definition": "reconstructed-audio WER minus paired original-audio WER",
            "comparison_rate_warning": (
                "EnCodec is 1500 nominal bps; custom cells are 500 nominal bps. "
                "This is not a matched-rate ranking."
            ),
        },
        "encodec": {
            "model_id": signal["model_id"],
            "revision": signal["revision"],
            "weight_sha256": signal["weight_sha256"],
            "nominal_bitrate_bps": signal["utilization"]["nominal_fixed_width_bps"],
            "empirical_bitrate_bps": signal["empirical_bitrate_bps"],
            "utilization": signal["utilization"],
            "si_sdr_db": signal["si_sdr_db"],
            "log_mel_l1_db": signal["log_mel_l1_db"],
            "wer": encodec_metrics,
        },
        "custom_cells": custom_cells,
        "comparisons": comparisons,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for quantizer, color in (("vq", "#1f77b4"), ("fsq", "#d62728")):
        cells = [row for row in custom_signal["cells"] if row["quantizer"] == quantizer]
        cells.sort(key=lambda row: row["nominal_bitrate_bps"])
        wer_cells = {
            row["nominal_bitrate_bps"]: row
            for row in custom_wer["cells"]
            if row["quantizer"] == quantizer
        }
        rates = [row["nominal_bitrate_bps"] for row in cells]
        axes[0].plot(rates, [row["si_sdr_db"]["mean"] for row in cells], "o-", color=color, label=quantizer.upper())
        axes[1].plot(rates, [row["log_mel_l1_db"]["mean"] for row in cells], "o-", color=color, label=quantizer.upper())
        axes[2].plot(
            rates,
            [100 * wer_cells[rate]["delta_wer"]["overall"]["adapted_whisper"]["mean_delta_wer"] for rate in rates],
            "o-",
            color=color,
            label=quantizer.upper(),
        )
    encodec_rate = summary["encodec"]["nominal_bitrate_bps"]
    axes[0].scatter([encodec_rate], [signal["si_sdr_db"]["mean"]], marker="*", s=160, color="#2ca02c", label="EnCodec")
    axes[1].scatter([encodec_rate], [signal["log_mel_l1_db"]["mean"]], marker="*", s=160, color="#2ca02c", label="EnCodec")
    axes[2].scatter([encodec_rate], [100 * encodec_metrics["overall"]["adapted_whisper"]["delta_wer_mean_ci95"][0]], marker="*", s=160, color="#2ca02c", label="EnCodec")
    for axis, title, ylabel in zip(
        axes,
        ("Signal fidelity", "Spectral distortion", "ASR task cost"),
        ("SI-SDR (dB), higher is better", "Log-mel L1 (dB), lower is better", "Adapted ΔWER (pp), lower is better"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Nominal fixed-width bitrate (bps)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("EnCodec is an external 1.5 kbps anchor, not a matched-rate point")
    figure.tight_layout()
    figure.savefig(args.plot_output, dpi=180)
    plt.close(figure)

    def interval(values, scale=1.0, suffix=""):
        return (
            f"{values[0] * scale:.2f}{suffix} "
            f"[{values[1] * scale:.2f}, {values[2] * scale:.2f}]"
        )

    lines = [
        "# External EnCodec medical-audio benchmark",
        "",
        "The pretrained EnCodec point is 1.5 kbps, three times the nominal rate of the closest custom cells. This is an external quality anchor, **not a matched-rate ranking**.",
        "",
        "## Signal and utilization",
        "",
        "| Codec | Nominal rate | Empirical entropy rate | Entropy utilization | SI-SDR | Log-mel L1 | Uncertainty unit |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    encodec_util = signal["utilization"]["entropy_utilization"]
    lines.append(
        f"| EnCodec 24 kHz | {signal['utilization']['nominal_fixed_width_bps']:.0f} bps | "
        f"{signal['empirical_bitrate_bps']['pooled']:.1f} bps | {encodec_util * 100:.1f}% | "
        f"{interval([signal['si_sdr_db']['mean'], *signal['si_sdr_db']['clip_bootstrap_95_ci']], suffix=' dB')} | "
        f"{interval([signal['log_mel_l1_db']['mean'], *signal['log_mel_l1_db']['clip_bootstrap_95_ci']], suffix=' dB')} | 24 clips |"
    )
    for condition in ("vq_500bps", "fsq_500bps"):
        cell = custom_cells[condition]
        utilization_ci = cell["entropy_utilization"]["mean_ci95"]
        si = cell["si_sdr_db"]
        mel = cell["log_mel_l1_db"]
        lines.append(
            f"| {condition.upper()} | 500 bps | {cell['empirical_bitrate_bps']['mean']:.1f} bps | "
            f"{interval(utilization_ci, scale=100, suffix='%')} | "
            f"{interval([si['mean'], *si['trial_bootstrap_95_ci']], suffix=' dB')} | "
            f"{interval([mel['mean'], *mel['trial_bootstrap_95_ci']], suffix=' dB')} | 5 training seeds |"
        )
    lines.extend(
        [
            "",
            "## Medical-ASR cost",
            "",
            "| Codec | Split | Base reconstructed WER [95% CI] | Base ΔWER [95% CI] | Adapted reconstructed WER mean [95% CI] | Adapted ΔWER mean [95% CI] | N_trials |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {"overall": "Overall", "domain_terms": "Domain", "common_terms": "Common"}
    for metric in METRICS:
        row = encodec_metrics[metric]
        base_absolute = row["base_whisper"]["absolute_wer"]
        base_delta = row["base_whisper"]["delta_wer"]
        lines.append(
            f"| EnCodec | {labels[metric]} | "
            f"{interval([base_absolute['estimate'], *base_absolute['clip_bootstrap_95_ci']], scale=100, suffix='%')} | "
            f"{interval([base_delta['estimate'], base_delta['ci_low'], base_delta['ci_high']], scale=100, suffix=' pp')} | "
            f"{interval(row['adapted_whisper']['absolute_wer_mean_ci95'], scale=100, suffix='%')} | "
            f"{interval(row['adapted_whisper']['delta_wer_mean_ci95'], scale=100, suffix=' pp')} | 5 |"
        )
    for condition in ("vq_500bps", "fsq_500bps"):
        for metric in METRICS:
            row = custom_cells[condition]["wer"][metric]
            delta = row["adapted_delta_wer"]
            base_absolute = row["base_absolute_wer"]
            base_delta = row["base_delta_wer"]
            lines.append(
                f"| {condition.upper()} | {labels[metric]} | "
                f"{interval([base_absolute['estimate'], *base_absolute['clip_bootstrap_95_ci']], scale=100, suffix='%')} | "
                f"{interval([base_delta['estimate'], base_delta['ci_low'], base_delta['ci_high']], scale=100, suffix=' pp')} | "
                f"{interval(row['adapted_absolute_wer_mean_ci95'], scale=100, suffix='%')} | "
                f"{interval([delta['mean_delta_wer'], *delta['trial_bootstrap_95_ci']], scale=100, suffix=' pp')} | 5 |"
            )
    lines.extend(
        [
            "",
            "## Bitrate-qualified distance from EnCodec",
            "",
            "Ratios use seed-matched absolute WER from the same five medical adapters. EnCodec receives 3× the nominal bitrate; ratios are descriptive and are not rate-controlled superiority estimates.",
            "",
            "| Custom codec | Split | Custom − EnCodec WER [95% CI] | Custom / EnCodec WER ratio [95% CI] |",
            "|---|---|---:|---:|",
        ]
    )
    for condition in ("vq_500bps", "fsq_500bps"):
        for metric in METRICS:
            row = comparisons[condition][metric]
            lines.append(
                f"| {condition.upper()} | {labels[metric]} | "
                f"{interval(row['custom_minus_encodec_absolute_wer_mean_ci95'], scale=100, suffix=' pp')} | "
                f"{interval(row['custom_over_encodec_absolute_wer_ratio_mean_ci95'], suffix='×')} |"
            )
    lines.extend(
        [
            "",
            "EnCodec codebook utilization is computed from pooled discrete-code entropy; custom utilization is empirical entropy rate divided by each configured fixed-width rate. Signal and WER metrics can disagree and are retained independently.",
            "",
            "Generated by `scripts/summarize_encodec_benchmark.py`.",
            "",
        ]
    )
    Path(args.markdown_output).write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
