#!/usr/bin/env python3
"""Aggregate the preregistered capacity/data and phoneme TTS interventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (11, 22, 33, 44, 55)
METRICS = ("overall", "domain_terms", "common_terms")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260801


def ci(values, n_resamples=N_BOOTSTRAP):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(n_resamples)]
    )
    return [
        float(values.mean()),
        *np.quantile(means, [0.025, 0.975]).tolist(),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="experiments/results/codec_tts_scale_study"
    )
    args = parser.parse_args()
    systems = {}
    for name, path in (
        ("text_forced", "experiments/results/codec_tts_text_only"),
        ("scaled_bytes", "experiments/results/codec_tts_scaled_bytes"),
        ("scaled_phonemes", "experiments/results/codec_tts_scaled_phonemes"),
    ):
        systems[name] = json.loads((Path(path) / "summary.json").read_text())

    piper_dir = Path("experiments/results/piper_lessac_low")
    piper = [json.loads((piper_dir / f"seed_{seed}.json").read_text()) for seed in SEEDS]
    result = {
        "schema_version": 2,
        "n_trials": len(SEEDS),
        "seeds": list(SEEDS),
        "bootstrap": {
            "unit": "training_seed",
            "n_resamples": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "metrics": {},
        "conditioning": {},
    }
    generation_report = piper_dir / "generation_report.json"
    if generation_report.exists():
        result["piper_generation_provenance"] = json.loads(
            generation_report.read_text()
        )

    for metric in METRICS:
        values = {
            name: data["metrics"][metric]["codec_tts_trial_values"]
            for name, data in systems.items()
        }
        values["edge_tts"] = systems["scaled_bytes"]["metrics"][metric][
            "edge_tts_trial_values"
        ]
        values["piper_lessac_low"] = [row["wer"][metric] for row in piper]
        if any(len(value) != len(SEEDS) for value in values.values()):
            raise ValueError(f"{metric}: expected exactly {len(SEEDS)} paired trials")

        result["metrics"][metric] = {
            name: {"mean_ci95": ci(value), "trial_values": value}
            for name, value in values.items()
        }
        text_forced = np.asarray(values["text_forced"])
        scaled_bytes = np.asarray(values["scaled_bytes"])
        result["metrics"][metric]["scaled_bytes"][
            "paired_minus_text_forced_ci95"
        ] = ci(scaled_bytes - text_forced)
        result["metrics"][metric]["scaled_phonemes"][
            "paired_minus_text_forced_ci95"
        ] = ci(np.asarray(values["scaled_phonemes"]) - text_forced)
        result["metrics"][metric]["scaled_phonemes"][
            "paired_minus_scaled_bytes_ci95"
        ] = ci(np.asarray(values["scaled_phonemes"]) - scaled_bytes)

    conditioning_paths = {
        "text_forced": Path("experiments/results/codec_tts_text_only"),
        "scaled_bytes": Path("experiments/results/codec_tts_scaled_bytes"),
        "scaled_phonemes": Path("experiments/results/codec_tts_scaled_phonemes"),
    }
    for name, conditioning_path in conditioning_paths.items():
        report = json.loads(
            (conditioning_path / "conditioning_summary.json").read_text()
        )
        result["conditioning"][name] = {
            "shuffled_minus_true_nll_mean_ci95": report[
                "shuffled_minus_true_nll_mean_ci95"
            ],
            "generated_sequence_sensitivity_mean_ci95": report[
                "generated_sequence_sensitivity_mean_ci95"
            ],
            "passes_locked_gate": not report["conditioning_broken_by_locked_gate"],
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [
        "# Codec-TTS scale study",
        "",
        "Five paired training seeds; intervals use 10,000 seed-level bootstrap resamples. "
        "WER may exceed 100% because insertions are unbounded.",
        "",
        "| System | Overall WER mean [95% CI] | Domain WER mean [95% CI] | Common WER mean [95% CI] | Conditioning gate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "text_forced",
        "scaled_bytes",
        "scaled_phonemes",
        "piper_lessac_low",
        "edge_tts",
    ):
        cells = []
        for metric in METRICS:
            estimate = result["metrics"][metric][name]["mean_ci95"]
            cells.append(
                f"{estimate[0]*100:.2f}% [{estimate[1]*100:.2f}, {estimate[2]*100:.2f}]"
            )
        gate = result["conditioning"].get(name, {}).get(
            "passes_locked_gate", "external comparator"
        )
        lines.append(f"| {name} | {' | '.join(cells)} | {gate} |")

    lines.extend(
        [
            "",
            "## Paired intervention effects",
            "",
            "Positive values are worse. Each interval is computed from the five "
            "seed-matched WER differences.",
            "",
            "| Metric | Scaled bytes − text-forced | Scaled phonemes − scaled bytes |",
            "|---|---:|---:|",
        ]
    )
    for metric in METRICS:
        stage_a = result["metrics"][metric]["scaled_bytes"][
            "paired_minus_text_forced_ci95"
        ]
        stage_b = result["metrics"][metric]["scaled_phonemes"][
            "paired_minus_scaled_bytes_ci95"
        ]
        lines.append(
            f"| {metric} | {stage_a[0]*100:+.2f} pp "
            f"[{stage_a[1]*100:+.2f}, {stage_a[2]*100:+.2f}] | "
            f"{stage_b[0]*100:+.2f} pp [{stage_b[1]*100:+.2f}, {stage_b[2]*100:+.2f}] |"
        )
    (output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
