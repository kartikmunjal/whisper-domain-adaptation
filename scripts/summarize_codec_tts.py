#!/usr/bin/env python3
"""Summarize the preregistered five-seed codec-token TTS experiment."""

from __future__ import annotations

import argparse
import json
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
SPLITS = ("overall", "domain_terms", "common_terms")


def trial_mean_ci(values: list[float], rng: np.random.Generator, n: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    means = np.asarray([
        rng.choice(array, len(array), replace=True).mean() for _ in range(n)
    ])
    return np.quantile(means, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results/codec_tts")
    parser.add_argument(
        "--edge-template",
        default="experiments/results/financial_research/seed_{seed}/finetuned_test.json",
    )
    parser.add_argument("--training-dir", default="checkpoints/codec_tts")
    parser.add_argument("--domain-vocab", default="configs/financial_terms.txt")
    parser.add_argument("--output", default="experiments/results/codec_tts/summary.json")
    parser.add_argument(
        "--markdown-output", default="experiments/results/codec_tts/summary.md"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    analyzer = DomainWERAnalyzer(load_domain_vocab(root / args.domain_vocab))
    rng = np.random.default_rng(20260729)
    trials = []
    for seed in SEEDS:
        generated = json.loads(
            (root / args.results_dir / f"seed_{seed}" / "round_trip_wer.json").read_text()
        )
        edge = json.loads((root / args.edge_template.format(seed=seed)).read_text())
        generation = json.loads(
            (root / args.results_dir / f"seed_{seed}" / "generation_report.json").read_text()
        )
        training = json.loads(
            (root / args.training_dir / f"seed_{seed}" / "run.json").read_text()
        )
        if [row["id"] for row in generated["predictions"]] != [
            row["id"] for row in edge["predictions"]
        ]:
            raise RuntimeError(f"Generated and Edge-TTS IDs are not paired for seed {seed}")
        best_epoch = min(training["history"], key=lambda row: row["validation_nll"])
        trials.append({
            "seed": seed,
            "generated": generated,
            "edge": edge,
            "generation": generation,
            "best_validation_nll": float(training["best_validation_nll"]),
            "best_validation_token_accuracy": float(
                best_epoch["validation_token_accuracy"]
            ),
            "trainable_parameters": int(training["trainable_parameters"]),
            "planned_optimizer_steps": int(training["planned_optimizer_steps"]),
        })

    summary = {
        "schema_version": 1,
        "n_trials": len(trials),
        "seeds": list(SEEDS),
        "comparison": "codec-token TTS minus paired Edge-TTS on identical held-out sentences",
        "metrics": {},
    }
    references = [row["reference"] for row in trials[0]["edge"]["predictions"]]
    masks = {
        "overall": [True] * len(references),
        "domain_terms": [analyzer._contains_domain_term(x) for x in references],
    }
    masks["common_terms"] = [not x for x in masks["domain_terms"]]
    for split in SPLITS:
        generated_values, edge_values, deltas, clip_cis = [], [], [], []
        keep = masks[split]
        refs = [x for x, flag in zip(references, keep) if flag]
        for trial in trials:
            generated_values.append(float(trial["generated"]["wer"][split]))
            edge_values.append(float(trial["edge"]["wer"][split]))
            generated_hypotheses = [
                row["hypothesis"]
                for row, flag in zip(trial["generated"]["predictions"], keep)
                if flag
            ]
            edge_hypotheses = [
                row["hypothesis"]
                for row, flag in zip(trial["edge"]["predictions"], keep)
                if flag
            ]
            paired = paired_bootstrap_difference_ci(
                refs,
                edge_hypotheses,
                generated_hypotheses,
                n_resamples=args.bootstrap_resamples,
                seed=trial["seed"],
            )
            deltas.append(float(paired["estimate"]))
            clip_cis.append(paired)
        summary["metrics"][split] = {
            "codec_tts_mean_wer": float(np.mean(generated_values)),
            "codec_tts_trial_values": generated_values,
            "codec_tts_trial_bootstrap_95_ci": trial_mean_ci(
                generated_values, rng, args.bootstrap_resamples
            ),
            "edge_tts_mean_wer": float(np.mean(edge_values)),
            "edge_tts_trial_values": edge_values,
            "edge_tts_trial_bootstrap_95_ci": trial_mean_ci(
                edge_values, rng, args.bootstrap_resamples
            ),
            "mean_paired_delta": float(np.mean(deltas)),
            "paired_delta_trial_values": deltas,
            "paired_delta_trial_bootstrap_95_ci": trial_mean_ci(
                deltas, rng, args.bootstrap_resamples
            ),
            "per_trial_paired_clip_bootstrap": clip_cis,
        }

    scalar_sources = {
        "validation_nll": [trial["best_validation_nll"] for trial in trials],
        "validation_token_accuracy": [
            trial["best_validation_token_accuracy"] for trial in trials
        ],
        "trainable_parameters": [
            trial["trainable_parameters"] for trial in trials
        ],
        "planned_optimizer_steps": [
            trial["planned_optimizer_steps"] for trial in trials
        ],
        "eos_rate": [trial["generation"]["eos_rate"] for trial in trials],
        "generation_failure_rate": [
            trial["generation"]["generation_failure_rate"] for trial in trials
        ],
        "empty_token_failure_rate": [
            trial["generation"]["empty_token_failure_rate"] for trial in trials
        ],
        "absolute_sequence_length_error_tokens": [
            trial["generation"]["sequence_length_error"]["mean_absolute_tokens"]
            for trial in trials
        ],
        "si_sdr_db_conditional_on_decodable_output": [
            trial["generation"]["si_sdr_db_conditional_on_decodable_output"]["mean"]
            for trial in trials
        ],
    }
    for name, values in scalar_sources.items():
        if any(value is None for value in values):
            scalar_sources[name] = [value for value in values if value is not None]
    summary["diagnostics"] = {
        name: {
            "mean": float(np.mean(values)),
            "trial_values": values,
            "trial_bootstrap_95_ci": trial_mean_ci(
                values, rng, args.bootstrap_resamples
            ),
        }
        for name, values in scalar_sources.items() if values
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = {"overall": "Overall", "domain_terms": "Domain", "common_terms": "Common"}
    lines = [
        "# Codec-token TTS held-out evaluation",
        "",
        "| Split | Codec TTS WER (95% trial CI) | Edge-TTS WER (95% trial CI) | Paired ΔWER (95% trial CI) | N_trials |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        row = summary["metrics"][split]
        model_ci = row["codec_tts_trial_bootstrap_95_ci"]
        edge_ci = row["edge_tts_trial_bootstrap_95_ci"]
        delta_ci = row["paired_delta_trial_bootstrap_95_ci"]
        lines.append(
            f"| {labels[split]} | {row['codec_tts_mean_wer'] * 100:.2f}% "
            f"({model_ci[0] * 100:.2f}–{model_ci[1] * 100:.2f}%) | "
            f"{row['edge_tts_mean_wer'] * 100:.2f}% "
            f"({edge_ci[0] * 100:.2f}–{edge_ci[1] * 100:.2f}%) | "
            f"{row['mean_paired_delta'] * 100:+.2f} pp "
            f"({delta_ci[0] * 100:+.2f}–{delta_ci[1] * 100:+.2f} pp) | 5 |"
        )
    lines.extend([
        "",
        "Diagnostics are reported with five-trial bootstrap confidence intervals in `summary.json`.",
        "This comparison uses synthetic held-out Edge-TTS references and must not be interpreted as natural-speech TTS quality.",
        "",
        "Generated by `scripts/summarize_codec_tts.py`.",
        "",
    ])
    markdown = root / args.markdown_output
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
