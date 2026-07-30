#!/usr/bin/env python3
"""Generate five-trial ASR tables and paired uncertainty from JSON predictions."""

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
METRICS = ("overall", "domain_terms", "common_terms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--trial-template", required=True, help="Contains {seed}")
    parser.add_argument("--domain-vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--caveat", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    trials = [
        json.loads(Path(args.trial_template.format(seed=seed)).read_text())
        for seed in SEEDS
    ]
    baseline_predictions = baseline["predictions"]
    baseline_ids = [row["id"] for row in baseline_predictions]
    analyzer = DomainWERAnalyzer(load_domain_vocab(args.domain_vocab))
    references = [row["reference"] for row in baseline_predictions]
    domain_mask = [analyzer._contains_domain_term(ref) for ref in references]
    summary = {
        "schema_version": 1,
        "title": args.title,
        "n_trials": len(trials),
        "seeds": list(SEEDS),
        "baseline": baseline["wer"],
        "caveat": args.caveat,
        "metrics": {},
    }
    rng = np.random.default_rng(20260729)
    for metric in METRICS:
        keep = (
            [True] * len(references) if metric == "overall"
            else domain_mask if metric == "domain_terms"
            else [not value for value in domain_mask]
        )
        refs = [x for x, flag in zip(references, keep) if flag]
        base_hyps = [
            row["hypothesis"] for row, flag in zip(baseline_predictions, keep) if flag
        ]
        values, deltas, clip_intervals = [], [], []
        for seed, trial in zip(SEEDS, trials):
            predictions = trial["predictions"]
            if [row["id"] for row in predictions] != baseline_ids:
                raise RuntimeError(f"Prediction IDs are not paired for seed {seed}")
            values.append(float(trial["wer"][metric]))
            adapted = [
                row["hypothesis"] for row, flag in zip(predictions, keep) if flag
            ]
            paired = paired_bootstrap_difference_ci(
                refs, base_hyps, adapted,
                n_resamples=args.bootstrap_resamples,
                seed=seed,
            )
            deltas.append(float(paired["estimate"]))
            clip_intervals.append(paired)
        values_array, deltas_array = np.array(values), np.array(deltas)
        trial_value_means = np.array([
            rng.choice(values_array, len(values_array), replace=True).mean()
            for _ in range(args.bootstrap_resamples)
        ])
        trial_delta_means = np.array([
            rng.choice(deltas_array, len(deltas_array), replace=True).mean()
            for _ in range(args.bootstrap_resamples)
        ])
        summary["metrics"][metric] = {
            "mean_wer": float(values_array.mean()),
            "trial_values": values,
            "trial_bootstrap_95_ci": np.quantile(
                trial_value_means, [0.025, 0.975]
            ).tolist(),
            "mean_paired_delta": float(deltas_array.mean()),
            "paired_delta_trial_values": deltas,
            "paired_delta_trial_bootstrap_95_ci": np.quantile(
                trial_delta_means, [0.025, 0.975]
            ).tolist(),
            "per_trial_paired_clip_bootstrap": clip_intervals,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    labels = {"overall": "Overall", "domain_terms": "Domain", "common_terms": "Common"}
    lines = [
        f"# {args.title}", "",
        "| Split | Baseline WER | Adapted mean WER (95% trial CI) | Paired ΔWER (95% trial CI) | N_trials |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        item = summary["metrics"][metric]
        ci, delta_ci = item["trial_bootstrap_95_ci"], item["paired_delta_trial_bootstrap_95_ci"]
        lines.append(
            f"| {labels[metric]} | {baseline['wer'][metric] * 100:.2f}% | "
            f"{item['mean_wer'] * 100:.2f}% ({ci[0] * 100:.2f}–{ci[1] * 100:.2f}%) | "
            f"{item['mean_paired_delta'] * 100:+.2f} pp "
            f"({delta_ci[0] * 100:+.2f}–{delta_ci[1] * 100:+.2f} pp) | 5 |"
        )
    if args.caveat:
        lines.extend(["", f"> {args.caveat}"])
    lines.extend(["", "Generated by `scripts/summarize_asr_trials.py`.", ""])
    markdown = Path(args.markdown_output)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
