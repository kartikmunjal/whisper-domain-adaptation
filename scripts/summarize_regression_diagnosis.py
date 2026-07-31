#!/usr/bin/env python3
"""Render the synthetic-vs-real and common-control diagnosis as Markdown."""
from __future__ import annotations
import argparse, json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="experiments/results/earnings21/regression_diagnosis.json"); p.add_argument("--output",default="experiments/results/earnings21/REGRESSION_DIAGNOSIS.md"); a=p.parse_args(); root=Path(__file__).resolve().parents[1]; data=json.loads((root/a.input).read_text()); metrics=data["acoustic_comparison"]["metrics"]
    lines=["# Synthetic-to-real ASR regression diagnosis","","All acoustic values are computed per clip. Mean differences use 10,000 paired-independent bootstrap resamples; effect size is Cliff's delta (real versus synthetic).","","| Metric | Synthetic mean | Real mean | Real−synthetic 95% CI | Cliff's δ |","|---|---:|---:|---:|---:|"]
    for name,x in metrics.items(): lines.append(f"| {name} | {x['synthetic_mean']:.3f} | {x['real_mean']:.3f} | [{x['real_minus_synthetic_mean_95_ci'][0]:.3f}, {x['real_minus_synthetic_mean_95_ci'][1]:.3f}] | {x['cliffs_delta_real_vs_synthetic']:.3f} |")
    trigger=any(abs(metrics[k]["cliffs_delta_real_vs_synthetic"])>=.474 for k in ("heuristic_snr_db","silence_fraction")); lines += ["",f"Locked augmentation trigger (|δ| ≥ 0.474 for SNR or silence): **{'met' if trigger else 'not met'}**.","","## Common-slice error transitions","",f"Common-control clips: {data['common_control_error_analysis']['n_common_clips']}",""]
    for trial in data["common_control_error_analysis"]["trials"]:
        lines += [f"### Seed {trial['seed']}","","| Edit | Introduced | Resolved | Retained |","|---|---:|---:|---:|"]+[f"| {op} | {x['introduced']} | {x['resolved']} | {x['retained']} |" for op,x in trial["transitions"].items()]+["",f"Error word classes: `{json.dumps(trial['adapted_error_word_classes'],sort_keys=True)}`",""]
    out=root/a.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(lines),encoding="utf-8"); print(out)
if __name__=="__main__": main()
