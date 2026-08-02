#!/usr/bin/env python3
"""Aggregate the locked five-seed codec-TTS conditioning diagnostic."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def compact_report(report):
    """Retain primary per-trial evidence while dropping large attention matrices."""
    return {
        "seed": report["seed"],
        "n_examples": report["n_examples"],
        "conditioning": report["conditioning"],
        "free_running_position_error": report["free_running_position_error"],
        "attention_summaries": [
            {
                "id": row["id"],
                "layer": row["layer"],
                "centroid_monotonicity_r": row["centroid_monotonicity_r"],
                "attention_entropy": row["attention_entropy"],
            }
            for row in report["attention"]
        ],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "provenance": report["provenance"],
    }


def ci(values, n=10_000):
    values=np.asarray(values,float); rng=np.random.default_rng(20260731)
    means=np.asarray([rng.choice(values,len(values),True).mean() for _ in range(n)])
    return [float(values.mean()),*np.quantile(means,[.025,.975]).tolist()]

def main():
    import matplotlib.pyplot as plt

    p=argparse.ArgumentParser(); p.add_argument("--input-dir",default="experiments/results/codec_tts_conditioning"); p.add_argument("--output-json",default="experiments/results/codec_tts_conditioning/summary.json"); p.add_argument("--output-md",default="experiments/results/codec_tts_conditioning/REPORT.md"); p.add_argument("--attention-plot",default="experiments/results/codec_tts_conditioning/attention.png"); a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    reports=[json.loads((root/a.input_dir/f"seed_{s}.json").read_text()) for s in (11,22,33,44,55)]
    nll=ci([r["conditioning"]["shuffled_minus_true_nll"] for r in reports]); sensitivity=ci([r["conditioning"]["mean_generated_true_vs_shuffled_edit_rate"] for r in reports])
    errors=np.asarray([r["free_running_position_error"]["error_counts"] for r in reports]).sum(0); events=np.asarray([r["free_running_position_error"]["event_counts"] for r in reports]).sum(0); rates=np.divide(errors,events,out=np.zeros(10),where=events>0)
    conditioning_broken=nll[0] <= .05 or sensitivity[0] <= .05
    attn_r=ci([x["centroid_monotonicity_r"] for r in reports for x in r["attention"]]); attn_entropy=ci([x["attention_entropy"] for r in reports for x in r["attention"]])
    result={"schema_version":2,"n_trials":5,"seeds":[11,22,33,44,55],"shuffled_minus_true_nll_mean_ci95":nll,"generated_sequence_sensitivity_mean_ci95":sensitivity,"conditioning_broken_by_locked_gate":conditioning_broken,"position_error_counts":errors.tolist(),"position_event_counts":events.tolist(),"position_error_rates":rates.tolist(),"attention_centroid_r_mean_ci95":attn_r,"attention_entropy_mean_ci95":attn_entropy,"trial_reports":[compact_report(report) for report in reports],"input_reports":[str(Path(a.input_dir)/f"seed_{s}.json") for s in (11,22,33,44,55)]}
    out=root/a.output_json; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    def fmt(x): return f"{x[0]:.4f} [{x[1]:.4f}, {x[2]:.4f}]"
    decision="conditioning is broken; repair the conditioning path" if conditioning_broken else "conditioning passes; use the pre-registered duration-aware non-autoregressive path if drift is position-dependent"
    lines=["# Codec-TTS conditioning diagnostic","",f"Five deterministic trials (seeds 11, 22, 33, 44, 55). Decision: **{decision}**.","","| Diagnostic | Mean [95% bootstrap CI] | Locked failure gate |","|---|---:|---:|",f"| Shuffled − true teacher-forced NLL (nats/token) | {fmt(nll)} | ≤ 0.05 |",f"| True-vs-shuffled generated token edit rate | {fmt(sensitivity)} | ≤ 0.05 |",f"| Cross-attention centroid monotonicity | {fmt(attn_r)} | descriptive |",f"| Normalized cross-attention entropy | {fmt(attn_entropy)} | descriptive |","","## Free-running error by normalized position","","| Decile | Errors / events | Rate |","|---:|---:|---:|"]
    lines += [f"| {i+1} | {int(errors[i])} / {int(events[i])} | {rates[i]:.4f} |" for i in range(10)]
    (root/a.output_md).write_text("\n".join(lines)+"\n",encoding="utf-8")
    rows=reports[0]["attention"]; n=min(4,len(rows)); fig,axes=plt.subplots(n,1,figsize=(9,2.5*n),squeeze=False)
    for ax,item in zip(axes[:,0],rows[:n]): ax.imshow(np.asarray(item["matrix"]).T,aspect="auto",origin="lower",cmap="magma"); ax.set(title=f"{item['id']} — decoder layer {item['layer']}",xlabel="Codec-token position",ylabel="Text-byte position")
    fig.tight_layout(); fig.savefig(root/a.attention_plot,dpi=160); plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
