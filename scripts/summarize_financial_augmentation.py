#!/usr/bin/env python3
"""Compare clean-synthetic and acoustically augmented financial LoRA trials."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
SEEDS=(11,22,33,44,55); METRICS=("overall","domain_terms","common_terms")
def boot(values,n=10_000):
    values=np.asarray(values,float); rng=np.random.default_rng(20260731); means=np.asarray([rng.choice(values,len(values),True).mean() for _ in range(n)]); return [float(values.mean()),*np.quantile(means,[.025,.975]).tolist()]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--clean-dir",default="experiments/results/earnings21"); p.add_argument("--augmented-dir",default="experiments/results/earnings21_augmented"); p.add_argument("--output",default="experiments/results/earnings21_augmented/comparison.json"); p.add_argument("--markdown-output",default="experiments/results/earnings21_augmented/comparison.md"); a=p.parse_args(); clean=[]; aug=[]
    for seed in SEEDS:
        clean.append(json.loads((Path(a.clean_dir)/f"seed_{seed}"/"finetuned.json").read_text())); aug.append(json.loads((Path(a.augmented_dir)/f"seed_{seed}"/"finetuned.json").read_text()))
        if clean[-1]["provenance"]["seed"]!=seed or aug[-1]["provenance"]["seed"]!=seed: raise RuntimeError("Seed provenance mismatch")
        if [x["id"] for x in clean[-1]["predictions"]]!=[x["id"] for x in aug[-1]["predictions"]]: raise RuntimeError("Unpaired prediction IDs")
    result={"schema_version":1,"n_trials":5,"seeds":list(SEEDS),"metrics":{}}
    for metric in METRICS:
        c=np.asarray([x["wer"][metric] for x in clean]); u=np.asarray([x["wer"][metric] for x in aug]); result["metrics"][metric]={"clean_mean_ci95":boot(c),"augmented_mean_ci95":boot(u),"paired_augmented_minus_clean_ci95":boot(u-c),"clean_trial_values":c.tolist(),"augmented_trial_values":u.tolist()}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    labels={"overall":"Overall","domain_terms":"Domain","common_terms":"Common"}; lines=["# Financial acoustic-augmentation ablation","","Five paired seeds evaluated on the fixed 20-clip Earnings-21 anchor. Negative paired change favors augmentation.","","| Split | Clean mean [95% CI] | Augmented mean [95% CI] | Paired change [95% CI] | N_trials |","|---|---:|---:|---:|---:|"]
    for m in METRICS:
        x=result["metrics"][m]; f=lambda v:f"{100*v[0]:.2f}% [{100*v[1]:.2f}, {100*v[2]:.2f}]"; lines.append(f"| {labels[m]} | {f(x['clean_mean_ci95'])} | {f(x['augmented_mean_ci95'])} | {f(x['paired_augmented_minus_clean_ci95'])} | 5 |")
    Path(a.markdown_output).write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
