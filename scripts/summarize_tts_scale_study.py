#!/usr/bin/env python3
"""Aggregate the preregistered capacity/data and phoneme TTS interventions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
SEEDS=(11,22,33,44,55); METRICS=("overall","domain_terms","common_terms")
def ci(v,n=10_000):
 v=np.asarray(v,float); rng=np.random.default_rng(20260801); means=np.asarray([rng.choice(v,len(v),True).mean() for _ in range(n)]); return [float(v.mean()),*np.quantile(means,[.025,.975]).tolist()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="experiments/results/codec_tts_scale_study"); a=p.parse_args(); systems={}
 for name,path in (("text_forced","experiments/results/codec_tts_text_only"),("scaled_bytes","experiments/results/codec_tts_scaled_bytes"),("scaled_phonemes","experiments/results/codec_tts_scaled_phonemes")):
  systems[name]=json.loads((Path(path)/"summary.json").read_text())
 piper=[json.loads((Path("experiments/results/piper_lessac_low")/f"seed_{s}.json").read_text()) for s in SEEDS]; result={"schema_version":1,"n_trials":5,"seeds":list(SEEDS),"metrics":{},"conditioning":{}}
 for metric in METRICS:
  values={name:[x["metrics"][metric]["codec_tts_trial_values"][i] for i in range(5)] for name,x in systems.items()}; values["edge_tts"]=systems["scaled_bytes"]["metrics"][metric]["edge_tts_trial_values"]; values["piper_lessac_low"]=[x["wer"][metric] for x in piper]
  result["metrics"][metric]={name:{"mean_ci95":ci(v),"trial_values":v} for name,v in values.items()}; base=np.asarray(values["text_forced"])
  for name in ("scaled_bytes","scaled_phonemes"): result["metrics"][metric][name]["paired_minus_text_forced_ci95"]=ci(np.asarray(values[name])-base)
 for name in ("scaled_bytes","scaled_phonemes"):
  x=json.loads((Path(f"experiments/results/codec_tts_{name}")/"conditioning_summary.json").read_text()); result["conditioning"][name]={"shuffled_minus_true_nll_mean_ci95":x["shuffled_minus_true_nll_mean_ci95"],"generated_sequence_sensitivity_mean_ci95":x["generated_sequence_sensitivity_mean_ci95"],"passes_locked_gate":not x["conditioning_broken_by_locked_gate"]}
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps(result,indent=2)); lines=["# Codec-TTS scale study","","Five paired seeds; WER is reported as a proportion and may exceed 1 because insertions are unbounded.","","| System | Overall WER mean [95% CI] | Conditioning gate |","|---|---:|---:|"]
 for name in ("text_forced","scaled_bytes","scaled_phonemes","piper_lessac_low","edge_tts"):
  x=result["metrics"]["overall"][name]["mean_ci95"]; gate=result["conditioning"].get(name,{}).get("passes_locked_gate","external comparator"); lines.append(f"| {name} | {x[0]*100:.2f}% [{x[1]*100:.2f}, {x[2]*100:.2f}] | {gate} |")
 (out/"REPORT.md").write_text("\n".join(lines)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
