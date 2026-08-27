#!/usr/bin/env python3
"""Aggregate continuous-codec trials and archived discrete comparators."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SEEDS=(11,22,33,44,55); METRICS=("overall","domain_terms","common_terms"); N=10_000; RNG=20260826
def ci(values):
 values=np.asarray(values,float); rng=np.random.default_rng(RNG); means=np.asarray([rng.choice(values,len(values),replace=True).mean() for _ in range(N)])
 return [float(values.mean()),*np.quantile(means,[.025,.975]).tolist()]
def cell(document,quantizer,rate): return next(x for x in document["cells"] if x["quantizer"]==quantizer and x["nominal_bitrate_bps"]==rate)
def main():
 root=Path("experiments/results/continuous_codec"); conditions=("posterior_mean","quantized_1bit","quantized_2bit","quantized_4bit","quantized_6bit","quantized_8bit")
 original={s:json.loads(Path(f"experiments/results/codec_medical_corrective/asr/original/seed_{s}.json").read_text()) for s in SEEDS}
 out={"schema_version":1,"n_trials":5,"seeds":list(SEEDS),"bootstrap":{"unit":"training_seed","n_resamples":N,"seed":RNG},"continuous":{},"archived_discrete":{}}
 for condition in conditions:
  reports=[json.loads((root/f"seed_{s}/report.json").read_text()) for s in SEEDS]; asr=[json.loads((root/f"seed_{s}/asr/{condition}.json").read_text()) for s in SEEDS]
  signal={k:ci([r["conditions"][condition][k]["mean"] for r in reports]) for k in ("si_sdr_db","log_mel_l1_db")}
  row={"quantization_bits":reports[0]["conditions"][condition]["quantization_bits"],"signal":signal,"posterior_kl_per_frame":ci([r["conditions"][condition]["posterior_kl_per_frame"] for r in reports]),"latent_saturation_fraction":ci([r["conditions"][condition]["latent_saturation_fraction"] for r in reports]),"wer":{}}
  if row["quantization_bits"] is not None:
   row["payload_bitrate_bps"]=reports[0]["conditions"][condition]["payload_bitrate_bps"]; row["effective_bitrate_bps_mean"]=ci([r["conditions"][condition]["effective_bitrate_bps_mean"] for r in reports])
  for metric in METRICS:
   absolute=[r["wer"][metric] for r in asr]; delta=[r["wer"][metric]-original[s]["wer"][metric] for r,s in zip(asr,SEEDS)]
   row["wer"][metric]={"absolute_mean_ci95":ci(absolute),"delta_mean_ci95":ci(delta),"trial_values":absolute}
  out["continuous"][condition]=row
 for label,path in (("pre_correction","experiments/results/codec_medical/wer_summary.json"),("corrected","experiments/results/codec_medical_corrective/wer_summary.json")):
  document=json.loads(Path(path).read_text()); out["archived_discrete"][label]={}
  for quantizer in ("vq","fsq"):
   source=cell(document,quantizer,400); out["archived_discrete"][label][f"{quantizer}_400bps"]={"si_sdr_db":source["si_sdr_db"],"empirical_bitrate_bps":source["empirical_bitrate_bps"],"delta_wer":source["delta_wer"]}
 root.mkdir(parents=True,exist_ok=True); (root/"summary.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8",newline="\n")
 matched=out["continuous"]["quantized_1bit"]
 lines=["# Continuous versus discrete codec study","","Five continuous-codec training seeds; 10,000 seed-bootstrap resamples. The 400-bps continuous payload point is rate-matched to archived VQ/FSQ-400 nominal payload, while effective continuous rate additionally includes the preregistered 128-bit per-clip header.","","| Continuous condition | Payload rate | SI-SDR [95% CI] | Log-mel L1 [95% CI] | Adapted overall WER [95% CI] | Delta WER [95% CI] |","|---|---:|---:|---:|---:|---:|"]
 for name,row in out["continuous"].items():
  rate="uncompressed" if row["quantization_bits"] is None else f"{row['payload_bitrate_bps']:.0f} bps"; s=row["signal"]["si_sdr_db"]; m=row["signal"]["log_mel_l1_db"]; w=row["wer"]["overall"]["absolute_mean_ci95"]; d=row["wer"]["overall"]["delta_mean_ci95"]
  lines.append(f"| {name} | {rate} | {s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}] | {m[0]:.2f} [{m[1]:.2f}, {m[2]:.2f}] | {w[0]*100:.2f}% [{w[1]*100:.2f}, {w[2]*100:.2f}] | {d[0]*100:+.2f} pp [{d[1]*100:+.2f}, {d[2]*100:+.2f}] |")
 lines += ["","## Archived 400-bps discrete anchors","","These rows are read without modification from the original pre-correction and corrective reports. Empirical entropy rates differ from nominal payload rates.","","| Study | Codec | SI-SDR mean | Empirical rate | Adapted overall delta WER |","|---|---|---:|---:|---:|"]
 for study,cells in out["archived_discrete"].items():
  for name,row in cells.items(): lines.append(f"| {study} | {name} | {row['si_sdr_db']['mean']:.2f} dB | {row['empirical_bitrate_bps']['mean']:.1f} bps | {row['delta_wer']['overall']['adapted_whisper']['mean_delta_wer']*100:+.2f} pp |")
 lines += ["","## Interpretation rule","","Recommend continuous latents only when reconstruction/downstream robustness justifies their explicit transmission rate; recommend discrete latents when symbolic compression or token generation is required and measured utilization is adequate. The result above characterizes this tradeoff and does not revise the archived discrete findings.","","Generated by `scripts/summarize_continuous_codec.py`."]
 (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
 print(json.dumps(out,indent=2))
if __name__=="__main__": main()
