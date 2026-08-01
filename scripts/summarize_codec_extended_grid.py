#!/usr/bin/env python3
"""Merge the corrective and low-rate codec studies into one 200–500 bps curve."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument("--high-dir",default="experiments/results/codec_medical_corrective"); p.add_argument("--low-dir",default="experiments/results/codec_medical_low_rate"); p.add_argument("--output-dir",default="experiments/results/codec_medical_extended"); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    signals=[]; wers=[]
    for directory in (Path(a.low_dir),Path(a.high_dir)):
        signals+=json.loads((directory/"signal_summary.json").read_text())["cells"]
        wers+=json.loads((directory/"wer_summary.json").read_text())["cells"]
    signals=sorted(signals,key=lambda x:(x["quantizer"],x["nominal_bitrate_bps"])); wers=sorted(wers,key=lambda x:(x["quantizer"],x["nominal_bitrate_bps"])); result={"schema_version":1,"range_bps":[200,500],"n_trials_per_cell":5,"signal_cells":signals,"wer_cells":wers,"sources":[a.low_dir,a.high_dir]}; (out/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for q,color,marker in (("vq","#1f77b4","o"),("fsq","#d62728","s")):
        sig=[x for x in signals if x["quantizer"]==q]; wer=[x for x in wers if x["quantizer"]==q]; x=[v["empirical_bitrate_bps"]["mean"] for v in sig]
        axes[0].plot(x,[v["si_sdr_db"]["mean"] for v in sig],marker=marker,color=color,label=q.upper())
        axes[1].plot(x,[v.get("log_mel_l1_db",{}).get("mean",float("nan")) for v in sig],marker=marker,color=color,label=q.upper())
        axes[2].plot([v["wer_representative_empirical_bitrate_bps"] for v in wer],[100*v["delta_wer"]["overall"]["adapted_whisper"]["mean_delta_wer"] for v in wer],marker=marker,color=color,label=q.upper())
    for ax,title,ylabel in zip(axes,("Signal distortion","Phase-insensitive spectral distortion","Medical-LoRA transcription cost"),("SI-SDR (dB, higher better)","Log-mel L1 (dB, lower better)","ΔWER (percentage points)")): ax.set(title=title,xlabel="Empirical bitrate (bps)",ylabel=ylabel); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out/"rate_distortion_200_500bps.png",dpi=180); plt.close(fig)
    lines=["# Codec rate–distortion study: 200–500 bps","","Each signal point is a five-seed mean; WER uses the seed selected by median SI-SDR before ASR inference.","","| Quantizer | Nominal bps | Empirical bps | SI-SDR | Log-mel L1 |","|---|---:|---:|---:|---:|"]
    for x in sorted(signals,key=lambda z:(z["nominal_bitrate_bps"],z["quantizer"])): lines.append(f"| {x['quantizer'].upper()} | {x['nominal_bitrate_bps']} | {x['empirical_bitrate_bps']['mean']:.1f} | {x['si_sdr_db']['mean']:.3f} | {x.get('log_mel_l1_db',{}).get('mean',float('nan')):.3f} |")
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps({"cells":len(signals),"output":str(out)},indent=2))
if __name__=="__main__":main()
