#!/usr/bin/env python3
"""Regenerate continuous-codec and latency result blocks from summaries."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def block(text,start,end,body):
 if text.count(start)!=1 or text.count(end)!=1: raise ValueError(f"expected one {start}")
 before,rest=text.split(start,1); _,after=rest.split(end,1)
 return f"{before}{start}\n{body.rstrip()}\n{end}{after}"
def ci(values,scale=1,suffix=""):
 return f"{values[0]*scale:.2f}{suffix} (95% CI {values[1]*scale:.2f}–{values[2]*scale:.2f})"
def main():
 p=argparse.ArgumentParser(); p.add_argument('--continuous',default='experiments/results/continuous_codec/summary.json'); p.add_argument('--latency',default='experiments/results/latency_benchmark/summary.json'); p.add_argument('--check',action='store_true'); a=p.parse_args()
 c=json.loads(Path(a.continuous).read_text()); l=json.loads(Path(a.latency).read_text()); matched=c['continuous']['quantized_1bit']; mean=c['continuous']['posterior_mean']; corrected=c['archived_discrete']['corrected']; codec200=l['codec']['200ms']['end_to_end']; tts200=l['tts']['200ms']['generation_and_decode']
 readme=f"""The completed [continuous-codec report](experiments/results/continuous_codec/REPORT.md)
finds that the 400-bps continuous point reaches {ci(matched['signal']['si_sdr_db'],suffix=' dB')}
SI-SDR but adds {ci(matched['wer']['overall']['delta_mean_ci95'],100,' WER points')}.
Corrected VQ-400 and FSQ-400 add {corrected['vq_400bps']['delta_wer']['overall']['adapted_whisper']['mean_delta_wer']*100:.2f}
and {corrected['fsq_400bps']['delta_wer']['overall']['adapted_whisper']['mean_delta_wer']*100:.2f} points, respectively. The unquantized posterior-mean path still adds
{mean['wer']['overall']['delta_mean_ci95'][0]*100:.2f} points, so uniform quantization is not the
primary content-accuracy bottleneck. Archived discrete findings are unchanged.

The completed [latency report](experiments/results/latency_benchmark/REPORT.md)
records corrected-codec 200-ms end-to-end p95 latency of {codec200['latency_ms_p95']:.2f} ms
(median RTF {codec200['realtime_factor_median']:.4f}) and parallel-TTS generation-plus-decode
p95 latency of {tts200['latency_ms_p95']:.2f} ms (median RTF {tts200['realtime_factor_median']:.4f}) on an
RTX 3070. Both clear the locked chunk-simulation bars, but this is synchronized
batch-one inference—not a production streaming or service-capacity claim."""
 continuous=f"""## Final result

All five training seeds, six reconstruction conditions per seed, and 30
seed-matched ASR evaluations completed with clean provenance. At the primary
400-bps payload point (mean effective rate {matched['effective_bitrate_bps_mean'][0]:.2f} bps),
continuous SI-SDR is {ci(matched['signal']['si_sdr_db'],suffix=' dB')} and adapted ΔWER is
{ci(matched['wer']['overall']['delta_mean_ci95'],100,' points')}. Corrected VQ-400/FSQ-400
record archived ΔWER of {corrected['vq_400bps']['delta_wer']['overall']['adapted_whisper']['mean_delta_wer']*100:.2f}/{corrected['fsq_400bps']['delta_wer']['overall']['adapted_whisper']['mean_delta_wer']*100:.2f} points. Posterior-mean ΔWER is
{mean['wer']['overall']['delta_mean_ci95'][0]*100:.2f} points, demonstrating that quantization alone does not
explain the continuous model's content failure. In this experiment, corrected
discrete codecs are preferred for downstream ASR and symbolic compression;
continuous latents do not provide a compensating usability advantage."""
 latency=f"""## Final result

All locked timing trials completed on the RTX 3070 with raw trials retained.
For 200-ms inputs, codec end-to-end median/p95 latency is
{codec200['latency_ms_median']:.2f}/{codec200['latency_ms_p95']:.2f} ms with median RTF
{codec200['realtime_factor_median']:.4f}. Parallel TTS generation plus waveform decoding is
{tts200['latency_ms_median']:.2f}/{tts200['latency_ms_p95']:.2f} ms with median RTF
{tts200['realtime_factor_median']:.4f}. Both clear the preregistered RTF < 1 and p95 < 300 ms
chunk-simulation bars on this hardware. Full-clip codec p95 is
{l['codec']['full_clip']['end_to_end']['latency_ms_p95']:.2f} ms, retained separately from the
conversational chunk criterion. These are batch-one research measurements, not
production streaming, concurrency, tail-at-load, or enterprise throughput claims."""
 targets=[(Path('README.md'),'<!-- BEGIN GENERATED CONTINUOUS LATENCY RESULT -->','<!-- END GENERATED CONTINUOUS LATENCY RESULT -->',readme),(Path('continuous_codec/PREREGISTRATION.md'),'<!-- BEGIN GENERATED CONTINUOUS FINAL RESULT -->','<!-- END GENERATED CONTINUOUS FINAL RESULT -->',continuous),(Path('latency_benchmark/PREREGISTRATION.md'),'<!-- BEGIN GENERATED LATENCY FINAL RESULT -->','<!-- END GENERATED LATENCY FINAL RESULT -->',latency)]
 stale=[]
 for path,start,end,body in targets:
  old=path.read_text(encoding='utf-8'); new=block(old,start,end,body)
  if old!=new:
   stale.append(str(path))
   if not a.check:path.write_text(new,encoding='utf-8',newline='\n')
 if a.check and stale: raise SystemExit('stale generated docs: '+', '.join(stale))
if __name__=='__main__':main()
