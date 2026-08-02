# Codec Rate-Distortion and ASR Robustness Plan

Status: locked before codec training or medical evaluation.

## Question

At matched nominal bitrate, how much do compact VQ-VAE and FSQ codecs degrade
signal fidelity and medical ASR, and does medical adaptation change robustness
to codec artifacts?

## Data boundaries

- Codec training: financial Edge-TTS training partition only.
- Medical adapter training/validation: a separately documented, licensed
  medical speech source with speaker-disjoint partitions.
- Confirmatory medical evaluation: all 40 `corti/med-dictate` recordings,
  evaluation-only under its dataset license.
- `med-dictate` is never used for codec or ASR training, hyperparameter
  selection, early stopping, normalization design, or qualitative tuning.

## Fixed conditions

- Codecs: VQ-VAE and FSQ.
- Three preregistered nominal-rate configurations per codec.
- Identical encoder/decoder capacity and training budget within each rate.
- Five seeds per codec/rate configuration: 11, 22, 33, 44, 55.
- ASR: frozen Whisper-small and five medical LoRA adapters trained with the
  same seeds.
- Audio conditions: original, VQ reconstruction, FSQ reconstruction.

Nominal bitrate is calculated from the actual latent frame rate and entropy of
the discrete representation. Both fixed-width bitrate and empirical token
entropy rate are reported. Parameter count and training steps are reported.

## Outcomes

- Primary: reconstructed-minus-original domain-term WER.
- Secondary: overall WER, common-term WER, SI-SDR, and codec token usage.
- Report every result with N_trials and a 95% CI.
- Use 10,000 paired clip-bootstrap resamples for within-run deltas and
  trial-level bootstrap intervals across the five seeds.
- Produce rate-distortion plots from a named script; no hand-edited figures.

## Interpretation

No codec is called better unless the improvement is supported at matched rate
and explained mechanically through frame rate, token utilization, distortion,
or ASR error categories. A signal metric and WER may disagree; both are
reported without selecting the favorable one.

## Amendment 1 — English evaluation subset

Documented 2026-07-30 after the signal grid and part of the ASR grid had been
observed. The original text incorrectly said that the confirmatory evaluation
used all 40 multilingual `corti/med-dictate` recordings. The committed medical
and codec launchers had fixed `eval_en_manifest.parquet` before any of these
evaluations, so the actual confirmatory set is the 24 recordings in the
dataset's English configuration.

This restriction is based only on source language metadata: the medical
vocabulary, synthetic adapter training data, and Whisper evaluation are
English. It was not selected using codec distortion or ASR output. The German
and French recordings remain materialized in `eval_manifest.parquet` but are
outside this English-domain estimand. All generated reports must state
`n_clips=24`; no result may be described as covering all 40 recordings.

## Amendment 2 — Corrective code-utilization study

Locked 2026-07-30 after the original grid was complete and disclosed codebook
collapse. This is a labeled corrective study, not a replacement for the
original result. Original checkpoints, predictions, summaries, and plots
remain immutable and are the "before" condition.

The corrective implementation adds:

- per-epoch code histograms, dead-code fraction, empirical entropy, and reset
  counts;
- EMA VQ codebook updates with decay 0.99 and epsilon 1e-5;
- replacement of codes unused for 100 batches using seeded samples from real
  encoder outputs; and
- per-frame RMS normalization to 1.0 before the fixed FSQ grid, with
  pre-quantization range and saturation telemetry.

The matched-rate grid, seeds, training data, epoch budget, 24-clip English
medical evaluation set, Whisper models, WER splits, SI-SDR, and 10,000-resample
uncertainty procedure are unchanged. The corrective report must show before
and after side by side. A fix is not called successful merely because code
entropy increases; reconstruction SI-SDR and ASR ΔWER must also improve. Any
post-lock implementation correction is documented here before its GPU outcome
is observed.

## Final corrective result

All 30 corrective training runs, 30 reconstruction evaluations, and 42 ASR
evaluations completed with clean provenance. Empirical bitrate and SI-SDR
improved in every cell, indicating that EMA/dead-code handling and FSQ
normalization mitigated utilization collapse. Adapted-Whisper overall ΔWER
improved at VQ-300 and FSQ-300, was inconclusive at FSQ-400 and FSQ-500, and
significantly worsened at VQ-400 and VQ-500. The corrective study is therefore
mixed: it supports a code-utilization and signal-fidelity improvement, but not
a general ASR-robustness or competitive-codec claim. Realized bitrate changed
substantially, so the before/after comparison is not a controlled
matched-bitrate codec ranking.

## Amendment 3 — Phase-insensitive metric and lower-rate extension

Locked 2026-07-31 before the additional metric or lower-rate outcomes were
observed. Every existing and new reconstruction adds 80-bin log-mel absolute
distance using a 25 ms window, 10 ms hop, 20–7600 Hz range, and per-clip mean
in decibels. This phase-insensitive metric is reported with clip and five-seed
bootstrap intervals alongside SI-SDR and WER; disagreement is retained rather
than resolved by selecting a favorable metric.

The corrective grid is extended downward to 200 bps (four nominal bits per
frame: VQ size 16; FSQ levels [4,4]) and 250 bps (five nominal bits per frame:
VQ size 32; FSQ levels [4,4,2]). Seeds, architecture, 30-epoch budget, training
data, 24-clip English medical evaluation, representative selection rule, ASR
models, and 10,000-resample uncertainty remain fixed. These cells extend a
rate-distortion curve; they are not selected using their WER.

## Amendment 4 — Fixed open pretrained codec comparator

Locked 2026-08-01 before downloading the comparator weights or observing any
comparator reconstruction, signal metric, or ASR output. The external model is
the open 24 kHz monophonic Meta EnCodec checkpoint
`facebook/encodec_24khz`, pinned to Hugging Face revision
`c1dbe2ae3f1de713481a3b3e7c47f357092ee040`, at its lowest officially
supported 1.5 kbps operating point (two 1,024-entry residual codebooks at
75 frames/s). The upstream implementation supports 1.5, 3, 6, 12, and
24 kbps; it has no official 200–500 bps operating point.

The same frozen 24-clip English `med-dictate` manifest, Whisper-small model,
five medical adapters, domain vocabulary, overall/domain/common WER splits,
SI-SDR, 80-bin log-mel distance, and 10,000-resample uncertainty procedures
are retained. Long clips use deterministic 10 s chunks with 100 ms overlap
and linear crossfades, matching the custom-codec reconstruction convention.
The pretrained comparator has one fixed checkpoint rather than five training
seeds: signal intervals therefore bootstrap the 24 clips; frozen-Whisper WER
uses a paired clip bootstrap; adapted-Whisper summaries and contrasts use the
five fixed adapter seeds with seed-level bootstrap intervals.

The preregistered custom contrasts are the corrective VQ-500 and FSQ-500
cells, the closest nominal design points already in the grid. EnCodec receives
three times their nominal bitrate and substantially more than their measured
entropy rates, so this is an external quality anchor, not a matched-rate codec
ranking. The report must show nominal fixed-width bitrate, pooled empirical
code entropy rate, entropy utilization, per-codebook unique-code fraction,
SI-SDR, log-mel distance, absolute reconstructed WER, and reconstructed-minus-
original ΔWER. WER ratios may be descriptive only, must carry a 95% interval,
and must be labeled with the bitrate mismatch. No claim may divide a quality
metric by bitrate or imply EnCodec performance at an unsupported rate.

<!-- BEGIN GENERATED ENCODEC FINAL RESULT -->
## Amendment 4 final result

The pinned EnCodec evaluation completed on all 24 clips, the
frozen Whisper baseline, and all 5 fixed medical adapters with clean
provenance and 10,000-resample uncertainty. EnCodec's mean
adapted absolute WER is 47.42% (95% CI 46.00–48.85), and reconstructed-minus-original
ΔWER is +17.62 points (95% CI 15.74–19.50). The corresponding
absolute-WER ratios are 2.63× (95% CI 2.41–2.89) for corrective VQ-500 and
2.85× (95% CI 2.58–3.11) for corrective FSQ-500.

EnCodec records -2.62 dB mean SI-SDR, 7.75 dB mean log-mel
distance, and 64.2% pooled entropy utilization. VQ-500 records
-15.73 dB, 12.51 dB, and 40.1%; FSQ-500 records
-28.44 dB, 13.95 dB, and 13.4%. This supports the
mechanical diagnosis that residual custom-codec under-utilization accompanies
materially worse reconstruction and transcription. It does not isolate
utilization as the sole cause. EnCodec receives 3× the nominal bitrate and
4.80×/14.40× the empirical entropy rate of VQ-500/FSQ-500, so no
matched-rate superiority claim is made.
The complete machine-generated tables, per-split intervals, hashes, and plot
are in `experiments/results/codec_medical_encodec/`.
<!-- END GENERATED ENCODEC FINAL RESULT -->
