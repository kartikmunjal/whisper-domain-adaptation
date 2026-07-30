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
