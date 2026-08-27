# Continuous-Latent Codec Preregistration

Status: locked before confirmatory training or evaluation.

## Question

How does a continuous Gaussian bottleneck compare with the existing corrected
VQ-VAE and FSQ systems when the waveform encoder, decoder, data split, training
schedule, and evaluation protocol are held fixed?

## Locked design

- Train five seeds: 11, 22, 33, 44, and 55.
- Use the existing financial-research training manifest and untouched codec
  evaluation manifest.
- Reuse `WaveformEncoder` and `WaveformDecoder` without architectural changes.
- Use a diagonal-Gaussian bottleneck with a standard-normal prior and report
  both deterministic posterior-mean reconstruction and a transmittable uniform-
  quantized representation.
- Compare the continuous system with the archived pre-correction and corrected
  VQ/FSQ results. No test result may be used to change training or select a seed.

## Locked outcomes

1. Mean clip SI-SDR with a 10,000-resample clip-bootstrap 95% interval.
2. Frozen-ASR WER on identical reconstructed clips and adapters.
3. Effective bitrate after fixed-width uniform quantization, including metadata
   overhead assumptions. An unquantized float tensor is never described as a
   compressed representation.
4. Posterior KL per frame and latent saturation fraction as mechanism checks.

The primary interpretation is a trade-off characterization, not a requirement
that either family win. Negative and inconclusive results will be retained.

## Deviations

Any change after the first confirmatory result must be recorded with date,
rationale, and whether the result had been inspected. Changed analyses are
exploratory and cannot replace the locked outcomes above.

## Execution amendment 1 — protocol/code reconciliation

Approved by the repository owner on 2026-08-26 before any continuous-codec
training, reconstruction, ASR, or latency outcome was observed. The original
discrete protocols, checkpoints, results, and conclusions remain immutable.

The English-only 24-clip medical manifest
`data/med_dictate_eval/eval_en_manifest.parquet` is used, matching the existing
codec study; the broader manifest also contains French clips and is not a valid
like-for-like comparison. Long clips use deterministic 10-second windows,
one-second overlap, and linear crossfades.

Both unquantized posterior-mean reconstruction and transmittable uniform-
quantized reconstruction are retained. The fixed eight-dimensional bottleneck
is evaluated at 1, 2, 4, 6, and 8 bits per scalar. The primary rate-controlled
point is 1 bit/scalar = 400 payload bps, compared with archived pre-correction
and corrected VQ-400/FSQ-400 cells. Higher-rate points form a descriptive
rate-distortion curve and are not called matched. Effective rate also reports
a fixed 128-bit per-clip header assumption (shape, scale, and bit depth);
payload-only rate is shown because archived discrete reports exclude framing.

Posterior KL/frame and saturation are evaluated on held-out clips. Five
training seeds are aggregated with 10,000 seed-bootstrap resamples. Frozen-ASR
uses the seed-matched five medical adapters and the existing overall/domain/
common WER and reconstructed-minus-original delta protocol. No seed or bit
depth is selected using test WER.

<!-- BEGIN GENERATED CONTINUOUS FINAL RESULT -->
## Final result

All five training seeds, six reconstruction conditions per seed, and 30
seed-matched ASR evaluations completed with clean provenance. At the primary
400-bps payload point (mean effective rate 400.93 bps),
continuous SI-SDR is -21.91 dB (95% CI -32.21–-14.90) and adapted ΔWER is
108.82 points (95% CI 85.18–135.39). Corrected VQ-400/FSQ-400
record archived ΔWER of 82.44/82.43 points. Posterior-mean ΔWER is
107.64 points, demonstrating that quantization alone does not
explain the continuous model's content failure. In this experiment, corrected
discrete codecs are preferred for downstream ASR and symbolic compression;
continuous latents do not provide a compensating usability advantage.
<!-- END GENERATED CONTINUOUS FINAL RESULT -->
