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
