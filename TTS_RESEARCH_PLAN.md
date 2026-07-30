# Text-to-Codec-Token TTS Plan

Status: locked before generative-model training.

## Question

Can a small autoregressive text-conditioned model predict VQ codec tokens
well enough to synthesize intelligible held-out financial speech, and how does
its round-trip ASR accuracy compare with Edge-TTS?

## Data and split

- Paired source: the existing 14-voice financial Edge-TTS generator.
- Train, validation, and test are disjoint by voice and sentence-template
  family.
- Test text, voices, and template families are not used for codec or token
  model selection.
- Input text is normalized by a committed deterministic pipeline.

## Model

- Freeze one preregistered VQ codec selected without round-trip test WER.
- Encode training audio to discrete targets.
- Train a causal transformer conditioned on text tokens.
- Five seeds: 11, 22, 33, 44, 55.
- Validation token negative log likelihood selects checkpoints.
- Decoding parameters are fixed before test synthesis.

## Outcomes

- Primary: held-out round-trip domain-term WER using the five financial LoRA
  adapters without choosing the best adapter.
- Secondary: overall/common WER, token accuracy, sequence length error,
  SI-SDR where a paired reference voice exists, and generation failure rate.
- Comparator: Edge-TTS on the identical held-out sentences and target voices.
- Report N_trials=5 and 95% CIs, plus 10,000 paired sentence-bootstrap
  intervals for model-minus-Edge-TTS deltas.

## Claim boundary

This is a narrow-domain neural-codec TTS experiment trained from synthetic
paired speech. It is not evidence of production-quality, natural, or
zero-shot multi-speaker TTS. Listening examples are supplemental and never
replace preregistered quantitative outcomes.
