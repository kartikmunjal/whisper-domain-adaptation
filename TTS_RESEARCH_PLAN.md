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

## Amendment 1 — Duration-control corrective study

Locked 2026-07-31 after the original five-seed result showed runaway
insertions and before the corrective tiny-set outcome was observed. The
original checkpoints and reports remain immutable and are the "before"
condition.

The diagnostic first trains on the 16 shortest training sequences (stable
length/ID ordering) for 1,000 optimizer steps and measures free-running token
error, exact match, and EOS rate. This is a memorization test, not a
generalization result. Failure to memorize triggers implementation debugging;
it is not repaired by adding more data.

The fixed corrective model and decoding changes are:

- a text-conditioned head predicting `log1p(codec_token_length)`, trained with
  weight 0.1 alongside token cross-entropy;
- scheduled sampling increasing linearly from 0 to 0.25 over the first half of
  the 50 training epochs, using a detached first-pass argmax;
- an inference cap of 1.25 times the rounded predicted length, bounded by the
  existing model maximum;
- EOS remains explicitly supervised and finished batch elements are forced to
  remain at EOS; and
- a frequency repetition penalty of 0.5 logit units per prior occurrence.

The same train/validation/test partitions, frozen VQ codec, five seeds,
checkpoint-selection metric, Edge-TTS comparator, Whisper adapters, WER
splits, generation-failure accounting, and 10,000-resample uncertainty
protocol are retained. The before/after report must include duration error,
EOS/failure rate, conditional SI-SDR, and round-trip WER. Improvements are not
claimed from the tiny-set diagnostic or teacher-forced metrics alone.

## Amendment 2 — Diagnostic-gated repetition ablation

Locked 2026-07-31 after the Amendment 1 tiny-set diagnostics and before any
corrective validation/test outcome or five-seed corrective training run.

On the fixed 16-example diagnostic, the full corrective recipe increased
free-running token error from 30.08% to 43.55% and reduced exact match from
50% to 0%. The prespecified component checks attributed this failure to the
frequency repetition penalty: duration control plus scheduled sampling with
the penalty disabled produced 28.49% token error and 50% exact match, while
duration control alone produced 30.28% and 43.75%. EOS rate was 100% in every
condition. These are diagnostic selection results, not held-out evidence.

The repetition-penalty implementation and failed result are retained, but the
confirmatory five-seed corrective comparison fixes the penalty at 0.0. All
other Amendment 1 settings and outcomes remain unchanged. No further setting
may be chosen from validation or test WER.
