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

## Final result

All five paired trials completed under the duplicate-safe evaluator. Duration
control reduced mean absolute sequence-length error from 217.71 to 69.79
tokens (paired change -147.91; 95% trial-bootstrap CI -278.70 to -48.38), but
overall round-trip WER changed from 1021.68% to 1151.24% (paired +129.56
percentage points; -432.97 to +546.02). Domain WER changed from 1057.93% to
1111.81%, and common WER worsened from 588.00% to 1623.00%. The corrective
study therefore supports a duration-control claim only, not improved speech
intelligibility or TTS quality.

## Amendment 3 — Content-conditioning root-cause gate

Locked 2026-07-31 before inspecting conditioning or attention diagnostics.
Using the same held-out token test partition and all five corrective
checkpoints, the diagnostic compares each example with a deterministic
different-text permutation. It reports true-minus-shuffled teacher-forced
NLL, normalized edit distance between true-text and shuffled-text free-running
sequences, token error in ten normalized reference-position bins, and each
decoder layer's cross-attention centroid monotonicity.

Conditioning is classified as broken if the five-seed mean shuffled-minus-true
NLL advantage is at most 0.05 nats/token or if generated-sequence sensitivity
is at most 5%. Attention plots are descriptive and cannot override those
numeric gates. If either gate fails, only a targeted conditioning/masking fix
is permitted. If both pass while free-running errors grow with position, the
next model is a duration-aware non-autoregressive predictor with output length
fixed before token prediction. Any replacement retains the same data, frozen
codec, five seeds, Edge-TTS comparator, WER splits, and uncertainty protocol.

## Amendment 4 — Text-forced parallel decoder

Locked 2026-07-31 after all five Amendment 3 diagnostics completed and before
replacement training. The shuffled-minus-true NLL was 0.00029 nats/token
(95% trial-bootstrap CI -0.00104 to 0.00176), triggering the conditioning
failure gate. Mean cross-attention entropy was 0.9713 and centroid
monotonicity was -0.058, supporting the mechanical diagnosis that
teacher-forced codec history provides a shortcut around text.

The targeted replacement removes codec-token identities from decoder inputs.
Training and inference use learned output-position queries cross-attending to
the encoded text, without a causal mask; target length is supplied during
training and fixed from the existing duration head at inference. Thus codec
content cannot be predicted from previous ground-truth or generated codec
tokens. Scheduled sampling is disabled because the model has no autoregressive
history. All other architecture dimensions, data and splits, frozen VQ codec,
duration loss and cap, optimizer and epochs, five seeds, checkpoint selection,
Edge-TTS comparator, Whisper evaluation, failure accounting, and uncertainty
protocol remain fixed. The same shuffled-text diagnostic is rerun on the five
replacement checkpoints before round-trip claims are interpreted.

## Amendment 5 — Capacity, paired-data, and phoneme scale study

Locked 2026-08-01 after Amendment 4 completed and before generating expanded
training audio or observing any scale-study validation/test outcome. The
text-forced model reduced mean round-trip WER from 1151.24% to 291.34%, but
its mean shuffled-minus-true NLL remained 0.0255 nats/token, below the locked
0.05 conditioning gate. Duration is therefore not used to select this study.

Two sequential interventions are evaluated; neither may be tuned on held-out
WER. Stage A retains byte conditioning and the parallel no-history decoder,
expands only the training split from 294 clips to a fixed target of 1,774 clips
(the original 294 plus eight new carrier templates crossed with every fixed
financial term and four existing training voices, plus fixed common controls),
and scales the Transformer from d_model 256, 4+4 layers, FFN 1024 to d_model
384, 6+6 layers, FFN 1536, 8 heads. Validation and test manifests, voices,
template families, codec, duration loss/cap, optimizer family, checkpoint
criterion, and five seeds remain unchanged. Training is 30 epochs, effective
batch 16, learning rate 3e-4, with no scheduled sampling.

Stage B changes only the text representation to deterministic CMUdict ARPAbet
phonemes (cmudict 1.1.1, first listed pronunciation, word-boundary tokens;
explicit grapheme fallback for out-of-vocabulary words). The phone vocabulary,
CMUdict version, OOV counts, manifests, and hashes are recorded. Stage B uses
the same capacity, data, training budget, seeds, selection, and evaluation as
Stage A. Each stage independently reruns the existing shuffled-text gate,
position errors, attention plots, generation failures, SI-SDR, and paired
overall/domain/common round-trip WER with 10,000-resample intervals.

The fixed external comparator is Piper 1.4.2 with the independently pretrained
`en_US-lessac-low` 16-kHz voice from `rhasspy/piper-voices`, synthesized once
on the identical 98 held-out sentences and transcribed by all five frozen
financial adapters. Model/config revisions and SHA-256 hashes are recorded;
weights and generated audio are not redistributed. Edge-TTS remains the upper
reference. Comparators contextualize distance from usable TTS and are not
used for model selection.

### Amendment 5 execution correction — GPU-tested micro-batch

Locked 2026-08-01 after a synthetic forward/backward memory smoke test and
before any Stage A or B training began. The 25.7M-parameter model used 346 MiB
peak allocated CUDA memory at batch 2 and 400 output tokens on the RTX 3070.
The execution setting is therefore fixed at micro-batch 8 with gradient
accumulation 2, preserving the preregistered effective batch 16 while avoiding
an unnecessarily slow run. Model, data, epochs, optimizer, learning rate,
selection, and every evaluation rule are unchanged. No validation or test
metric informed this correction.

## Amendment 5 final result

All ten learned-model trials and all five Piper adapter evaluations completed
on the frozen 98-sentence test manifest. Every learned trial produced 98/98
waveforms with zero generation failures. The committed
[scale-study report](experiments/results/codec_tts_scale_study/REPORT.md) and
[machine-readable summary](experiments/results/codec_tts_scale_study/summary.json)
regenerate the following conclusions from seed-level primary reports with
10,000 bootstrap resamples.

Stage A (larger model plus 1,774 training pairs, byte conditioning) changed
overall WER from 291.34% to 240.12%; the paired change was -51.22 percentage
points (95% CI -269.76 to +127.39). Domain WER changed by -79.41 points
(-292.31 to +66.31). Neither interval excludes zero. The shuffled-minus-true
NLL was 0.0385 (0.0267–0.0528), so Stage A still fails the locked 0.05 primary
conditioning gate even though generated-sequence sensitivity was 0.2609.

Stage B (phonemes only) achieved shuffled-minus-true NLL 0.0871
(0.0626–0.1125) and generated-sequence sensitivity 0.2390
(0.1922–0.3001), clearing both locked gates. Its overall WER was nevertheless
273.98% (138.46–437.70%). Against paired Stage A seeds, overall WER changed by
+33.86 points (-54.39 to +151.88) and domain WER worsened by +65.79 points
(+0.86 to +171.28). Passing the adequacy gate is therefore necessary but not
sufficient for phonetic codec-token accuracy. Validation NLL selects very
early epochs while later epochs overfit, and similar validation losses coexist
with widely varying WER; teacher-forced validation loss is not a proxy for
usable speech.

Piper 1.4.2 `en_US-lessac-low` produced 2.86% overall WER (2.76–2.95) and
Edge-TTS produced 1.18% (1.06–1.27), compared with 240.12% for scaled bytes and
273.98% for scaled phonemes. Mean conditional SI-SDR remained -44.01 dB and
-43.00 dB, respectively. The held-out audio and training pairs are synthesized,
so this TTS-on-TTS evaluation is optimistic; the five WER trials reflect five
frozen ASR adapters over the same audio, not independent speech samples. No
real-audio, perceptual-quality, usable-TTS, or competitive-baseline claim is
made. The mechanistic conclusion is narrower: additional capacity/data alone
did not reliably clear the conditioning gate, phonemes did clear it, and
neither intervention solved content generation.

## Amendment 6 — Fixed ElevenLabs API comparator

Locked 2026-08-03 before making any ElevenLabs synthesis request or observing
any ElevenLabs audio or ASR outcome. The scale study adds one commercial API
comparator without changing the frozen 98-sentence financial test manifest,
the five financial adapters (seeds 11, 22, 33, 44, and 55), domain vocabulary,
Whisper-small base model, normalization, overall/domain/common splits, or
10,000-resample seed-level interval procedure.

The fixed synthesizer is ElevenLabs `eleven_multilingual_v2`, voice ID
`JBFqnCBsd6RMkjVDRZzb`, with stability 0.50, similarity boost 0.75, style 0.00,
speaker boost enabled, and `mp3_44100_128` output. Each sentence is submitted
independently through the documented text-to-speech endpoint. Requests are
restart-safe, use bounded exponential retry for transient HTTP failures, and
store no API key in arguments, reports, manifests, logs, or Git. The generated
audio is not redistributed. The report records endpoint/model/voice/settings,
input-manifest hash, per-file hashes, response request IDs when supplied, and
clean repository provenance.

The 98 generated clips are transcribed once by each of the same five frozen
financial adapters. ElevenLabs is added to the existing scale-study table with
mean overall/domain/common WER and 95% seed-bootstrap intervals. It is an
external contextual comparator, not a training intervention and not a model-
selection input. Because the held-out text and adapter training data originate
from synthetic financial speech, the existing TTS-on-TTS optimism caveat
remains. No human preference, naturalness, speaker-similarity, latency, cost,
or real-audio claim is inferred from round-trip WER.
