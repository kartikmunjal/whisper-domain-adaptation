# Real Financial Audio Evaluation Plan

Status: locked before evaluation.

## Question

How do the frozen Whisper-small baseline and each of the five preregistered
financial LoRA adapters perform on real earnings-call speech?

## Data

- Source: Earnings-21 (`Revai/earnings21`), CC BY-SA 4.0.
- Evaluation only; no Earnings-21 audio or transcript may enter training,
  checkpoint selection, prompt design, or normalization tuning.
- Select 20 deterministic clips using a committed script and seed 20260729.
- Sample across distinct calls before taking a second clip from any call.
- Clip boundaries and references must come from source alignment metadata.
- Every selected reference is manually checked against its audio. Corrections,
  exclusions, reasons, source revision, and hashes are committed.

## Models and metrics

- Frozen `openai/whisper-small`.
- Financial LoRA seeds 11, 22, 33, 44, and 55.
- Overall, financial-domain, and common-term WER using the committed
  normalization pipeline.
- Report N_trials=5, trial-level 95% CIs, per-clip paired bootstrap CIs with
  10,000 resamples, and all per-clip predictions.
- The 20-clip set is a small external-validity anchor, not a benchmark-quality
  estimate of the full Earnings-21 corpus.

## Amendment 1 — Synthetic-to-real regression diagnosis

Locked 2026-07-31 before computing acoustic or error-transition outcomes.
Compare the full fixed synthetic training manifest with the 20-clip
Earnings-21 anchor using duration, RMS, heuristic SNR, silence fraction,
spectral centroid, spectral tilt, voiced fraction, and voiced pitch range.
Report corpus medians, 10,000-resample 95% CIs for mean differences, and
Cliff's delta; these descriptors do not validate perceptual quality.

On the preregistered common-control slice, align reference, frozen-model, and
each of five adapter hypotheses with word-level dynamic programming. Report
introduced, resolved, and retained substitutions, insertions, and deletions;
top confusion pairs; and errors grouped into function, numeric, financial, and
other word classes. No clip or category is selected after inspecting errors.
An augmented-training ablation is launched only if the acoustic comparison
shows a large clean-speech mismatch (absolute Cliff's delta at least 0.474 for
SNR or silence fraction); its recipe must be locked before training.

## Amendment 2 — Active-frame SNR correction

Locked 2026-07-31 after the first diagnostic exposed an invalid estimator and
before inspecting any corrected value or training an augmentation ablation.
The original report is retained as a failed diagnostic. Its percentile SNR
included exact digital-silence frames at the numerical floor, producing an
implausible synthetic mean of 132.25 dB. The corrected heuristic excludes
frames below both -80 dBFS and 50 dB below the clip peak, then reports the
90th-minus-10th percentile RMS level among the remaining active frames. Clips
with fewer than two active frames are undefined for this metric. Silence
fraction and every other preregistered feature are unchanged. The locked
augmentation gate is reevaluated using corrected active-frame SNR and the
unchanged silence fraction.

If the gate remains met, the sole ablation adds deterministic on-the-fly
acoustic augmentation to the existing synthetic financial training audio:
room impulse response convolution (image-source rectangular rooms, RT60
uniform 0.2–0.8 s), additive stationary noise at SNR uniform 5–25 dB, gain
uniform -6–6 dB, and leading/trailing silence uniform 0–0.5 s. Each transform
is independently applied with probability 0.5 using the trial seed. Text,
splits, optimizer, LoRA configuration, epoch count, checkpoint rule, five
seeds, and Earnings-21 evaluation remain fixed. No augmentation parameter may
be selected using Earnings-21 outcomes.

## Leakage guard

The selection manifest is hashed before inference. Evaluation exits if an
audio or normalized-reference hash overlaps any financial train or validation
manifest.

## Implementation correction log

On 2026-07-29, the first pipeline execution revealed that all 20 deterministically
selected complete speaker turns fell into the common-term slice, making the
preregistered financial-domain split undefined. Those preliminary outputs are
quarantined and are not confirmatory results. Before inspecting any adapter
comparison, selection was corrected to a fixed 10-domain/10-common balance,
where domain membership requires at least one term from the already-committed
`configs/financial_terms.txt`. This rule uses official reference text only, is
applied identically across calls, and never uses an ASR hypothesis or WER. The
vocabulary hash and matched terms are recorded in the generated manifest.
