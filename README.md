[![CI](https://github.com/kartikmunjal/whisper-domain-adaptation/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmunjal/whisper-domain-adaptation/actions/workflows/ci.yml)

# Whisper Domain Adaptation

Reproducible evaluation of Whisper-small LoRA adaptation for specialized
vocabulary, plus preregistered neural-codec and text-to-codec-token studies.

Author: **Kartik Munjal**.

## Research status

The projected financial value has been withdrawn. It is replaced by two direct
five-seed measurements:

- the leakage-controlled but optimistic
  [Edge-TTS result](experiments/results/financial_research/summary.md); and
- the external-validity
  [real Earnings-21 result](experiments/results/earnings21/summary.md).

The real-audio study finds that the synthetic-trained adapters regress overall
relative to frozen Whisper-small; the paired trial interval excludes zero.
Domain change is inconclusive, while the common-control slice exposes a stable
regression. Figures, intervals, and per-seed predictions live in the generated
artifacts rather than being hand-entered here.

The preregistered [synthetic-to-real diagnosis](experiments/results/earnings21/REGRESSION_DIAGNOSIS.md)
finds a large silence-fraction mismatch (Cliff's delta -0.510) after correcting
an invalid silence-floor SNR estimator. That correction reduced the SNR effect
to 0.448, below its locked trigger, but silence still triggered the fixed
augmentation ablation. Five augmented adapters averaged 11.70% real-audio WER
versus 11.47% for the clean-synthetic adapters (paired +0.23 points; the 95%
trial interval crosses zero), while common-control WER worsened by 0.53 points.
The [ablation report](experiments/results/earnings21_augmented/comparison.md)
therefore rejects augmentation as an improvement rather than selecting it.

The previous medical, rank, data-scaling, synthetic-mixture, prefix-tuning, and
catastrophic-forgetting numbers were not accompanied by sufficient primary run
artifacts and are not treated as verified. The completed replacement medical
study uses synthetic train/validation data and 24 evaluation-only English Corti
`med-dictate` recordings. Frozen Whisper-small obtains 24.93% WER; five medical
adapters average 29.80% (95% trial-bootstrap CI 28.89–30.73%), a regression of
4.87 points (3.99–5.87). This negative result is retained rather than selected
away.

## Locked follow-on studies

Three follow-on studies are preregistered and complete:

- [real earnings-call evaluation](REAL_FINANCIAL_AUDIO_PLAN.md) — five-seed
  results complete; manual listening ledger pending;
- [codec rate-distortion and ASR robustness](CODEC_RESEARCH_PLAN.md) — matched
  VQ/FSQ grid, reconstruction, signal metrics, and ASR evaluation complete;
- [text-to-codec-token TTS](TTS_RESEARCH_PLAN.md) — five training and
  end-to-end evaluation trials complete.

Their evaluation sets were isolated until training and selection procedures
were fixed. The linked generated artifacts contain the complete per-seed
predictions and uncertainty calculations; no value below is projected.

Two additional extensions are preregistered and complete:

- [continuous-latent codec comparison](continuous_codec/PREREGISTRATION.md), a
  matched-backbone Gaussian baseline with explicit fixed-width transmission
  cost; and
- [latency/throughput benchmarking](latency_benchmark/PREREGISTRATION.md), with
  synchronized raw timing trials, RTF, p95 latency, and hardware provenance.

Run the unit-tested codec benchmark with:

```bash
python scripts/benchmark_latency.py \
  --checkpoint checkpoints/audio_codec/codec.pt \
  --output experiments/results/latency/codec.json
```

The continuous experiment has separate training and held-out reconstruction
entrypoints (`scripts/train_continuous_codec.py` and
`scripts/evaluate_continuous_codec.py`). Its reconstructed manifest is accepted
by the existing frozen-ASR evaluation pipeline. Parallel TTS timing is provided
by `scripts/benchmark_tts_latency.py`; it rejects autoregressive checkpoints so
the reported condition cannot silently differ from the locked protocol.

<!-- BEGIN GENERATED CONTINUOUS LATENCY RESULT -->
The completed [continuous-codec report](experiments/results/continuous_codec/REPORT.md)
finds that the 400-bps continuous point reaches -21.91 dB (95% CI -32.21–-14.90)
SI-SDR but adds 108.82 WER points (95% CI 85.18–135.39).
Corrected VQ-400 and FSQ-400 add 82.44
and 82.43 points, respectively. The unquantized posterior-mean path still adds
107.64 points, so uniform quantization is not the
primary content-accuracy bottleneck. Archived discrete findings are unchanged.

The completed [latency report](experiments/results/latency_benchmark/REPORT.md)
records corrected-codec 200-ms end-to-end p95 latency of 2.59 ms
(median RTF 0.0113) and parallel-TTS generation-plus-decode
p95 latency of 11.91 ms (median RTF 0.0536) on an
RTX 3070. Both clear the locked chunk-simulation bars, but this is synchronized
batch-one inference—not a production streaming or service-capacity claim.
<!-- END GENERATED CONTINUOUS LATENCY RESULT -->

## Open-corpus curation bridge

The adjacent Audio-Data-Creation repository now exports its locked OpenSLR
SLR31 crawler pilot through the same `id`/`path`/`sentence` Parquet contract.
Frozen Whisper-small and the seed-11 financial adapter were evaluated on all
250 acquired clips; their committed per-clip predictions and the quality-
selection analysis live with the data pipeline, where the intervention belongs.
This repository's WER/OOV code now treats a zero-example domain slice as an
explicit undefined (`NaN`) result instead of crashing, while still reporting
overall/common WER and paired confidence intervals. No SLR31 audio or manifest
is mixed into the original financial training corpus.

A subsequent matched-size study in Audio-Data-Creation did train five new
adapters after replacing half of the 294 financial examples with 147 crawler
clips. On two held-out SLR31 speakers, mean WER worsened from 4.43% to 5.60%
(paired +1.17 points; 95% seed-bootstrap CI +0.88 to +1.46; `N_trials=5`). Real
Earnings-21 moved from 11.47% to 10.83% (-0.64 points, -1.43 to +0.21), an
inconclusive overall change. The locked beneficial gate therefore failed. The
new adapters and result artifacts remain owned by the data-curation study;
this repository supplies the unchanged training and evaluation machinery.

## Confirmatory financial experiment

The experiment uses:

- `openai/whisper-small` with LoRA;
- five seeds: 11, 22, 33, 44, and 55;
- train, validation, and test sets disjoint by both Edge-TTS voice and sentence
  template family;
- an untouched test set used only after checkpoint selection;
- frozen-base and adapted predictions on identical test utterances;
- overall, domain-utterance, and common-control WER;
- 10,000 paired utterance-bootstrap resamples; and
- complete run provenance, manifest hashes, predictions, and adapters.

The confirmatory table in this section uses synthetic evaluation.
**TTS-on-TTS evaluation is optimistic.** The separate Earnings-21 anchor uses
licensed real audio and official transcript/RTTM metadata. Its generated
manual-listening ledger remains explicitly pending, so the repository does not
overstate that validation step.

## Reproduce on an NVIDIA GPU

Use Python 3.11 and install the project in a clean environment:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/pip install -e .
```

On Windows PowerShell:

```powershell
./scripts/run_financial_research.ps1
```

The launcher generates the preregistered corpus, evaluates the frozen baseline,
trains all five seeds, evaluates each adapter, and writes:

```text
data/financial_research/
checkpoints/financial_research/seed_<seed>/
experiments/results/financial_research/baseline_test.json
experiments/results/financial_research/seed_<seed>/finetuned_test.json
experiments/results/financial_research/summary.json
```

The summary is only created when all five seed reports exist and their seed
provenance matches the preregistration.

## Individual commands

Create leakage-controlled data:

```bash
python scripts/prepare_financial_research_data.py
```

Train one development run:

```bash
python scripts/run_finetune.py \
  --config configs/financial_finetune.yaml \
  --train_manifest data/financial_research/train_manifest.parquet \
  --eval_manifest data/financial_research/validation_manifest.parquet \
  --output_dir checkpoints/financial_research/seed_11 \
  --seed 11
```

Evaluate without suppressing inference failures:

```bash
python scripts/evaluate_finetuned.py \
  --adapter_path checkpoints/financial_research/seed_11/adapter \
  --base_model openai/whisper-small \
  --eval_manifest data/financial_research/test_manifest.parquet \
  --domain_vocab configs/financial_terms.txt \
  --baseline_report experiments/results/financial_research/baseline_test.json \
  --seed 11 \
  --output experiments/results/financial_research/seed_11/finetuned_test.json
```

## Repository structure

```text
configs/                    Locked model/training configuration and vocabularies
scripts/                    Data, training, evaluation, and aggregation entrypoints
src/whisper_adapt/data/     Medical, financial, and curation bridges
src/whisper_adapt/models/   Whisper LoRA implementation
src/whisper_adapt/evaluation/ WER, OOV, and uncertainty calculations
tests/                      Unit and leakage-control tests
continuous_codec/           Locked continuous-baseline protocol and configuration
latency_benchmark/          Locked timing protocol and configuration
RESEARCH_PLAN.md            Locked confirmatory protocol
DATA_CARD.md                Data provenance and limitations
```

## Codec and generative TTS work

The repository includes matched-rate VQ-VAE and FSQ codecs at the preregistered
rate grid, inference-only encode/quantize/decode and chunked reconstruction,
empirical bitrate and SI-SDR evaluation with clip bootstrap intervals, and
restart-safe GPU launchers.

The completed [codec report](experiments/results/codec_medical/wer_summary.md)
shows severe task degradation at every tested operating point. Adapted-Whisper
ΔWER ranges from +71.50 to +169.04 percentage points for representative VQ
conditions and from +106.55 to +119.40 points for FSQ. Empirical entropy also
reveals codebook collapse: the nominal 300–500 bps settings realize only
3.3–13.0 bps for FSQ and 47.1–62.3 bps for VQ. Conditional SI-SDR is negative
throughout. These are measured failure modes, not competitive codec claims.

The completed [code-utilization correction](experiments/results/codec_medical_corrective/comparison.md)
adds EMA/dead-code handling for VQ and running per-frame normalization for
FSQ. Across all six five-seed cells, empirical bitrate and SI-SDR improved;
for example, FSQ-300 moved from 12.56 to 56.10 empirical bps and from -57.24
to -29.05 dB. Downstream ASR was mixed: adapted-Whisper ΔWER improved by
23.51–43.27 points at VQ-300/FSQ-300, was inconclusive at FSQ-400/500, and
worsened by 10.21 and 23.53 points at VQ-400/500. Because realized bitrates
also changed substantially, this is evidence that collapse was mitigated, not
a matched-bitrate claim that the corrected codecs are competitive. All six
conditions still have negative SI-SDR and large positive ΔWER.

The extended [200–500 bps rate-distortion report](experiments/results/codec_medical_extended/REPORT.md)
adds 200/250-bps VQ and FSQ trials and an 80-bin log-mel distance, a
phase-insensitive complement to SI-SDR. At the four lower-rate representative
points, adapted-Whisper ΔWER remains +70.20 to +94.58 percentage points.
VQ-250 has the best lower-grid SI-SDR (-23.01 dB) and log-mel distance
(12.98 dB), while FSQ remains less stable. Across the full grid, log-mel
distance generally improves with empirical bitrate even where SI-SDR and WER
do not rank systems identically; this quantifies the previously unexplained
metric disagreement without claiming perceptual quality.

<!-- BEGIN GENERATED ENCODEC RESULT -->
The preregistered [external EnCodec benchmark](experiments/results/codec_medical_encodec/REPORT.md)
anchors those results to the pinned open `facebook/encodec_24khz` checkpoint.
At its lowest supported rate, 1.5 kbps, EnCodec's
five-adapter mean absolute WER is 47.42% (95% CI 46.00–48.85), and its reconstructed-minus-
original ΔWER is +17.62 points (95% CI 15.74–19.50). At 500 nominal bps,
VQ-VAE and FSQ have 2.63× (95% CI 2.41–2.89) and 2.85× (95% CI 2.58–3.11) EnCodec's
absolute WER, respectively. Signal fidelity and utilization show the same gap:
EnCodec reaches -2.62 dB SI-SDR and 64.2%
entropy utilization, versus -15.73 dB/40.1% for
VQ-VAE and -28.44 dB/13.4% for FSQ. This is an
external anchor, not a matched-rate ranking: EnCodec receives 3× the nominal
bitrate and 4.80×/14.40× the measured empirical entropy rate of
the VQ-VAE/FSQ cells. The elevated but improved custom-codec utilization is
therefore a plausible mechanism for part of the gap, not proof that utilization
alone causes it.
<!-- END GENERATED ENCODEC RESULT -->

It also includes a small autoregressive Transformer mapping UTF-8 text tokens
to VQ codec tokens. The held-out path is
text → predicted tokens → codec decoder → waveform, followed by round-trip
Whisper WER and comparison with the corresponding Edge-TTS clips. No TTS claim
is inferred from teacher-forced validation alone.

The duplicate-safe [five-seed before report](experiments/results/codec_tts_unique_before/summary.md)
finds mean codec-TTS round-trip WER of 1021.68% (95% trial-bootstrap CI
743.45–1326.60%) versus 1.18% for paired Edge-TTS. WER can exceed 100% because
insertions outnumber reference words. A monitoring audit found four repeated
source IDs in 98 rows; stable row-indexed filenames now prevent silent waveform
overwrites, and both conditions were regenerated with that fix.

The preregistered [duration-control correction](experiments/results/codec_tts_corrective/comparison.md)
reduced mean absolute codec-token length error from 217.71 to 69.79 tokens
(paired change -147.91, 95% trial-bootstrap CI -278.70 to -48.38), but did not
improve the primary outcome: overall WER changed to 1151.24%, a paired
+129.56-point change with a wide CI (-432.97 to +546.02). Domain WER was also
not improved, and common-term WER worsened. The repetition penalty remains
implemented but was disabled after its locked tiny-set diagnostic increased
token error from 30.08% to 43.55%. Thus the correction controls duration, not
intelligibility; this small autoregressive model remains decisively
noncompetitive.

The follow-up [text-conditioning diagnosis](experiments/results/codec_tts_text_only/CONDITIONING_REPORT.md)
confirmed the autoregressive shortcut: shuffled text changed teacher-forced
NLL by only 0.00029 nats/token. A preregistered text-forced parallel decoder
then removed codec-token history entirely. It reduced mean round-trip WER to
291.34% (95% trial-bootstrap CI 148.23–534.16%) with 100% EOS completion, but
remained far behind Edge-TTS at 1.18%. Its shuffled-text NLL difference rose
to 0.0255 but still failed the locked 0.05 conditioning gate, and mean SI-SDR
was -43.56 dB. The replacement is therefore reported as a useful root-cause
intervention and negative generative result, not a successful TTS system.

The completed [capacity/data/phoneme scale study](experiments/results/codec_tts_scale_study/REPORT.md)
then expanded the training split from 294 to 1,774 paired clips and scaled the
parallel decoder to approximately 25.7M trainable parameters. Across the same
five seeds, byte conditioning reduced mean overall WER from 291.34% to 240.12%
(paired change -51.22 percentage points; 95% seed-bootstrap CI -269.76 to
+127.39), but its shuffled-minus-true NLL remained 0.0385 and still failed the
locked 0.05 conditioning gate. Deterministic CMUdict phonemes raised that
diagnostic to 0.0871 (0.0626–0.1125), clearing the gate, yet overall WER was
273.98% (138.46–437.70%). Relative to scaled bytes, the paired overall change
was +33.86 points (-54.39 to +151.88), while domain WER worsened by +65.79
points (+0.86 to +171.28). Thus explicit phonemes repair measured text
sensitivity but do not repair codec-token content accuracy.

The fixed open comparator, Piper 1.4.2 `en_US-lessac-low`, reaches 2.86%
overall WER (2.76–2.95), versus 1.18% (1.06–1.27) for Edge-TTS. Both learned
systems generated all 490 seed-by-clip outputs without execution failure, but
their mean conditional SI-SDR remained -44.01 dB for bytes and -43.00 dB for
phonemes. These are TTS-on-TTS held-out sentences transcribed by five frozen
financial adapters, so the absolute comparator WER is optimistic and the five
values are adapter trials, not five independent audio datasets. The study
supports a mechanical conditioning diagnosis and a negative scaling
result—not usable or competitive TTS. Compact per-seed reports, hashes,
data-generation records, and 10,000-resample intervals are committed;
generated audio and model weights are not.

<!-- BEGIN GENERATED ELEVENLABS RESULT -->
The fixed ElevenLabs `eleven_multilingual_v2` comparator on the same 98
held-out sentences reaches 0.484% (95% seed-bootstrap CI 0.310–0.658) overall,
0.522% (95% seed-bootstrap CI 0.332–0.712) on domain sentences, and
0.000% (95% seed-bootstrap CI 0.000–0.000) on common controls across the same 5
frozen adapters. It is included beside Piper and Edge-TTS in the
[scale-study table](experiments/results/codec_tts_scale_study/REPORT.md).
This round-trip content metric does not measure naturalness or preference, and
the TTS-on-TTS evaluation remains optimistic.
<!-- END GENERATED ELEVENLABS RESULT -->

## License

Code is released under the MIT License. Dataset and model artifacts retain
their upstream licenses; consult their source cards before redistribution.
