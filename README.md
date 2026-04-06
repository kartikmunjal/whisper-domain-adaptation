# whisper-domain-adaptation

**Fine-tuning Whisper on domain-specific vocabulary — medical and financial speech**

Base Whisper is an impressive general-purpose ASR model. It handles accents, background noise, and
conversational speech reasonably well. But put it in a radiology dictation room or on an earnings
call, and it falls apart. A cardiologist saying "the patient presents with esophagogastroduodenoscopy
findings consistent with Barrett's oesophagus" will get back something like "the patient presents
with echo gastro due odd any findings consistent with Barret's esophagus." Comprehensible, barely,
but not acceptable for clinical documentation where every word matters.

This project fine-tunes Whisper-small using LoRA on two domain-specific datasets:
- **Medical**: real clinical dictation audio from a HuggingFace dataset, quality-filtered using the
  same pipeline as the Audio-Data-Creation project
- **Financial**: synthesized earnings call audio via Edge-TTS, using the 14-voice catalog from the
  rlhf-and-reward-modelling-alt project

The result is a pair of domain-adapted ASR models that cut domain-term WER roughly in half while
preserving performance on general speech.

---

## Results

| Metric | Medical Baseline | Medical Fine-tuned | Financial Baseline | Financial Fine-tuned |
|---|---|---|---|---|
| WER (overall) | 34.1% | **18.3%** | 28.4% | **14.2%** |
| WER (domain terms) | 48.7% | **22.1%** | 52.1% | **26.7%** |
| WER (common terms) | 19.4% | 15.8% | 14.2% | 11.1% |
| LibriSpeech test-clean WER | 4.3% | 5.1% | 4.3% | 4.7% |

The LibriSpeech row is the catastrophic forgetting check — the fine-tuned model degrades by less
than 1 percentage point on general English speech, which I'm comfortable calling acceptable.

Domain term WER is the number I actually care about. Going from 48.7% → 22.1% on medical terms
means the model now gets the majority of hard terms right, where before it was getting more than
half wrong. There's still room to improve — 22% is not good enough for unreviewed clinical use —
but it's a meaningful step.

---

## The Three-Repo System

This project sits in the middle of a three-repo pipeline, with data flowing in both directions:

```
┌─────────────────────────────────────┐
│        Audio-Data-Creation          │
│  quality filter → dedup → diversity │
└──────────────┬──────────────────────┘
               │  filtered_manifest.parquet
(1) forward    │  import_from_curation.py
               ▼
┌─────────────────────────────────────┐
│     whisper-domain-adaptation       │  ← this repo
│  LoRA fine-tune on domain corpus    │
└──────────────┬──────────────────────┘
               │  checkpoints/*/adapter
(2) backward   │  evaluate_with_domain_model.py
               ▼
┌─────────────────────────────────────┐
│        Audio-Data-Creation          │
│  AblationEvaluator(fine_tuned_model │
│  _path=...) for domain-accurate WER │
└─────────────────────────────────────┘

rlhf-and-reward-modelling-alt
  │  14-voice TTS catalog
  └──> financial speech synthesis
```

**The forward direction** (`import_from_curation.py`): Audio-Data-Creation outputs a
quality-filtered `filtered_manifest.parquet`. Because both projects use identical manifest schemas
(`id, path, sentence, duration_sec, snr_db, silence_ratio, source`), no format conversion is
needed. `src/whisper_adapt/data/curation_bridge.py` adds domain/general splitting and optional
domain oversampling, then writes train/eval splits ready for `run_finetune.py`:

```bash
# After a curation run in Audio-Data-Creation:
python scripts/import_from_curation.py \
    --manifest ../Audio-Data-Creation/outputs/filtered_manifest.parquet \
    --domain_vocab configs/medical_terms.txt \
    --output_dir data/medical_curated \
    --domain_oversample 2.0
```

**The backward direction** (`evaluate_with_domain_model.py` in Audio-Data-Creation): once a
fine-tuned adapter exists, Audio-Data-Creation's `AblationEvaluator` can use it instead of base
Whisper to evaluate subsequent curation runs. Base Whisper WER on medical speech is ~34% even on
perfectly clean audio (OOV inflation) — differences between ablation splits get lost in the noise.
The fine-tuned model's WER actually reflects data quality:

```bash
# In Audio-Data-Creation, after fine-tuning here:
python scripts/evaluate_with_domain_model.py \
    --manifest outputs/filtered_manifest.parquet \
    --model_path ../whisper-domain-adaptation/checkpoints/medical/adapter \
    --compare_base \
    --output experiments/results/domain_eval.json
```

Each curation iteration feeds a better fine-tuned model, which gives cleaner WER signal, which
guides better curation decisions on the next pass.

**Audio-Data-Creation** also contains the reference implementation of the quality filtering pipeline —
SNR estimation, silence ratio detection, duration bounds, clipping checks. The thresholds used
here (`min_snr_db=15`, `max_silence_ratio=0.40`, `min_duration_sec=0.5`) are identical to what
that project uses for Common Voice curation. I deliberately kept them in sync so that if someone
updates the thresholds over there they can just propagate them here.

**rlhf-and-reward-modelling-alt** has a TTS synthesis pipeline (Extension 11) that maintains a
catalog of 14 Edge-TTS voices spanning different genders, accents, and prosodic styles. The
financial speech dataset here uses that exact voice catalog — same voice names, same round-robin
voice selection strategy. The reasoning is that if we're doing RLHF on synthesized speech quality
over there, the ASR model being evaluated should ideally match the synthesis distribution used
in training data here.

---

## Why Base Whisper Fails on Domain Speech

The short version: Whisper's language model prior heavily discounts rare tokens.

Whisper is a sequence-to-sequence model. The encoder processes audio features; the decoder generates
transcript tokens autoregressively. During decoding, the model combines acoustic evidence from the
cross-attention with the language model prior learned during training on 680k hours of web audio.

For general vocabulary, this works great — the prior acts as a smoothing regularizer that prevents
phonetically ambiguous words from being transcribed nonsensically. But for domain vocabulary, the
prior actively works against correct transcription. "Esophagogastroduodenoscopy" appears essentially
zero times in web audio. "Cholecystectomy" is similarly absent. The model knows roughly what the
word sounds like acoustically — the encoder activations will light up in roughly the right region
of token space — but the decoder assigns such a low prior probability to these tokens that the
acoustic signal gets overridden.

The result is characteristic domain errors:
- Polysyllabic words get split: "echocardiogram" → "echo cardio gram"
- Rare words get substituted with phonetically similar common ones: "cholecystectomy" →
  "colas to me" (approximately)
- Acronyms get spelled out phonetically: "EBITDA" → "EB it da"
- Technical compound terms get each component interpreted separately: "yield curve inversion" →
  "yield curve invasion" (common word substitution)

LoRA fine-tuning on domain-specific audio retrains the decoder's attention projections and feed-
forward layers to shift the prior probability toward domain tokens. After training, the model has
seen enough examples of "echocardiogram" in context that the cross-attention signal is sufficient
to tip the decoder toward the correct token sequence.

---

## Model Architecture

### Base Model

**Whisper-small** (openai/whisper-small) — 244M parameters total.

I tried Whisper-tiny first. It was fast but the base WER was too high (domain term WER >65%) for
the fine-tuning results to be compelling. Whisper-medium is probably better still but I didn't
have the compute budget to run it. Whisper-small is the sweet spot for this project.

### LoRA Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Rank (r) | 32 | Higher than text tasks; audio cross-entropy is more rank-sensitive |
| Alpha | 64 | Standard 2× alpha/r scaling |
| Dropout | 0.05 | Minimal; domain datasets are small enough that regularization matters |
| Target modules | q_proj, v_proj, k_proj, out_proj, fc1, fc2 | See below |
| Trainable params | 7.1M / 244M | 2.9% of total parameters |

**Why LoRA instead of full fine-tuning?**

Full fine-tuning 244M parameters on 8-10k samples would overfit badly. More practically, the
catastrophic forgetting problem is severe with full fine-tuning on small domain datasets — I ran a
quick sanity check with full fine-tuning on 1k medical samples and LibriSpeech WER went from 4.3%
to 11.7%. LoRA's low-rank constraint acts as a form of regularization that forces the model
updates to live in a subspace that doesn't completely disrupt the general representations.

**Why rank 32?**

The rlhf-and-reward-modelling-alt project used r=16 for text alignment tasks (Extension 4) and got
results essentially matching full fine-tuning. I ran ablations at r=8, 16, 32 on the medical
dataset. The pattern:
- r=8: domain WER 28.4% (good, but leaves performance on the table)
- r=16: domain WER 24.1%
- r=32: domain WER 22.1%
- r=64: domain WER 22.0% (not worth 2× the parameters)

Audio appears more rank-sensitive than text. My interpretation: the log-mel spectrogram has much
higher intrinsic dimensionality than token embeddings, so the cross-attention projections need
higher rank to capture the relevant acoustic-semantic correspondences.

**Why those target modules?**

The conventional wisdom for Whisper LoRA is to target only the decoder attention (q_proj, v_proj)
and leave the encoder alone. I tested this and it left ~4 percentage points of domain WER
improvement on the table.

The encoder acoustic representations are already decent — the model can "hear" the rare word.
The problem is in the decoder's language model prior, which is distributed across:
1. Cross-attention (q_proj, k_proj, v_proj, out_proj) — where acoustic evidence meets LM prior
2. Feed-forward layers (fc1, fc2) — where token probabilities are effectively stored

Adding fc1/fc2 to the target modules gave a consistent 3-4% relative improvement in domain term
WER across both domains. The adapter parameter count goes up but it's still a small fraction of
the total model.

---

## Data Pipeline

### Medical Domain

**Source**: `speech-recognition/medical-speech-transcription` on HuggingFace Hub

This dataset contains ~10,000 de-identified medical dictation clips — clinical notes, discharge
summaries, radiology reports. The audio was recorded by medical professionals in clinical settings
(some office noise, variable mic quality, occasional masking). After quality filtering, about 80%
of clips pass, giving 8,432 training samples and 937 eval samples.

The quality filtering pipeline (implemented in `src/whisper_adapt/data/medical.py`) applies:

1. **SNR >= 15 dB** — reference-free SNR estimation using 10th/90th percentile frame RMS.
   This threshold matches what Audio-Data-Creation uses for Common Voice validation filtering.

2. **Duration 0.5 – 30.0 seconds** — excludes near-empty clips and very long dictations.
   Medical dictation can run long; 30 seconds covers most clinical notes.

3. **Silence ratio < 0.40** — rejects clips where more than 40% of frames are more than 30 dB
   below the peak. Catches recordings where the speaker paused before starting.

4. **Clipping < 0.1%** — rejects clips with significant amplitude clipping, which introduces
   harmonics that confuse the spectrogram.

The filtered dataset is written to disk as individual 16 kHz mono WAV files with a Parquet
manifest (`data/medical/train_manifest.parquet`, `data/medical/eval_manifest.parquet`).

Vocabulary coverage: 60+ medical terms across 10 specialties. Not every term appears in every
split — the rarer specialties (neurology pharmacology) are underrepresented. See the data card
for term frequency breakdown.

### Financial Domain

**Source**: Synthesized via Edge-TTS from earnings call sentence templates

Earnings call audio with aligned transcripts is surprisingly hard to get. The audio exists (Q4
calls are public) but extracting properly aligned transcripts with speaker diarization is a
research project in itself. SEC filings have transcripts but not audio.

The solution: write context sentence templates that embed each financial term, then synthesize
audio using Edge-TTS across all 14 voices in the catalog. The synthesis-then-filter approach
gives clean, well-aligned training pairs. The obvious downside is that TTS audio is cleaner and
more prosodically uniform than real earnings call speech — see limitations section.

Sentences are generated from templates like:
- "Our {term} for the quarter came in at the high end of guidance."
- "Turning to {term}, we saw meaningful improvement versus prior year."

Each term appears in multiple sentence contexts with different voices. Tier-2 terms (more
technical, higher base WER) get more sentence variants to compensate for their difficulty.

After synthesis and quality filtering (pass rate ~98%, since TTS is clean), the dataset has
1,340 training samples and 237 eval samples. Smaller than medical, but the WER improvement is
larger — possibly because the eval set is also synthesized, making the distribution match better.

```
data/
├── medical/
│   ├── train_manifest.parquet      # 8,432 clips after quality filter
│   ├── eval_manifest.parquet       # 937 clips
│   └── wavs/                       # individual 16kHz mono WAV files
├── financial_synth/
│   ├── train_manifest.parquet      # 1,340 clips
│   ├── eval_manifest.parquet       # 237 clips
│   └── wavs/                       # synthesized WAVs
```

---

## Quick Start

Prerequisites: Python 3.10+, ~8 GB VRAM (A100 preferred, runs on 3090/4090), or MPS for
slower CPU/MPS runs.

```bash
# 1. Clone and install
git clone <repo>
cd whisper-domain-adaptation
pip install -e .

# 2. Prepare medical data (downloads from HuggingFace, ~2 hours)
python scripts/prepare_medical_data.py \
    --output_dir data/medical

# 3. Prepare financial data (TTS synthesis, ~30 minutes)
python scripts/prepare_financial_data.py \
    --output_dir data/financial_synth

# 4. Evaluate baseline
python scripts/evaluate_baseline.py \
    --eval_manifest data/medical/eval_manifest.parquet \
    --domain_vocab configs/medical_terms.txt \
    --output experiments/results/medical/baseline_wer.json

# 5. Fine-tune
python scripts/run_finetune.py \
    --config configs/medical_finetune.yaml \
    --train_manifest data/medical/train_manifest.parquet \
    --eval_manifest data/medical/eval_manifest.parquet

# 6. Evaluate fine-tuned model
python scripts/evaluate_finetuned.py \
    --adapter_path checkpoints/medical/adapter \
    --eval_manifest data/medical/eval_manifest.parquet \
    --domain_vocab configs/medical_terms.txt \
    --output experiments/results/medical/finetuned_wer.json
```

For financial domain, swap `medical` → `financial` and use the financial config/manifests.

---

## Scripts

### `scripts/prepare_medical_data.py`

Downloads `speech-recognition/medical-speech-transcription` from the HuggingFace Hub,
resamples to 16 kHz, runs quality filtering, and writes `train_manifest.parquet` /
`eval_manifest.parquet`.

The download is ~20 GB and takes a while. Filtered WAV files add another ~15 GB. Make sure
you have space.

```bash
python scripts/prepare_medical_data.py \
    --output_dir data/medical \
    --max_train_samples 10000   # optional cap for faster iteration
```

### `scripts/prepare_financial_data.py`

Generates sentence templates, synthesizes audio via Edge-TTS for all 14 voices, runs quality
filtering, and writes the financial manifest.

```bash
python scripts/prepare_financial_data.py \
    --output_dir data/financial_synth \
    --dry_run   # preview without actually synthesizing
```

Requires an internet connection for Edge-TTS (it calls the Microsoft TTS API). The synthesis
is cached on disk so reruns are fast.

### `scripts/evaluate_baseline.py`

Runs the base Whisper-small model on an eval manifest and produces a JSON report with overall
WER, domain-term WER, common-term WER, and per-term breakdown. Run this before fine-tuning to
establish the baseline numbers.

```bash
python scripts/evaluate_baseline.py \
    --eval_manifest data/medical/eval_manifest.parquet \
    --domain_vocab configs/medical_terms.txt \
    --output experiments/results/medical/baseline_wer.json
```

### `scripts/run_finetune.py`

The main training script. Takes a YAML config, train/eval manifests, and optional CLI
overrides. Handles feature extraction, model setup (load base → attach LoRA → freeze base),
training loop via Seq2SeqTrainer, and saves the best checkpoint.

The YAML configs in `configs/` are the reference configs used for the results in this README.

### `scripts/evaluate_finetuned.py`

Same as evaluate_baseline.py but loads the LoRA adapter and optionally merges weights for
faster inference. Produces the "after" JSON for the before/after comparison.

### `scripts/run_ablations.py`

Runs two ablation studies:
- `data_scaling`: trains on 10/25/50/75/100% of the training set to show the data scaling curve
- `synthetic_mix`: varies the ratio of real-to-synthetic data (financial domain only)

Each ablation run is independent — they can be parallelized if you have multiple GPUs.

### `scripts/import_from_curation.py`

Imports an Audio-Data-Creation `filtered_manifest.parquet` directly into this project's data
pipeline. No format conversion needed — the manifest schema is shared. The script handles:

- Optional secondary quality gates (stricter SNR or duration cap for Whisper)
- Domain/general split using a vocabulary file (`configs/medical_terms.txt`, etc.)
- Domain oversampling to compensate for domain utterances being a minority of the corpus
- Eval set drawn from domain utterances only (so the eval WER reflects domain performance)

```bash
python scripts/import_from_curation.py \
    --manifest ../Audio-Data-Creation/outputs/filtered_manifest.parquet \
    --domain_vocab configs/medical_terms.txt \
    --output_dir data/medical_curated \
    --domain_oversample 2.0 \
    --eval_fraction 0.1
```

Outputs `train_manifest.parquet`, `eval_manifest.parquet`, and `import_report.json` (same
format as Audio-Data-Creation's `curation_report.json` for direct comparison).

---

## OOV Analysis: Hardest Terms

These are the terms with the highest base Whisper WER and the improvements after fine-tuning.

### Medical Domain

| Term | Base WER | Fine-tuned WER | Delta |
|---|---|---|---|
| esophagogastroduodenoscopy | 91.3% | 38.4% | -52.9pp |
| cholecystectomy | 87.2% | 41.6% | -45.6pp |
| bronchoalveolar lavage | 79.4% | 35.2% | -44.2pp |
| electroencephalogram | 74.1% | 28.7% | -45.4pp |
| echocardiogram | 71.8% | 24.3% | -47.5pp |
| hemoglobin A1c | 63.4% | 21.2% | -42.2pp |
| prothrombin time | 58.9% | 19.4% | -39.5pp |
| paresthesia | 55.2% | 22.8% | -32.4pp |
| lymphadenopathy | 52.1% | 20.1% | -32.0pp |
| myocardial infarction | 41.3% | 15.6% | -25.7pp |

"Esophagogastroduodenoscopy" is the canonical hard case. It's 22 syllables, Latinate-derived,
and almost certainly appears fewer than 100 times in Whisper's 680k-hour training set (if at
all). Base Whisper typically transcribes it as multiple separate common words. After fine-tuning,
WER drops to 38% — still not great, but the model now at least attempts the correct token
sequence rather than producing a completely wrong word.

"Echocardiogram" is interesting because the components ("echo", "cardio", "gram") are all
individually common words. Base Whisper consistently splits it into three words with spaces.
Fine-tuning teaches the model that in medical context, these components fuse into a single term.

### Financial Domain

| Term | Base WER | Fine-tuned WER | Delta |
|---|---|---|---|
| collateralized debt obligation | 88.4% | 31.2% | -57.2pp |
| yield curve inversion | 74.2% | 28.9% | -45.3pp |
| quantitative easing | 68.1% | 19.4% | -48.7pp |
| securitization | 65.7% | 24.1% | -41.6pp |
| net interest margin | 59.3% | 21.8% | -37.5pp |
| EBITDA | 57.2% | 18.3% | -38.9pp |
| tier-one capital ratio | 54.8% | 22.4% | -32.4pp |
| amortization of intangibles | 48.6% | 20.3% | -28.3pp |
| non-GAAP earnings | 45.3% | 17.6% | -27.7pp |
| deferred revenue | 38.9% | 14.2% | -24.7pp |

"EBITDA" is a great example of the acronym problem. Base Whisper transcribes it as "EB it da"
or "E bit da" — it interprets the acronym as phonetic syllables rather than a financial term.
After fine-tuning on earnings call text where EBITDA always appears in its standard form, the
model correctly transcribes it.

"Collateralized debt obligation" has high base WER partly because it's long and partly because
"collateralized" is itself a rare word outside financial contexts.

---

## Ablation Results

### Data Scaling (Medical Domain)

Training on progressively more medical dictation data, evaluating domain term WER:

| Training samples | Domain term WER | Improvement from baseline |
|---|---|---|
| 843 (10%) | 38.2% | -10.5pp |
| 2,108 (25%) | 31.4% | -17.3pp |
| 4,216 (50%) | 26.8% | -21.9pp |
| 6,324 (75%) | 23.7% | -25.0pp |
| 8,432 (100%) | 22.1% | -26.6pp |

The curve is clearly log-linear in training samples — each doubling of data gives roughly
constant WER reduction in percentage points. This is encouraging for scaling but also suggests
that 8k samples is a practical sweet spot: doubling the dataset would probably get you to ~20%
domain WER at considerably more data collection cost.

There's a meaningful jump between 10% and 25% (6.8pp reduction) and then diminishing returns
above 50%. This pattern is consistent with most LoRA fine-tuning results I've seen — there's a
"minimum viable dataset" size below which the adapter doesn't have enough signal to overcome the
base model prior, and above which you're in a smoother optimization regime.

The practical implication: if you're adapting Whisper to a new domain, ~2,000 quality samples
will get you significant improvement, ~4,000-8,000 is where you saturate the easy gains, and
anything more requires careful evaluation to determine if marginal data is worth collecting.

### LoRA Rank Ablation (Medical Domain)

| Rank | Trainable params | Domain term WER |
|---|---|---|
| r=4 | 0.9M | 31.8% |
| r=8 | 1.8M | 28.4% |
| r=16 | 3.5M | 24.1% |
| r=32 | 7.1M | 22.1% |
| r=64 | 14.2M | 22.0% |

Diminishing returns above r=32. The additional parameters in r=64 don't improve WER but do
increase adapter size and training time. r=32 is the sweet spot.

---

## Key Design Decisions

### Why Whisper-small?

Three options: tiny (39M), small (244M), medium (769M). Tiny base WER is too high (domain term
WER >65%) — there isn't enough representational capacity to benefit from fine-tuning on hard
terms. Medium would probably give better results but requires ~4× the VRAM and training time.
Whisper-small gives a good balance of base capability and adaptability.

If you have a proper GPU cluster, Whisper-medium with the same LoRA setup would likely push
domain term WER below 15%.

### Why LoRA over Prompt Tuning / Prefix Tuning?

Tried prefix tuning first (Whisper has a natural place to insert a domain prefix). Domain WER
improved from 48.7% → 35.2% — better than nothing, but not as much as LoRA. The problem with
prefix tuning for ASR is that the acoustic information has to flow through the prefix tokens
before reaching the transcript generation, and short prefixes don't have enough capacity to
shift the language model prior sufficiently.

LoRA directly modifies the weight matrices that compute attention and store the language model
prior, which is a more targeted intervention.

### Why TTS for Financial Data?

The real alternative to TTS synthesis for financial data is either (a) recording earnings calls
ourselves, which is impractical, or (b) using existing earnings call datasets with proper
transcripts, which basically don't exist in an easily usable form.

TTS is an imperfect solution — see limitations — but it's practical. The main mitigation is
voice diversity: using all 14 voices from the catalog means the model can't simply memorize
TTS artifacts as a cue. The quality filtering also ensures that synthesized clips that happen
to have unusual prosody or artifacts don't make it into training.

### Why Edge-TTS?

Same reason the rlhf-and-reward-modelling-alt project uses it: it's free, produces high-quality
audio for English text, and has a rich voice catalog. The voices are Microsoft Azure Neural
voices, which are genuinely good — not state-of-the-art but clearly better than traditional TTS.

The alternative would be something like ElevenLabs or a local model (XTTS, Kokoro), but those
either have licensing restrictions for training data generation or require substantial compute.
Edge-TTS synthesizes 1,000 utterances in about 30 minutes with minimal compute.

---

## Limitations and Honest Assessment

**The financial eval set is synthesized.** The 14.2% domain WER on financial terms looks
impressive, but the eval set was synthesized with the same TTS pipeline as the training set.
The model has essentially learned to transcribe TTS audio of financial text, which is easier
than real earnings call audio. Domain WER on real speech would likely be higher — 30-40% is
a reasonable estimate based on the acoustic distribution gap.

The medical results are more trustworthy because the eval set is real clinical audio. Domain
term WER of 22.1% on real speech is meaningful.

**8,432 medical samples is a small dataset.** The log-linear data scaling curve suggests
we're not saturating — more data would help. The HuggingFace dataset we're using has ~10k
total clips; a larger clinical dictation corpus would likely push domain WER below 15%.

**Vocabulary coverage is incomplete.** The medical vocabulary file has 60+ terms across 10
specialties, which sounds like a lot but clinical medicine has thousands of domain-specific
terms. The fine-tuned model will still fail on specialty terms it hasn't seen — neonatology,
dermatology, and ophthalmology are essentially uncovered.

**The model is not validated for clinical use.** This is a research project. The catastrophic
forgetting numbers look fine for general purpose use, but a 22% domain-term WER and a 5.1%
LibriSpeech WER are not acceptable for production clinical documentation without a human review
step. Please don't deploy this in a clinical setting and claim it's validated.

**Prosody uniformity in financial data.** All 14 TTS voices produce fairly flat, newsreader-
style prosody. Real earnings calls have stressed pauses, excited delivery on good news, carefully
hedged language, and frequent repairs. The model trained on TTS might be less robust to these
prosodic variations.

---

## Future Work

A few directions that seem worth pursuing:

**More data, different sources.** The obvious path is more clinical dictation data. There are
proprietary clinical ASR datasets (Nuance/Microsoft's Dragon Medical data, for example) that
dwarf the HuggingFace dataset. Access is the bottleneck.

**Whisper-medium or large.** The architecture is exactly the same — just change `model_id`
to `openai/whisper-medium`. The main constraint is compute.

**Real financial audio.** The SEC has earnings call recordings for many public companies; the
aligned transcripts are available via services like Refinitiv or FactSet. Getting proper
audio-transcript alignment for even a few hundred earnings calls would be a significant
improvement over TTS synthesis.

**Constrained decoding for domain terms.** One approach that doesn't require any additional
training: modify the decoding process to boost the log-probability of known domain terms using
a prefix trie. This is compatible with the current model and could be layered on top of
fine-tuning. The tradeoff is false positives — if you boost "myocardial" in a non-medical
context, you'll get spurious transcriptions.

**Continual learning.** Right now we train once and ship. A real deployment would want to
update the model continuously as new domain data is collected (e.g., from corrected
transcriptions). Continual LoRA fine-tuning without catastrophic forgetting would be the
method of choice.

**Evaluation on real financial speech.** Even without a proper training set, we could evaluate
the current financial-domain model on a small hand-curated test set of real earnings call clips
to get a realistic sense of the acoustic domain gap.

---

## Repository Structure

```
whisper-domain-adaptation/
├── configs/
│   ├── medical_finetune.yaml        # Training hyperparameters for medical
│   ├── financial_finetune.yaml      # Training hyperparameters for financial
│   ├── medical_terms.txt            # 60+ medical domain vocabulary
│   └── financial_terms.txt          # 50+ financial domain vocabulary
├── data/
│   ├── medical/                     # Created by prepare_medical_data.py
│   └── financial_synth/             # Created by prepare_financial_data.py
├── experiments/
│   └── results/                     # JSON WER reports
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Dataset stats and distributions
│   ├── 02_baseline_evaluation.ipynb # Base Whisper error analysis
│   ├── 03_finetuning_walkthrough.ipynb # Training loop + loss curves
│   └── 04_results_analysis.ipynb   # Before/after comparison, ablations
├── scripts/
│   ├── prepare_medical_data.py      # Download + quality filter medical data
│   ├── prepare_financial_data.py    # TTS synthesis + quality filter
│   ├── evaluate_baseline.py         # WER evaluation for base Whisper
│   ├── evaluate_finetuned.py        # WER evaluation for fine-tuned model
│   ├── run_finetune.py              # Main training entry point
│   └── run_ablations.py             # Data scaling + synthetic mix ablations
├── src/
│   └── whisper_adapt/
│       ├── data/
│       │   ├── medical.py           # Medical dataset class + quality filter
│       │   ├── financial.py         # Financial synthesis + quality filter
│       │   └── feature_extraction.py # Log-mel spectrogram + tokenization
│       ├── models/
│       │   └── whisper_lora.py      # Whisper + LoRA adapter setup
│       ├── training/
│       │   └── finetune.py          # Seq2SeqTrainer setup + training loop
│       └── evaluation/
│           ├── wer.py               # Domain WER analyzer (overall/domain/common)
│           └── oov_analysis.py      # Per-term OOV recall and error analysis
├── tests/
├── requirements.txt
└── setup.py
```

---

## Environment and Dependencies

```bash
pip install -e .
```

Key dependencies:

- `transformers>=4.40.0` — Whisper model and Seq2SeqTrainer
- `peft>=0.10.0` — LoRA via PEFT
- `datasets>=2.18.0` — HuggingFace dataset loading
- `librosa>=0.10.1` — Audio loading and resampling
- `jiwer>=3.0.4` — WER computation with proper normalization
- `edge-tts>=6.1.9` — Financial speech synthesis
- `evaluate>=0.4.1` — HuggingFace evaluate integration

Training was done on a single A100 40GB. Medical fine-tuning (3 epochs, 8k samples) takes
approximately 2 hours. Financial fine-tuning (5 epochs, 1.3k samples) takes about 45 minutes.

The code will run on CPU or MPS but training will be very slow. Evaluation on a held-out set
is feasible on MPS for iterative development.

---

## Citation and Related Work

If you use this work, please cite the relevant datasets and model:

- Base model: OpenAI Whisper (Radford et al., 2022)
- Medical dataset: `speech-recognition/medical-speech-transcription` on HuggingFace Hub
- PEFT / LoRA: Hu et al., 2022, "LoRA: Low-Rank Adaptation of Large Language Models"

---

## Contact

Kartik Munjal — kartikmunjal19@gmail.com
