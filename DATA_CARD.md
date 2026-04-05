# Data Card: whisper-domain-adaptation

**Version**: 1.0
**Author**: Kartik Munjal (kartikmunjal19@gmail.com)
**Last updated**: 2025-10

This data card documents the two datasets used to fine-tune Whisper for domain-specific speech
recognition: a real clinical dictation dataset (medical domain) and a synthesized earnings call
speech dataset (financial domain).

---

## Overview

| Property | Medical | Financial |
|---|---|---|
| Source type | Real recordings | TTS synthesis |
| Source dataset | speech-recognition/medical-speech-transcription | Edge-TTS + earnings call templates |
| Raw samples (before filtering) | ~10,500 | ~1,370 |
| Train samples (after filtering) | 8,432 | 1,340 |
| Eval samples (after filtering) | 937 | 237 |
| Total audio duration (train) | ~11.4 hours | ~1.8 hours |
| Audio format | 16 kHz mono WAV | 16 kHz mono WAV |
| Domain vocabulary | 60+ terms, 10 specialties | 50+ terms |
| License | See dataset card | Synthesized — no redistribution restrictions |

---

## Dataset 1: Medical Speech

### Provenance

**HuggingFace dataset ID**: `speech-recognition/medical-speech-transcription`

This dataset contains de-identified medical dictation recordings from clinical professionals —
physicians, nurses, and transcriptionists. Content spans clinical notes (SOAP format), discharge
summaries, operative reports, and radiology dictations. The recordings were made in clinical
and office settings; audio quality ranges from clear close-talk microphone to more distant room
audio with some background noise.

The dataset was originally collected and distributed via HuggingFace by its dataset authors.
The de-identification process applied to the transcripts is described in the original dataset
card. We do not have visibility into the specific de-identification methodology; users should
treat any downstream use with appropriate caution.

### Licensing

The dataset is distributed under the terms described in the HuggingFace dataset card for
`speech-recognition/medical-speech-transcription`. Users should review those terms before
use. This fine-tuning project uses the data for research purposes (model training) only.

**This dataset should not be used for clinical decision support without proper validation.**
The source recordings are de-identified, but that does not remove the need for appropriate
governance around clinical AI systems.

### Splits

| Split | Raw samples | After QF | % retained |
|---|---|---|---|
| train | ~10,500 | 8,432 | 80.3% |
| validation | ~1,170 | 937 | 80.1% |

The validation split is taken from the dataset's own validation split where available; otherwise
we perform a 10% random hold-out from the training split using a fixed seed (42).

### Audio Characteristics

After quality filtering, the train split has these characteristics:

- **Duration**: Mean 4.87s, median 4.12s, p95 12.4s, max 29.8s
- **SNR**: Mean 24.3 dB, median 23.8 dB, min 15.0 dB (filter threshold), p5 16.2 dB
- **Silence ratio**: Mean 0.19, median 0.17, max 0.39 (just below filter threshold)
- **Sample rate**: 16,000 Hz (resampled from original if necessary)
- **Channels**: Mono

Most clips are 3-7 seconds long — typical for dictated sentences or short paragraphs. The
radiology report clips tend to be longer (8-15 seconds) due to structured report format.

### Transcript Characteristics

- **Vocabulary size**: approximately 4,200 unique word types in train split
- **Mean transcript length**: 18.4 words (tokens), median 15 words
- **Domain term frequency**: 62% of training utterances contain at least one term from
  `configs/medical_terms.txt`; this high rate reflects that the dataset was specifically
  curated for clinical dictation content

### Quality Filtering Methodology

The filtering pipeline in `src/whisper_adapt/data/medical.py` implements the same thresholds
as the Audio-Data-Creation project's Common Voice curation pipeline:

**SNR estimation** (reference-free):
- Compute RMS energy per 25ms frame with 50% overlap
- SNR = 20 * log10(90th percentile RMS / 10th percentile RMS)
- This treats the quietest frames as the noise floor without requiring a separate noise reference
- Threshold: SNR >= 15 dB
- Rejection rate for this criterion: approximately 8% of clips

**Silence ratio**:
- Frame is "silent" if its RMS is more than 30 dB below the peak frame RMS
- Threshold: silence_ratio < 0.40
- Rejects recordings where the speaker paused before starting or left long trailing silence
- Rejection rate: approximately 6% of clips

**Duration bounds**:
- Minimum 0.5 seconds: excludes near-empty or accidentally triggered recordings
- Maximum 30.0 seconds: excludes very long dictations that would require chunking
- Rejection rate: approximately 4% of clips (mostly too-long clips)

**Clipping check**:
- Compute fraction of samples with |amplitude| >= 0.99 (normalized float32)
- Threshold: clipping_ratio < 0.001 (0.1%)
- Rejection rate: approximately 2% of clips

Total rejection rate: approximately 20%, consistent with the 80% pass rate in the Audio-Data-Creation
pipeline for similar-quality recordings. Rejection reasons are logged so the distribution can be
inspected — the most common reason is SNR (recording quality) rather than duration or silence.

### Domain Vocabulary Coverage

The 60 terms in `configs/medical_terms.txt` span 10 clinical specialties:

| Specialty | Terms | Example |
|---|---|---|
| Cardiology | 10 | atrial fibrillation, echocardiogram, myocardial infarction |
| Pulmonology | 9 | pneumothorax, bronchoalveolar lavage, bronchoscopy |
| Gastroenterology | 9 | cholecystectomy, esophagogastroduodenoscopy, colonoscopy |
| Oncology | 10 | metastasis, lymphadenopathy, chemotherapy |
| Neurology | 10 | encephalopathy, electroencephalogram, paresthesia |
| General / Laboratory | 10 | hemoglobin A1c, prothrombin time, troponin |
| Pharmacology | 10 | anticoagulation, vasopressor, thrombolysis |

Term frequency in the training set is uneven — cardiology and pulmonology terms are more
frequent (reflecting the dominance of hospital medicine in the dataset), while rare specialties
like neuropharmacology are underrepresented. Full term frequency counts are in the data
exploration notebook (01_data_exploration.ipynb).

The term "esophagogastroduodenoscopy" has the lowest frequency (~28 training examples) and
highest base Whisper WER (~91%). The term "chemotherapy" has the highest frequency (~340
training examples) and relatively lower base WER (~29%) — it appears frequently enough in web
text that Whisper's prior isn't completely uninformed.

---

## Dataset 2: Financial Speech (Synthesized)

### Provenance

This dataset is fully synthesized. There is no source audio. The transcripts are generated
from template sentences embedding financial terms, and the audio is synthesized using Edge-TTS
(Microsoft Azure Neural voices) via the `edge-tts` Python library.

The sentence templates are written in `src/whisper_adapt/data/financial.py`
(`CONTEXT_TEMPLATES` list). The templates were authored to mimic earnings call register — hedged
language, guidance-speak, metric-focus. Example:
- "Our {term} for the quarter came in at the high end of guidance."
- "Excluding restructuring charges, {term} expanded by 120 basis points."

Each term is embedded in 3-5 different sentence contexts (more contexts for harder Tier-2 terms).
Each (sentence, voice) pair is synthesized separately, giving prosodic variety even within the
same sentence.

### Voice Catalog

The 14 voices used match exactly the catalog in `rlhf-and-reward-modelling-alt`:

| Gender | Accent | Voices |
|---|---|---|
| Male | American | en-US-GuyNeural, en-US-ChristopherNeural |
| Female | American | en-US-JennyNeural, en-US-AriaNeural |
| Male | British | en-GB-RyanNeural, en-GB-ThomasNeural |
| Female | British | en-GB-SoniaNeural, en-GB-LibbyNeural |
| Male | Australian | en-AU-WilliamNeural |
| Female | Australian | en-AU-NatashaNeural |
| Male | Indian | en-IN-PrabhatNeural |
| Female | Indian | en-IN-NeerjaNeural |

Voice assignment is round-robin by synthesis index — the ith utterance uses voice
`ALL_VOICES[i % 14]`. This ensures that each voice is used roughly equally and that no single
voice dominates any particular term.

### Splits

| Split | Samples | % retained from synthesis |
|---|---|---|
| train | 1,340 | 97.8% |
| eval | 237 | 98.3% |

The very high pass rate (>97%) reflects that TTS audio is clean by construction. The few
rejected clips were synthesis failures (Edge-TTS API timeout) or, in rare cases, clips where
the synthesis produced unusual prosody that failed the silence ratio check.

### Audio Characteristics

- **Duration**: Mean 3.41s, median 3.18s (shorter than medical — sentences are shorter)
- **SNR**: Mean 38.2 dB, median 37.8 dB (TTS is very clean — well above the 20 dB threshold)
- **Silence ratio**: Mean 0.11, median 0.10 (low silence; TTS has little dead air)
- **Sample rate**: 16,000 Hz (resampled from MP3 output of Edge-TTS)
- **Channels**: Mono

The high SNR and low silence ratio are the key distributional differences from real speech.
Real earnings call audio would have SNR ~18-25 dB with more background room noise, occasional
coughs, and silence during question transitions.

### Quality Filtering for Synthesized Audio

The thresholds for financial data are stricter than for real audio:

- **SNR >= 20 dB** (vs. 15 dB for real audio): TTS should be much cleaner; anything below 20 dB
  likely indicates a synthesis artifact or encoding issue
- **Silence ratio < 0.50** (vs. 0.40): TTS sentences sometimes have long pauses at the
  end of the audio buffer; slightly more lenient threshold to accommodate this
- **Duration 1.0 – 20.0 seconds** (vs. 0.5 – 30.0): TTS sentences are shorter by design

These stricter thresholds are configured in `configs/financial_finetune.yaml`.

### Domain Vocabulary Coverage

The 50+ terms in `configs/financial_terms.txt` are divided into two tiers:

**Tier 1** (high-frequency in earnings calls, base Whisper WER ~28-45%):
- EBITDA, EBITDA margin, free cash flow, gross margin, revenue recognition
- non-GAAP earnings, diluted EPS, year-over-year growth, sequential revenue
- operating leverage, basis points, forward guidance, organic growth, run rate
- capital expenditures, accounts receivable, inventory turnover, deferred revenue
- amortization, depreciation, restructuring charges, impairment charge, working capital

**Tier 2** (more technical, base Whisper WER ~50-90%):
- quantitative easing, yield curve inversion, credit default swap
- collateralized debt obligation, securitization, repo rate, mark-to-market
- net interest margin, tier-one capital ratio, loan-to-value ratio
- debt-to-equity ratio, price-to-earnings ratio
- comparable store sales, same-store sales, adjusted operating income
- pro forma revenue, normalised earnings, recurring revenue
- churn rate, customer acquisition cost, lifetime value, net revenue retention

Tier-2 terms receive 5 sentence variants per term; Tier-1 terms receive 3 variants. This
compensates for the higher difficulty of Tier-2 terms during model training.

---

## Known Biases and Limitations

### Medical Dataset

**Speaker demographics are opaque.** The dataset documentation does not describe the
demographic distribution of speakers (age, gender, accent, specialty background). Any biases
in the underlying speaker pool will be present in the fine-tuned model's performance
characteristics. It is plausible that certain accents or speaking styles are underrepresented.

**De-identification quality is unknown.** The transcripts have been de-identified but we
cannot independently verify the quality of de-identification. Users should treat the data
with appropriate caution and should not attempt to re-identify individuals.

**Specialty imbalance.** Cardiology and general medicine are overrepresented relative to
sub-specialties. The fine-tuned model may perform significantly worse on radiology-specific
vocabulary or specialty pharmacology than the average WER suggests.

**Possible overlap with Whisper training data.** We cannot rule out that some of the web audio
in Whisper's 680k-hour training set overlaps with clips from this or related clinical datasets.
If so, the "base Whisper baseline" may slightly underestimate true OOV error rates.

### Financial Dataset

**Acoustic domain mismatch.** This is the biggest limitation. The eval set is synthesized
from the same distribution as the training set — same voice catalog, same sentence templates.
The reported 14.2% overall WER and 26.7% domain WER on financial terms represent performance
on TTS audio, not on real earnings call recordings. A realistic estimate for real financial
speech, based on acoustic domain gap analysis in the literature, is 30-40% domain WER.

**Prosodic uniformity.** All 14 Edge-TTS Neural voices produce relatively flat, newsreader-
style prosody. Real earnings calls have emotional variation, stressed pauses, repairs, fillers
("uh", "you know"), and spontaneous speech phenomena that TTS doesn't reproduce. The fine-
tuned model may struggle with these.

**Template diversity is limited.** Ten context sentence templates drive the entire financial
dataset. The model has essentially learned to transcribe financial terms in these specific
sentence patterns. Performance on novel sentence structures may be degraded.

**Term coverage is incomplete.** The 50 terms in `configs/financial_terms.txt` represent a
small fraction of the vocabulary in actual earnings calls. Terms not in the training set will
still use the base Whisper prior.

### Both Datasets

**No demographic diversity information.** Neither dataset provides speaker demographic metadata,
so we cannot characterize performance across age groups, accents, or genders with any
statistical rigor.

**English only.** Both datasets are English-only. The multi-lingual Whisper model could
in principle be fine-tuned for medical terminology in other languages, but this project does
not address that.

**Small scale.** 8,432 + 1,340 = 9,772 training samples total. This is small by modern
standards. The data scaling ablation suggests we're not saturating — more data would help.

---

## Intended Use

**In scope:**
- Domain ASR fine-tuning research
- Studying OOV vocabulary adaptation in neural ASR
- Benchmarking LoRA vs. full fine-tuning for audio models
- Educational use (understanding Whisper architecture and domain adaptation)

**Out of scope:**
- Production medical documentation without human review
- Clinical decision support
- Patient-facing applications
- Any use requiring medical device validation (FDA or equivalent)

The medical model in particular should be treated as a research prototype. A WER of 22.1%
on domain terms means roughly 1 in 5 domain-specific words is transcribed incorrectly.
For clinical documentation where accuracy is safety-critical, this requires mandatory review.

---

## Data Access and Reproduction

### Medical data

```bash
# Download and filter (requires HuggingFace token for gated datasets)
python scripts/prepare_medical_data.py --output_dir data/medical
```

Total download: ~20 GB. Filtered WAVs: ~15 GB. Parquet manifests: ~5 MB.

The HuggingFace dataset may require acceptance of terms of service. Run:
```bash
huggingface-cli login
```
before downloading.

### Financial data

```bash
# Synthesize (requires internet connection for Edge-TTS)
python scripts/prepare_financial_data.py --output_dir data/financial_synth
```

Total synthesized audio: ~250 MB. No external data download required.

Edge-TTS synthesis calls the Microsoft Azure API. There are rate limits that may cause
occasional synthesis failures; these are handled gracefully (failed clips are skipped and
logged). Running with `--dry_run` previews the synthesis plan without making API calls.

---

## Changelog

- **v1.0 (2025-10)**: Initial release. Medical dataset from HuggingFace; financial dataset
  synthesized via Edge-TTS with 14-voice catalog. Quality filtering thresholds aligned with
  Audio-Data-Creation project.
