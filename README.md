[![CI](https://github.com/kartikmunjal/whisper-domain-adaptation/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmunjal/whisper-domain-adaptation/actions/workflows/ci.yml)

# Whisper Domain Adaptation

Reproducible evaluation of Whisper-small LoRA adaptation for specialized
vocabulary, plus preregistered neural-codec and text-to-codec-token studies.

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

It also includes a small autoregressive Transformer mapping UTF-8 text tokens
to VQ codec tokens. The held-out path is
text → predicted tokens → codec decoder → waveform, followed by round-trip
Whisper WER and comparison with the corresponding Edge-TTS clips. No TTS claim
is inferred from teacher-forced validation alone.

The completed [five-seed TTS report](experiments/results/codec_tts/summary.md)
finds mean codec-TTS round-trip WER of 1020.62% (95% trial-bootstrap CI
701.83–1353.10%) versus 1.18% for paired Edge-TTS. Mean generation-failure rate
is 27.76% (8.37–61.43%), and SI-SDR conditional on decodable output is
-42.99 dB (-44.07 to -42.02). WER can exceed 100% because uncontrolled
insertions outnumber reference words. The implementation closes the
text-to-token-to-waveform loop, but this small autoregressive model is
decisively noncompetitive under the locked experiment.

## License

Code is released under the MIT License. Dataset and model artifacts retain
their upstream licenses; consult their source cards before redistribution.
