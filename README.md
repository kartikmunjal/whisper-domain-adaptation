[![CI](https://github.com/kartikmunjal/whisper-domain-adaptation/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmunjal/whisper-domain-adaptation/actions/workflows/ci.yml)

# Whisper Domain Adaptation

Reproducible evaluation of Whisper-small LoRA adaptation for specialized
vocabulary. The current confirmatory study targets financial speech generated
with Edge-TTS. Medical experiments remain exploratory until their original
primary artifacts are recovered or rerun.

## Research status

The projected financial value has been withdrawn and replaced by a directly
measured five-seed result. The generated overall/domain/common table, trial
confidence intervals, and paired changes are in
[the confirmatory result](experiments/results/financial_research/summary.md).
The protocol was locked in [RESEARCH_PLAN.md](RESEARCH_PLAN.md) before the GPU
runs.

The previous medical, rank, data-scaling, synthetic-mixture, prefix-tuning, and
catastrophic-forgetting numbers were not accompanied by sufficient primary run
artifacts. They are not treated as verified results in this repository.

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

All evaluation audio is synthetic. **TTS-on-TTS evaluation is optimistic and
does not establish performance on real earnings-call audio.** A real-audio
result will only be added from a licensed, manually verified, speaker-disjoint
evaluation set.

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

## Experimental codec work

An uncommitted VQ-VAE/FSQ prototype may be present in some working copies. It is
not part of the ASR study and is intentionally excluded from the primary
research narrative pending its own protocol, bitrate accounting, perceptual
metrics, validation set, and baselines.

## License

Code is released under the MIT License. Dataset and model artifacts retain
their upstream licenses; consult their source cards before redistribution.
