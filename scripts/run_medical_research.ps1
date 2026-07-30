$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-CheckedPython {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE"
  }
}

$seeds = @(11, 22, 33, 44, 55)
$baseline = "experiments/results/medical_research/baseline_en.json"
if (-not (Test-Path $baseline)) {
  Invoke-CheckedPython scripts/evaluate_longform.py `
    --eval-manifest data/med_dictate_eval/eval_en_manifest.parquet `
    --domain-vocab configs/medical_terms.txt `
    --seed 20260729 `
    --output $baseline
}

foreach ($seed in $seeds) {
  $checkpoint = "checkpoints/medical_research/seed_$seed"
  $result = "experiments/results/medical_research/seed_$seed/finetuned_en.json"
  if (-not (Test-Path "$checkpoint/adapter/adapter_config.json")) {
    Invoke-CheckedPython scripts/run_finetune.py `
      --config configs/medical_research_finetune.yaml `
      --train_manifest data/medical_research/train_manifest.parquet `
      --eval_manifest data/medical_research/validation_manifest.parquet `
      --output_dir $checkpoint `
      --seed $seed
  }
  if (-not (Test-Path $result)) {
    Invoke-CheckedPython scripts/evaluate_longform.py `
      --adapter-path "$checkpoint/adapter" `
      --base-model openai/whisper-small `
      --eval-manifest data/med_dictate_eval/eval_en_manifest.parquet `
      --domain-vocab configs/medical_terms.txt `
      --seed $seed `
      --output $result
  }
}
