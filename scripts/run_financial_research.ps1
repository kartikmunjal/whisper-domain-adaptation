$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$Python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing locked virtual environment: $Python"
}

function Invoke-Python {
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE`: $args"
  }
}

Invoke-Python scripts/gpu_preflight.py
Invoke-Python scripts/prepare_financial_research_data.py
Invoke-Python scripts/gpu_training_smoke.py
Invoke-Python scripts/evaluate_baseline.py `
  --eval_manifest data/financial_research/test_manifest.parquet `
  --domain_vocab configs/financial_terms.txt `
  --model_id openai/whisper-small `
  --device cuda `
  --seed 17 `
  --output experiments/results/financial_research/baseline_test.json

$seeds = @(11, 22, 33, 44, 55)
foreach ($seed in $seeds) {
  $runDir = "experiments/results/financial_research/seed_$seed"
  $checkpointDir = "checkpoints/financial_research/seed_$seed"
  $resultPath = "$runDir/finetuned_test.json"
  if (Test-Path $resultPath) {
    Write-Host "Seed $seed already has a final report; leaving it unchanged."
    continue
  }
  Invoke-Python scripts/run_finetune.py `
    --config configs/financial_finetune.yaml `
    --train_manifest data/financial_research/train_manifest.parquet `
    --eval_manifest data/financial_research/validation_manifest.parquet `
    --output_dir $checkpointDir `
    --seed $seed
  Invoke-Python scripts/evaluate_finetuned.py `
    --adapter_path "$checkpointDir/adapter" `
    --base_model openai/whisper-small `
    --eval_manifest data/financial_research/test_manifest.parquet `
    --domain_vocab configs/financial_terms.txt `
    --baseline_report experiments/results/financial_research/baseline_test.json `
    --device cuda `
    --seed $seed `
    --output $resultPath
}

Invoke-Python scripts/summarize_financial_trials.py
