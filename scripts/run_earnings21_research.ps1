$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-CheckedPython {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE"
  }
}

$manifest = "data/earnings21_eval/eval_manifest.parquet"
$baseline = "experiments/results/earnings21/baseline.json"
if (-not (Test-Path $baseline)) {
  Invoke-CheckedPython scripts/evaluate_longform.py `
    --eval-manifest $manifest `
    --domain-vocab configs/financial_terms.txt `
    --seed 20260729 `
    --output $baseline
}

foreach ($seed in @(11, 22, 33, 44, 55)) {
  $result = "experiments/results/earnings21/seed_$seed/finetuned.json"
  if (-not (Test-Path $result)) {
    Invoke-CheckedPython scripts/evaluate_longform.py `
      --adapter-path "checkpoints/financial_research/seed_$seed/adapter" `
      --base-model openai/whisper-small `
      --eval-manifest $manifest `
      --domain-vocab configs/financial_terms.txt `
      --seed $seed `
      --output $result
  }
}
