$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-CheckedPython {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE"
  }
}

foreach ($seed in @(11, 22, 33, 44, 55)) {
  $generation = "experiments/results/codec_tts/seed_$seed"
  if (-not (Test-Path "$generation/generation_report.json")) {
    Invoke-CheckedPython scripts/synthesize_codec_tts.py `
      --tts-checkpoint "checkpoints/codec_tts/seed_$seed/model.pt" `
      --seed $seed `
      --output-dir $generation
  }
  $result = "$generation/round_trip_wer.json"
  if (-not (Test-Path $result)) {
    Invoke-CheckedPython scripts/evaluate_longform.py `
      --adapter-path "checkpoints/financial_research/seed_$seed/adapter" `
      --base-model openai/whisper-small `
      --eval-manifest "$generation/generated_manifest.parquet" `
      --domain-vocab configs/financial_terms.txt `
      --seed $seed `
      --output $result
  }
}

Invoke-CheckedPython scripts/summarize_codec_tts.py
