$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-CheckedPython {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}

Invoke-CheckedPython scripts/codec_tts_training_smoke.py

if (-not (Test-Path data/codec_tts_tokens/dataset_report.json)) {
  Invoke-CheckedPython scripts/prepare_codec_tts_tokens.py
}

foreach ($seed in @(11, 22, 33, 44, 55)) {
  $output = "checkpoints/codec_tts/seed_$seed"
  if (-not (Test-Path "$output/model.pt")) {
    Invoke-CheckedPython scripts/train_codec_tts.py --output-dir $output --seed $seed
  }
}
