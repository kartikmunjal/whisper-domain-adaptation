$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
foreach ($seed in @(11,22,33,44,55)) {
  $output="experiments/results/codec_tts_conditioning/seed_$seed.json"
  if (-not (Test-Path $output)) { & .\.venv\Scripts\python.exe scripts/analyze_codec_tts_conditioning.py --checkpoint "checkpoints/codec_tts_corrective/seed_$seed/model.pt" --output $output --seed $seed --device cuda; if ($LASTEXITCODE -ne 0) { throw "Diagnostic failed: $LASTEXITCODE" } }
}
& .\.venv\Scripts\python.exe scripts/summarize_codec_tts_conditioning.py
if ($LASTEXITCODE -ne 0) { throw "Diagnostic summary failed: $LASTEXITCODE" }
