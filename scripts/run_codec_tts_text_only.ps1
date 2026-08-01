$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
function RunPy { & .\.venv\Scripts\python.exe @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $LASTEXITCODE" } }
foreach ($seed in @(11,22,33,44,55)) {
  $out="checkpoints/codec_tts_text_only/seed_$seed"
  if (-not (Test-Path "$out/run.json")) {
    RunPy scripts/train_codec_tts.py --output-dir $out --seed $seed --decoder-input-mode text_only --duration-loss-weight 0.1 --scheduled-sampling-max 0.25 --scheduled-sampling-warmup-fraction 0.5
  }
}
