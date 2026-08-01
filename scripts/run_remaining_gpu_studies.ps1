$ErrorActionPreference="Stop"
$repoRoot=Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
foreach($script in @("run_codec_tts_text_only_evaluation.ps1","run_financial_augmentation_ablation.ps1","run_codec_low_rate_grid.ps1","run_codec_low_rate_evaluation.ps1","run_codec_refresh_logmel_high.ps1")){
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot $script)
  if($LASTEXITCODE -ne 0){throw "$script failed: $LASTEXITCODE"}
}
& .\.venv\Scripts\python.exe scripts/summarize_codec_extended_grid.py
if($LASTEXITCODE -ne 0){throw "Extended codec summary failed: $LASTEXITCODE"}
