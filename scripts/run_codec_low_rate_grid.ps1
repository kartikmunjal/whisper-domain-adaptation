$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
function Invoke-CheckedPython { & .\.venv\Scripts\python.exe @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $LASTEXITCODE" } }
$rates = @(@{Label="200bps";Vq=16;Fsq=@(4,4)}, @{Label="250bps";Vq=32;Fsq=@(4,4,2)})
foreach ($rate in $rates) { foreach ($seed in @(11,22,33,44,55)) {
  $vq="checkpoints/codec_low_rate_grid/vq_$($rate.Label)/seed_$seed"
  if (-not (Test-Path "$vq/run.json")) { Invoke-CheckedPython scripts/train_audio_codec.py --train_manifest data/financial_research/train_manifest.parquet --output_dir $vq --quantizer vq --codebook_size $rate.Vq --vq_ema_decay 0.99 --vq_ema_epsilon 0.00001 --vq_dead_code_batches 100 --epochs 30 --seed $seed }
  $fsq="checkpoints/codec_low_rate_grid/fsq_$($rate.Label)/seed_$seed"
  if (-not (Test-Path "$fsq/run.json")) { $a=@("scripts/train_audio_codec.py","--train_manifest","data/financial_research/train_manifest.parquet","--output_dir",$fsq,"--quantizer","fsq","--fsq_input_scale","1.0","--epochs","30","--seed","$seed","--fsq_levels")+$rate.Fsq; Invoke-CheckedPython @a }
} }
