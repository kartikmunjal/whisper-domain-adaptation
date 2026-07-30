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
$rates = @(
  @{ Label = "300bps"; Vq = 64;   Fsq = @(4, 4, 4) },
  @{ Label = "400bps"; Vq = 256;  Fsq = @(4, 4, 4, 4) },
  @{ Label = "500bps"; Vq = 1024; Fsq = @(4, 4, 4, 4, 4) }
)

Invoke-CheckedPython scripts/codec_training_smoke.py

foreach ($rate in $rates) {
  foreach ($seed in $seeds) {
    $vqOut = "checkpoints/codec_rate_grid/vq_$($rate.Label)/seed_$seed"
    if (-not (Test-Path "$vqOut/codec.pt")) {
      Invoke-CheckedPython scripts/train_audio_codec.py `
        --train_manifest data/financial_research/train_manifest.parquet `
        --output_dir $vqOut --quantizer vq `
        --codebook_size $rate.Vq --epochs 30 --seed $seed
    }

    $fsqOut = "checkpoints/codec_rate_grid/fsq_$($rate.Label)/seed_$seed"
    if (-not (Test-Path "$fsqOut/codec.pt")) {
      $fsqArgs = @(
        "scripts/train_audio_codec.py",
        "--train_manifest", "data/financial_research/train_manifest.parquet",
        "--output_dir", $fsqOut,
        "--quantizer", "fsq",
        "--epochs", "30",
        "--seed", "$seed",
        "--fsq_levels"
      ) + $rate.Fsq
      Invoke-CheckedPython @fsqArgs
    }
  }
}
