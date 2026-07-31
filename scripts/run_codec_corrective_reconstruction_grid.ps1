$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-CheckedPython {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE"
  }
}

$manifest = "data/med_dictate_eval/eval_en_manifest.parquet"
foreach ($rate in @(300, 400, 500)) {
  foreach ($quantizer in @("vq", "fsq")) {
    foreach ($seed in @(11, 22, 33, 44, 55)) {
      $name = "${quantizer}_${rate}bps"
      $checkpoint = "checkpoints/codec_rate_grid_corrective/$name/seed_$seed/codec.pt"
      $output = "experiments/results/codec_medical_corrective/$name/seed_$seed"
      if (-not (Test-Path "$output/report.json")) {
        Invoke-CheckedPython scripts/reconstruct_codec_eval.py `
          --checkpoint $checkpoint `
          --eval-manifest $manifest `
          --output-dir $output
      }
    }
  }
}
