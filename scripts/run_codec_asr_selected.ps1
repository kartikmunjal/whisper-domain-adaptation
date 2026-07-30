$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-CheckedPython {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE"
  }
}

$summaryPath = "experiments/results/codec_medical/signal_summary.json"
if (-not (Test-Path $summaryPath)) {
  throw "Run reconstruction grid and summarize_codec_signal_grid.py first"
}
$summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
$conditions = @(
  [PSCustomObject]@{
    name = "original"
    manifest = "data/med_dictate_eval/eval_en_manifest.parquet"
  }
)
foreach ($cell in $summary.cells) {
  $name = "$($cell.quantizer)_$($cell.nominal_bitrate_bps)bps"
  $seed = $cell.wer_representative_seed
  $conditions += [PSCustomObject]@{
    name = $name
    manifest = "experiments/results/codec_medical/$name/seed_$seed/reconstructed_manifest.parquet"
  }
}

foreach ($condition in $conditions) {
  $baseResult = "experiments/results/codec_medical/asr/$($condition.name)/baseline.json"
  if (-not (Test-Path $baseResult)) {
    Invoke-CheckedPython scripts/evaluate_longform.py `
      --eval-manifest $condition.manifest `
      --domain-vocab configs/medical_terms.txt `
      --seed 20260729 `
      --output $baseResult
  }
  foreach ($seed in @(11, 22, 33, 44, 55)) {
    $result = "experiments/results/codec_medical/asr/$($condition.name)/seed_$seed.json"
    if (-not (Test-Path $result)) {
      Invoke-CheckedPython scripts/evaluate_longform.py `
        --adapter-path "checkpoints/medical_research/seed_$seed/adapter" `
        --base-model openai/whisper-small `
        --eval-manifest $condition.manifest `
        --domain-vocab configs/medical_terms.txt `
        --seed $seed `
        --output $result
    }
  }
}
