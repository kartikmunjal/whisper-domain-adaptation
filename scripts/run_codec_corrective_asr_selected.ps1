$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-CheckedPython {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE"
  }
}

$results = "experiments/results/codec_medical_corrective"
$summaryPath = "$results/signal_summary.json"
if (-not (Test-Path $summaryPath)) {
  throw "Run the corrective reconstruction grid and signal summarizer first"
}

# Original-audio ASR predictions are byte-for-byte reused from the immutable
# before study; only reconstructed conditions change.
if (-not (Test-Path "$results/asr/original/baseline.json")) {
  New-Item -ItemType Directory -Force "$results/asr/original" | Out-Null
  Copy-Item "experiments/results/codec_medical/asr/original/*.json" `
    "$results/asr/original/"
}

$summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
foreach ($cell in $summary.cells) {
  $name = "$($cell.quantizer)_$($cell.nominal_bitrate_bps)bps"
  $representativeSeed = $cell.wer_representative_seed
  $manifest = "$results/$name/seed_$representativeSeed/reconstructed_manifest.parquet"
  $baseResult = "$results/asr/$name/baseline.json"
  if (-not (Test-Path $baseResult)) {
    Invoke-CheckedPython scripts/evaluate_longform.py `
      --eval-manifest $manifest `
      --domain-vocab configs/medical_terms.txt `
      --seed 20260729 `
      --output $baseResult
  }
  foreach ($seed in @(11, 22, 33, 44, 55)) {
    $result = "$results/asr/$name/seed_$seed.json"
    if (-not (Test-Path $result)) {
      Invoke-CheckedPython scripts/evaluate_longform.py `
        --adapter-path "checkpoints/medical_research/seed_$seed/adapter" `
        --base-model openai/whisper-small `
        --eval-manifest $manifest `
        --domain-vocab configs/medical_terms.txt `
        --seed $seed `
        --output $result
    }
  }
}

Invoke-CheckedPython scripts/summarize_codec_wer.py `
  --results-dir $results `
  --output "$results/wer_summary.json" `
  --plot-output "$results/rate_distortion_wer.png" `
  --markdown-output "$results/wer_summary.md"
