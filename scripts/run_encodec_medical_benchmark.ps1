$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Py {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) { throw "Python failed: $LASTEXITCODE" }
}

$results = "experiments/results/codec_medical_encodec"
if (-not (Test-Path "$results/signal_report.json")) {
  Py scripts/reconstruct_encodec_eval.py `
    --eval-manifest data/med_dictate_eval/eval_en_manifest.parquet `
    --output-dir $results --bandwidth-kbps 1.5 --device cuda
}

if (-not (Test-Path "$results/asr/original/baseline.json")) {
  New-Item -ItemType Directory -Force "$results/asr/original" | Out-Null
  Copy-Item "experiments/results/codec_medical_corrective/asr/original/*.json" `
    "$results/asr/original/"
}

if (-not (Test-Path "$results/asr/encodec_1.5kbps/baseline.json")) {
  Py scripts/evaluate_longform.py `
    --eval-manifest "$results/reconstructed_manifest.parquet" `
    --domain-vocab configs/medical_terms.txt --seed 20260801 `
    --output "$results/asr/encodec_1.5kbps/baseline.json"
}
foreach ($seed in @(11, 22, 33, 44, 55)) {
  $output = "$results/asr/encodec_1.5kbps/seed_$seed.json"
  if (-not (Test-Path $output)) {
    Py scripts/evaluate_longform.py `
      --adapter-path "checkpoints/medical_research/seed_$seed/adapter" `
      --base-model openai/whisper-small `
      --eval-manifest "$results/reconstructed_manifest.parquet" `
      --domain-vocab configs/medical_terms.txt --seed $seed --output $output
  }
}

Py scripts/summarize_encodec_benchmark.py
