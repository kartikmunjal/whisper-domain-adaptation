$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
function Py { & .\.venv\Scripts\python.exe @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $LASTEXITCODE" } }
$seeds = @(11,22,33,44,55)
foreach ($seed in $seeds) {
  $checkpointDir = "checkpoints/continuous_codec/seed_$seed"
  if (-not (Test-Path "$checkpointDir/run.json")) {
    Py scripts/train_continuous_codec.py --output-dir $checkpointDir --seed $seed --epochs 30 --device cuda
  }
  $resultDir = "experiments/results/continuous_codec/seed_$seed"
  if (-not (Test-Path "$resultDir/report.json")) {
    Py scripts/evaluate_continuous_codec.py --checkpoint "$checkpointDir/continuous_codec.pt" --eval-manifest data/med_dictate_eval/eval_en_manifest.parquet --output-dir $resultDir --device cuda
  }
  foreach ($condition in @("posterior_mean","quantized_1bit","quantized_2bit","quantized_4bit","quantized_6bit","quantized_8bit")) {
    $asr = "$resultDir/asr/$condition.json"
    if (-not (Test-Path $asr)) {
      Py scripts/evaluate_longform.py --adapter-path "checkpoints/medical_research/seed_$seed/adapter" --base-model openai/whisper-small --eval-manifest "$resultDir/$condition/reconstructed_manifest.parquet" --domain-vocab configs/medical_terms.txt --seed $seed --output $asr
    }
  }
}
Py scripts/summarize_continuous_codec.py
