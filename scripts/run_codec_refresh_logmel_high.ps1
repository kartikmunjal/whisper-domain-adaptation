$ErrorActionPreference="Stop"
$repoRoot=Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
function RunPy { & .\.venv\Scripts\python.exe @args; if($LASTEXITCODE -ne 0){throw "Python failed: $LASTEXITCODE"} }
$results="experiments/results/codec_medical_corrective"
foreach($rate in @(300,400,500)){foreach($q in @("vq","fsq")){foreach($seed in @(11,22,33,44,55)){
  $name="${q}_${rate}bps"; RunPy scripts/reconstruct_codec_eval.py --checkpoint "checkpoints/codec_rate_grid_corrective/$name/seed_$seed/codec.pt" --eval-manifest data/med_dictate_eval/eval_en_manifest.parquet --output-dir "$results/$name/seed_$seed"
}}}
RunPy scripts/summarize_codec_signal_grid.py --results-dir $results --checkpoint-dir checkpoints/codec_rate_grid_corrective --output "$results/signal_summary.json" --rates 300 400 500
