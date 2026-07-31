$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
function Invoke-CheckedPython { & .\.venv\Scripts\python.exe @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $LASTEXITCODE" } }
$results="experiments/results/codec_medical_low_rate"
foreach ($rate in @(200,250)) { foreach ($q in @("vq","fsq")) { foreach ($seed in @(11,22,33,44,55)) {
  $name="${q}_${rate}bps"; $out="$results/$name/seed_$seed"
  if (-not (Test-Path "$out/report.json")) { Invoke-CheckedPython scripts/reconstruct_codec_eval.py --checkpoint "checkpoints/codec_low_rate_grid/$name/seed_$seed/codec.pt" --eval-manifest data/med_dictate_eval/eval_en_manifest.parquet --output-dir $out }
} } }
Invoke-CheckedPython scripts/summarize_codec_signal_grid.py --results-dir $results --checkpoint-dir checkpoints/codec_low_rate_grid --output "$results/signal_summary.json" --rates 200 250
$summary=Get-Content "$results/signal_summary.json" -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force "$results/asr/original" | Out-Null
Copy-Item "experiments/results/codec_medical/asr/original/*.json" "$results/asr/original/" -Force
foreach ($cell in $summary.cells) { $name="$($cell.quantizer)_$($cell.nominal_bitrate_bps)bps"; $manifest="$results/$name/seed_$($cell.wer_representative_seed)/reconstructed_manifest.parquet"
  if (-not (Test-Path "$results/asr/$name/baseline.json")) { Invoke-CheckedPython scripts/evaluate_longform.py --eval-manifest $manifest --domain-vocab configs/medical_terms.txt --seed 20260731 --output "$results/asr/$name/baseline.json" }
  foreach ($seed in @(11,22,33,44,55)) { if (-not (Test-Path "$results/asr/$name/seed_$seed.json")) { Invoke-CheckedPython scripts/evaluate_longform.py --adapter-path "checkpoints/medical_research/seed_$seed/adapter" --base-model openai/whisper-small --eval-manifest $manifest --domain-vocab configs/medical_terms.txt --seed $seed --output "$results/asr/$name/seed_$seed.json" } }
}
Invoke-CheckedPython scripts/summarize_codec_wer.py --results-dir $results --output "$results/wer_summary.json" --plot-output "$results/rate_distortion_wer.png" --markdown-output "$results/wer_summary.md"
