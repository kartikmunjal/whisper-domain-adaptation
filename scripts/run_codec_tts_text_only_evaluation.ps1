$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
function RunPy { & .\.venv\Scripts\python.exe @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $LASTEXITCODE" } }
foreach ($seed in @(11,22,33,44,55)) {
  $dir="experiments/results/codec_tts_text_only/seed_$seed"
  if (-not (Test-Path "$dir/generation_report.json")) { RunPy scripts/synthesize_codec_tts.py --tts-checkpoint "checkpoints/codec_tts_text_only/seed_$seed/model.pt" --seed $seed --output-dir $dir --use-duration-control --length-cap-multiplier 1.25 --repetition-penalty 0.0 }
  if (-not (Test-Path "$dir/round_trip_wer.json")) { RunPy scripts/evaluate_longform.py --adapter-path "checkpoints/financial_research/seed_$seed/adapter" --base-model openai/whisper-small --eval-manifest "$dir/generated_manifest.parquet" --domain-vocab configs/financial_terms.txt --seed $seed --output "$dir/round_trip_wer.json" }
  RunPy scripts/analyze_codec_tts_conditioning.py --checkpoint "checkpoints/codec_tts_text_only/seed_$seed/model.pt" --output "experiments/results/codec_tts_text_only/conditioning/seed_$seed.json" --seed $seed --device cuda
}
RunPy scripts/summarize_codec_tts.py --results-dir experiments/results/codec_tts_text_only --training-dir checkpoints/codec_tts_text_only --output experiments/results/codec_tts_text_only/summary.json --markdown-output experiments/results/codec_tts_text_only/summary.md
RunPy scripts/summarize_codec_tts_conditioning.py --input-dir experiments/results/codec_tts_text_only/conditioning --output-json experiments/results/codec_tts_text_only/conditioning_summary.json --output-md experiments/results/codec_tts_text_only/CONDITIONING_REPORT.md --attention-plot experiments/results/codec_tts_text_only/attention.png
