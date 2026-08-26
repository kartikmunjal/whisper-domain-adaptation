$ErrorActionPreference="Stop"; $root=Split-Path -Parent $PSScriptRoot; Set-Location $root
function Py { & .\.venv\Scripts\python.exe @args; if($LASTEXITCODE -ne 0){throw "Python failed: $LASTEXITCODE"} }
$out="experiments/results/latency_benchmark"; New-Item -ItemType Directory -Force $out | Out-Null
if(-not(Test-Path "$out/codec.json")){Py scripts/benchmark_latency.py --checkpoint checkpoints/codec_rate_grid_corrective/vq_400bps/seed_11/codec.pt --eval-manifest data/med_dictate_eval/eval_en_manifest.parquet --include-full-clips --output "$out/codec.json" --device cuda}
if(-not(Test-Path "$out/tts.json")){Py scripts/benchmark_tts_latency.py --tts-checkpoint checkpoints/codec_tts_scaled_phonemes/seed_11/model.pt --codec-checkpoint checkpoints/codec_rate_grid_corrective/vq_400bps/seed_11/codec.pt --eval-manifest data/financial_research/test_manifest.parquet --output "$out/tts.json" --device cuda}
Py scripts/summarize_latency_benchmarks.py
