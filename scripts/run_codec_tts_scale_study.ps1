$ErrorActionPreference="Stop"
$repoRoot=Split-Path -Parent $PSScriptRoot; Set-Location $repoRoot
function Py { & .\.venv\Scripts\python.exe @args; if($LASTEXITCODE -ne 0){throw "Python failed: $LASTEXITCODE"} }
if(-not(Test-Path data/financial_tts_scaled/dataset_report.json)){Py scripts/prepare_codec_tts_scaled_data.py}
foreach($rep in @("bytes","phonemes")){if(-not(Test-Path "data/codec_tts_tokens_scaled_$rep/dataset_report.json")){Py scripts/prepare_codec_tts_tokens.py --data-dir data/financial_tts_scaled --output-dir "data/codec_tts_tokens_scaled_$rep" --text-representation $rep --device cuda}}
foreach($rep in @("bytes","phonemes")){foreach($seed in @(11,22,33,44,55)){
 $ck="checkpoints/codec_tts_scaled_$rep/seed_$seed"; if(-not(Test-Path "$ck/run.json")){Py scripts/train_codec_tts.py --data-dir "data/codec_tts_tokens_scaled_$rep" --output-dir $ck --seed $seed --epochs 30 --batch-size 2 --gradient-accumulation-steps 4 --learning-rate 3e-4 --duration-loss-weight 0.1 --scheduled-sampling-max 0 --decoder-input-mode text_only --d-model 384 --nhead 8 --encoder-layers 6 --decoder-layers 6 --dim-feedforward 1536 --device cuda}
 $dir="experiments/results/codec_tts_scaled_$rep/seed_$seed"; if(-not(Test-Path "$dir/generation_report.json")){Py scripts/synthesize_codec_tts.py --tts-checkpoint "$ck/model.pt" --target-token-data "data/codec_tts_tokens_scaled_$rep" --seed $seed --output-dir $dir --use-duration-control --length-cap-multiplier 1.25 --repetition-penalty 0 --device cuda}
 if(-not(Test-Path "$dir/round_trip_wer.json")){Py scripts/evaluate_longform.py --adapter-path "checkpoints/financial_research/seed_$seed/adapter" --base-model openai/whisper-small --eval-manifest "$dir/generated_manifest.parquet" --domain-vocab configs/financial_terms.txt --seed $seed --output "$dir/round_trip_wer.json"}
 $cond="experiments/results/codec_tts_scaled_$rep/conditioning/seed_$seed.json"; if(-not(Test-Path $cond)){Py scripts/analyze_codec_tts_conditioning.py --checkpoint "$ck/model.pt" --tokens "data/codec_tts_tokens_scaled_$rep/test.parquet" --output $cond --seed $seed --device cuda}
 }
 Py scripts/summarize_codec_tts.py --results-dir "experiments/results/codec_tts_scaled_$rep" --training-dir "checkpoints/codec_tts_scaled_$rep" --output "experiments/results/codec_tts_scaled_$rep/summary.json" --markdown-output "experiments/results/codec_tts_scaled_$rep/summary.md"
 Py scripts/summarize_codec_tts_conditioning.py --input-dir "experiments/results/codec_tts_scaled_$rep/conditioning" --output-json "experiments/results/codec_tts_scaled_$rep/conditioning_summary.json" --output-md "experiments/results/codec_tts_scaled_$rep/CONDITIONING_REPORT.md" --attention-plot "experiments/results/codec_tts_scaled_$rep/attention.png"
}
if(-not(Test-Path experiments/results/piper_lessac_low/generation_report.json)){Py scripts/synthesize_piper_baseline.py}
foreach($seed in @(11,22,33,44,55)){if(-not(Test-Path "experiments/results/piper_lessac_low/seed_$seed.json")){Py scripts/evaluate_longform.py --adapter-path "checkpoints/financial_research/seed_$seed/adapter" --base-model openai/whisper-small --eval-manifest experiments/results/piper_lessac_low/generated_manifest.parquet --domain-vocab configs/financial_terms.txt --seed $seed --output "experiments/results/piper_lessac_low/seed_$seed.json"}}
Py scripts/summarize_tts_scale_study.py
