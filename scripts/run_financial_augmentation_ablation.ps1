$ErrorActionPreference="Stop"
$repoRoot=Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
function RunPy { & .\.venv\Scripts\python.exe @args; if($LASTEXITCODE -ne 0){throw "Python failed: $LASTEXITCODE"} }
$diagnosis=Get-Content "experiments/results/earnings21/regression_diagnosis.json" -Raw|ConvertFrom-Json
$snr=[math]::Abs($diagnosis.acoustic_comparison.metrics.heuristic_snr_db.cliffs_delta_real_vs_synthetic)
$silence=[math]::Abs($diagnosis.acoustic_comparison.metrics.silence_fraction.cliffs_delta_real_vs_synthetic)
if($snr -lt 0.474 -and $silence -lt 0.474){ throw "Locked augmentation gate did not pass (SNR=$snr, silence=$silence)" }
foreach($seed in @(11,22,33,44,55)){
  $checkpoint="checkpoints/financial_augmented/seed_$seed"
  if(-not(Test-Path "$checkpoint/run_provenance.json")){ RunPy scripts/run_finetune.py --config configs/financial_finetune.yaml --train_manifest data/financial_research/train_manifest.parquet --eval_manifest data/financial_research/validation_manifest.parquet --output_dir $checkpoint --seed $seed --acoustic_augmentation }
  $result="experiments/results/earnings21_augmented/seed_$seed/finetuned.json"
  if(-not(Test-Path $result)){ RunPy scripts/evaluate_longform.py --adapter-path "$checkpoint/adapter" --base-model openai/whisper-small --eval-manifest data/earnings21_eval/eval_manifest.parquet --domain-vocab configs/financial_terms.txt --seed $seed --output $result }
}
RunPy scripts/summarize_financial_augmentation.py
