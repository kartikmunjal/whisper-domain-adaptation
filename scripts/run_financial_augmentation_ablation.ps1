$ErrorActionPreference="Stop"
$repoRoot=Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
function RunPy { & .\.venv\Scripts\python.exe @args; if($LASTEXITCODE -ne 0){throw "Python failed: $LASTEXITCODE"} }
foreach($seed in @(11,22,33,44,55)){
  $checkpoint="checkpoints/financial_augmented/seed_$seed"
  if(-not(Test-Path "$checkpoint/run_provenance.json")){ RunPy scripts/run_finetune.py --config configs/financial_finetune.yaml --train_manifest data/financial_research/train_manifest.parquet --eval_manifest data/financial_research/validation_manifest.parquet --output_dir $checkpoint --seed $seed --acoustic_augmentation }
  $result="experiments/results/earnings21_augmented/seed_$seed/finetuned.json"
  if(-not(Test-Path $result)){ RunPy scripts/evaluate_longform.py --adapter-path "$checkpoint/adapter" --base-model openai/whisper-small --eval-manifest data/earnings21_eval/eval_manifest.parquet --domain-vocab configs/financial_terms.txt --seed $seed --output $result }
}
