$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

python scripts/gpu_preflight.py
python scripts/prepare_financial_research_data.py
python scripts/evaluate_baseline.py `
  --eval_manifest data/financial_research/test_manifest.parquet `
  --domain_vocab configs/financial_terms.txt `
  --model_id openai/whisper-small `
  --device cuda `
  --seed 17 `
  --output experiments/results/financial_research/baseline_test.json

$seeds = @(11, 22, 33, 44, 55)
foreach ($seed in $seeds) {
  $runDir = "experiments/results/financial_research/seed_$seed"
  $checkpointDir = "checkpoints/financial_research/seed_$seed"
  python scripts/run_finetune.py `
    --config configs/financial_finetune.yaml `
    --train_manifest data/financial_research/train_manifest.parquet `
    --eval_manifest data/financial_research/validation_manifest.parquet `
    --output_dir $checkpointDir `
    --seed $seed
  python scripts/evaluate_finetuned.py `
    --adapter_path "$checkpointDir/adapter" `
    --base_model openai/whisper-small `
    --eval_manifest data/financial_research/test_manifest.parquet `
    --domain_vocab configs/financial_terms.txt `
    --baseline_report experiments/results/financial_research/baseline_test.json `
    --device cuda `
    --seed $seed `
    --output "$runDir/finetuned_test.json"
}

python scripts/summarize_financial_trials.py
