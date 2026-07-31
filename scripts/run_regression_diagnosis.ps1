$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
& .\.venv\Scripts\python.exe scripts/analyze_synthetic_real_regression.py
if ($LASTEXITCODE -ne 0) { throw "Regression diagnosis failed: $LASTEXITCODE" }
& .\.venv\Scripts\python.exe scripts/summarize_regression_diagnosis.py
if ($LASTEXITCODE -ne 0) { throw "Regression summary failed: $LASTEXITCODE" }
