$ErrorActionPreference = "Stop"
$python = ".\env\Scripts\python.exe"

Write-Host "Installing dependencies..."
& $python -m pip install -r requirements.txt

$model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
$hard_data = "data/raw/prompts_hard.jsonl"
$out_dir = "data/processed_hard"

mkdir $out_dir -Force

# 1. Run Original Pipeline on Hard Dataset
Write-Host "Running Original Pipeline on Hard Dataset..."
& $python -m src.run --model $model --input $hard_data --outdir $out_dir

# 2. Run Benchmarks on Stealthy Dataset (Main Result)
Write-Host "Running Benchmarks on Stealthy Dataset..."
$stealthy_data = "data/raw/prompts_stealthy_large.jsonl"
$stealthy_out = "data/processed_stealthy"
& $python -m src.run --model $model --input $stealthy_data --outdir $stealthy_out

# 3. Generate Evaluation Plots
Write-Host "Generating Evaluation Plots..."
& $python scripts/generate_plots.py

Write-Host "Benchmark Complete. Results available in data/processed_hard and data/processed_stealthy"
