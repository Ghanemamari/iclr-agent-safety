# ICLR Agent Safety: Prompt Injection Detection Mapping

This repository contains a reproducible research pipeline for detecting **Prompt Injection** signals in Large Language Models (LLMs) using latent activation probing.

## 🚀 Overview

Our methodology demonstrates that prompt injections (such as "ignore previous instructions") leave unique "semantic signatures" in the hidden states (activations) of transformer models. By training lightweight linear probes on these activations, we can detect attacks with near-perfect accuracy, even when the attacks are designed to be "stealthy" and lack obvious structural cues (like `Task:` headers).

### Key Features
- **Activation Extraction**: Support for extracting layer-wise hidden states from Hugging Face models (TinyLlama, Qwen, etc.).
- **Latent Probing**: High-performance linear probes with strict **Group Split** validation to prevent paired-sample leakage.
- **Stealthy Benchmarking**: Evaluation on datasets designed to isolate semantic signal from formatting artifacts.
- **Statistical Rigor**: Built-in tools for computing p-values, Cohen's d effect sizes, and bootstrap confidence intervals.

## 📊 Results Summary

| Model | Dataset | Probe AUC | Baseline (Embeddings) | Statistical (PPL) |
| :--- | :--- | :---: | :---: | :---: |
| TinyLlama-1.1B | **Stealthy** | **0.99** | 0.25 | 0.28 |
| Qwen2.5-0.5B | **Stealthy** | **1.00** | 0.25 | 0.28 |
| TinyLlama-1.1B | **Complex** | **1.00** | 1.00* | - |

> [!IMPORTANT]
> The **Stealthy Dataset** is our most rigorous test. It removes all formatting "watermarks" (like `Task:`), forcing the probe to identify the actual intent of the injection. Our method maintains perfect performance while traditional embeddings fail.

## 🛠️ Installation

```bash
# Create environment
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### 1. Generate Dataset
```bash
python -m src.generate.generate_stealthy --output data/raw/prompts_stealthy.jsonl --n 400
```

### 2. Run Detection Pipeline
The main entry point performs activation extraction followed by probe training and evaluation.
```bash
python -m src.run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --input data/raw/prompts_stealthy.jsonl --outdir data/processed
```

### 3. Generate Analysis Plots
```bash
python scripts/generate_plots.py
```
*Plots will be saved to `data/plots/` in PDF (vector) and PNG formats.*

## 📂 Repository Structure

- `src/`: Core implementation.
  - `extract/`: Hidden state extraction logic.
  - `probes/`: Linear probe training and leakage validation.
  - `generate/`: Dataset generation scripts (Base, Stealthy, Complex).
  - `baselines/`: TF-IDF, Semantic, and Statistical baseline implementations.
  - `analysis/`: Comprehensive reporting and advanced stats.
- `data/`: Raw prompts, processed features, and visualization artifacts.
- `scripts/`: Utility scripts (plotting).

## 📄 Citation
If you use this work in your research, please cite:
```bibtex
@article{ag-iclr-safety-2026,
  title={Detecting Latent Prompt Injection Signals via Activation Probing},
  author={Amari, Ghanem and et al.},
  journal={ICLR Agent Safety Workshop},
  year={2026}
}
```
