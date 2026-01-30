@"
# iclr-agent-safety

Reproducible pipeline for detecting prompt-injection signals using:
- TF-IDF text baseline
- Activation extraction from a HF model
- Linear probes + cross-validation
- Group split evaluation by pair_id (prevents paired leakage)

## Setup
```powershell
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
