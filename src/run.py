import os
import argparse

from src.extract.extract_activations import extract_features
from src.probes.train_linear_probe import train_probe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default="data/raw/prompts.jsonl")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--load4bit", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    safe_model_name = args.model.replace("/", "_").replace(":", "_")
    npz_path = os.path.join(args.outdir, f"{safe_model_name}_feats.npz")
    metrics_path = os.path.join(args.outdir, f"{safe_model_name}_metrics.json")

    extract_features(
        model_name=args.model,
        input_jsonl=args.input,
        out_npz=npz_path,
        load_in_4bit=args.load4bit,
    )

    train_probe(npz_path, out_json=metrics_path)

if __name__ == "__main__":
    main()
