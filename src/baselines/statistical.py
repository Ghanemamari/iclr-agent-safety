import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from src.utils.io import read_jsonl, write_json

def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        # loss is explicitly the cross-entropy loss
        neg_log_likelihood = outputs.loss
    
    # Perplexity is exp(cross_entropy)
    ppl = torch.exp(neg_log_likelihood)
    return ppl.item(), neg_log_likelihood.item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model path")
    ap.add_argument("--input", required=True, help="Path to prompts.jsonl")
    ap.add_argument("--out", required=True, help="Path to save metrics")
    ap.add_argument("--load4bit", action="store_true")
    args = ap.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True
    }
    if args.load4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        **model_kwargs
    )

    data = read_jsonl(args.input)
    ppls = []
    nlls = []
    labels = []

    print("Calculating perplexity...")
    for entry in tqdm(data):
        prompt = entry["prompt"]
        label = 1 if entry["label"] == "injected" else 0
        
        ppl, nll = calculate_perplexity(model, tokenizer, prompt)
        ppls.append(ppl)
        nlls.append(nll)
        labels.append(label)

    # Calculate metrics
    # For PPL, we assume injections might have HIGHER perplexity (weird syntax) 
    # OR LOWER (if they are very common command sequences, but usually injections are 'weird')
    # We'll test both directions or just report AUC using the score directly.
    # Usually AUC handles the sign, but we should be aware.
    
    metrics = {
        "ppl_auc": roc_auc_score(labels, ppls),
        "nll_auc": roc_auc_score(labels, nlls),
        "mean_ppl_benign": float(np.mean([p for p, l in zip(ppls, labels) if l == 0])),
        "mean_ppl_injected": float(np.mean([p for p, l in zip(ppls, labels) if l == 1])),
    }
    
    print(metrics)
    write_json(args.out, metrics)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
