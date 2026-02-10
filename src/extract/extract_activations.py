import json
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.models.load_model import load_llm

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def pick_layers(num_layers: int) -> List[int]:
    # 4 points : bas / milieu / haut / dernier
    return sorted(set([0, num_layers//4, num_layers//2, (3*num_layers)//4, num_layers-1]))

@torch.no_grad()
def extract_features(
    model_name: str,
    input_jsonl: str,
    out_npz: str,
    load_in_4bit: bool = False,
    max_length: int = 512,
):
    tok, model = load_llm(model_name, load_in_4bit=load_in_4bit)

    rows = read_jsonl(input_jsonl)
    labels = [r["label"] for r in rows]
    prompts = [r["prompt"] for r in rows]
    ids = [r["id"] for r in rows]

    # label mapping
    label_to_y = {"benign": 0, "injected": 1}
    #y = np.array([label_to_y[l] for l in labels], dtype=np.int64)

    y = np.array([normalize_label(l) for l in labels], dtype=np.int64)

    # figure out layer count
    num_layers = model.config.num_hidden_layers
    selected_layers = pick_layers(num_layers)

    X_list = []
    for p in tqdm(prompts, desc="Extracting"):
        enc = tok(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states  # tuple: (embeddings, layer1, ..., layerN)

        # On prend le dernier token de l'INPUT (représentation de contexte)
        last_token_idx = enc["input_ids"].shape[1] - 1

        feats = []
        for layer_idx in selected_layers:
            # +1 car hidden_states[0] = embeddings, hidden_states[1] = layer0
            h = hidden_states[layer_idx + 1][0, last_token_idx, :].detach().float().cpu().numpy()
            feats.append(h)

        feat_vec = np.concatenate(feats, axis=0)  # [len(layers)*hidden_size]
        X_list.append(feat_vec)

    X = np.stack(X_list, axis=0)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        ids=np.array(ids),
        layers=np.array(selected_layers),
        model_name=np.array([model_name]),
    )

    print(f"Saved features: {out_npz} | X={X.shape}, y={y.shape}, layers={selected_layers}")



def normalize_label(l):
    """Accepts int labels (0/1) or string labels ('benign'/'malicious')."""
    if isinstance(l, (int, float)):
        return int(l)

    if isinstance(l, str):
        s = l.strip().lower()
        if s in ["benign", "safe", "clean", "0"]:
            return 0
        if s in ["malicious", "unsafe", "injected", "1"]:
            return 1

    raise ValueError(f"Unknown label format: {l} ({type(l)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load4bit", action="store_true")
    args = ap.parse_args()

    extract_features(
        model_name=args.model,
        input_jsonl=args.input,
        out_npz=args.out,
        load_in_4bit=args.load4bit,
    )
