import argparse
import os
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

def compute_fpr_at_recall(y_true, y_probs, recall_level=0.95):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Find threshold where tpr >= recall_level
    idx = np.searchsorted(tpr, recall_level)
    if idx < len(fpr):
        return fpr[idx]
    return 1.0

def train_probe_on_layer(X_layer, y):
    # X_layer: (N, d)
    # Simple split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_layer, y, test_size=0.3, random_state=42, stratify=y)
    
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, probs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex_dir", required=True, help="Directory containing results for complex dataset")
    args = ap.parse_args()
    
    report = {}
    
    # 1. Load Baseline Metrics
    # Semantic
    sem_path = os.path.join(args.complex_dir, "semantic_metrics.json")
    if os.path.exists(sem_path):
        with open(sem_path) as f:
            report["semantic"] = json.load(f)
            
    # Statistical
    stat_path = os.path.join(args.complex_dir, "statistical_metrics.json")
    if os.path.exists(stat_path):
        with open(stat_path) as f:
            report["statistical"] = json.load(f)
            
    # TF-IDF (assuming it printed to stdout, user might need to input it manually or we fetch from file if we saved it)
    # For now, we'll placeholder it or try to run it inside here if missing? 
    # Let's assume user provides it or we find a file. (Actually we didn't save a json for tf-idf in the command)
    
    # 2. Probe Analysis (Layer-wise)
    # Find the .npz file
    npz_files = [f for f in os.listdir(args.complex_dir) if f.endswith("_feats.npz")]
    if npz_files:
        npz_path = os.path.join(args.complex_dir, npz_files[0])
        print(f"Loading features from {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        # X might be (N, Layers * D) or we might need to verify shape
        # In src/extract/extract_activations.py, it stacks them: np.concatenate(all_layers, axis=1)
        # We need to know the dimension per layer. 
        # TinyLlama 1.1B dim is 2048.
        
        X = data["X"]
        y = data["y"]
        
        # Estimate layers
        dim = 2048
        num_layers = X.shape[1] // dim
        print(f"Detected {num_layers} layers (Total dim {X.shape[1]}).")
        
        layer_aucs = []
        for i in range(num_layers):
            start = i * dim
            end = (i + 1) * dim
            X_layer = X[:, start:end]
            auc = train_probe_on_layer(X_layer, y)
            layer_aucs.append({"layer_idx": i, "auc": auc})
            print(f"Layer {i}: AUC {auc:.4f}")
            
        report["probe_layerwise"] = layer_aucs
        report["probe_final_auc"] = layer_aucs[-1]["auc"] if layer_aucs else 0.0
        
    # 3. Save Final Report
    out_path = os.path.join(args.complex_dir, "comprehensive_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Saved comprehensive report to {out_path}")
    
    # Text summary
    print("\n=== FINAL RESULTS (Complex Dataset) ===")
    print(f"Semantic Baseline: {report.get('semantic', {}).get('semantic_auc_mean', 'N/A')}")
    print(f"Statistical Baseline: {report.get('statistical', {}).get('ppl_auc', 'N/A')}")
    if "probe_layerwise" in report:
        print("Probe Layer-wise:")
        for res in report["probe_layerwise"]:
            print(f"  L{res['layer_idx']}: {res['auc']:.4f}")

if __name__ == "__main__":
    main()
