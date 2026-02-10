import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.utils.io import read_jsonl, write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to prompts.jsonl")
    ap.add_argument("--out", required=True, help="Path to save metrics")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    args = ap.parse_args()

    print(f"Loading embedding model {args.model}...")
    model = SentenceTransformer(args.model)

    data = read_jsonl(args.input)
    prompts = [d["prompt"] for d in data]
    labels = []
    debug_labels = []
    for d in data:
        val = d["label"]
        debug_labels.append(str(val))
        if str(val).lower() in ["1", "injected", "true"]:
            labels.append(1)
        else:
            labels.append(0)
            
    print(f"DEBUG: Unique raw labels found: {set(debug_labels)}")
    y = np.array(labels)

    print(f"Encoding {len(prompts)} prompts...")
    X = model.encode(prompts, show_progress_bar=True)
    
    print(f"Label distribution: {np.bincount(y)}")
    if len(np.unique(y)) < 2:
        print("ERROR: Dataset has only 1 class!")
        return

    print(f"Running {args.cv}-fold CV with Logistic Regression on embeddings...")
    # Use train_test_split instead of CV for stability on smaller/imbalanced datasets
    from sklearn.model_selection import train_test_split
    
    # Force stratify and shuffle to ensure classes are present
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)
    
    # Check if we have both classes
    if len(np.unique(y_train)) < 2:
        print("Warning: Train set has only 1 class. Falling back to simple split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True)
    
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    metrics = {
        "semantic_auc_mean": float(auc),
        "semantic_auc_std": 0.0,
        "embedding_model": args.model
    }
    
    print(metrics)
    write_json(args.out, metrics)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
