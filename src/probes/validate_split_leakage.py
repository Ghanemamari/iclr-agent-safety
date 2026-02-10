import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from src.utils.io import write_json

def get_group_id(sample_id):
    # hb01 -> 01
    # hi01 -> 01
    # Assumes format xxNN
    return str(sample_id)[-2:]

def validate_split(npz_path, out_json):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    ids = data["ids"]
    model_name = str(data["model_name"][0]) if "model_name" in data else "unknown"

    # Create groups
    groups = np.array([get_group_id(sid) for sid in ids])
    print(f"Found {len(np.unique(groups))} unique groups: {np.unique(groups)}")

    # Use StratifiedGroupKFold to ensure we keep balance if possible, or just GroupKFold
    # We want to ensure that if Group 01 is in test, ALL of Group 01 is in test.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    aucs = []
    accs = []
    
    print(f"Running 5-fold Stratified Group Cross-Validation...")
    
    for i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check if groups overlap
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert len(train_groups.intersection(test_groups)) == 0, "Group leakage detected!"
        
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        
        aucs.append(auc)
        accs.append(acc)
        print(f"Fold {i+1}: AUC={auc:.4f}, Acc={acc:.4f} | Test Groups: {sorted(list(test_groups))}")

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    mean_acc = float(np.mean(accs))
    
    print(f"\nResults:")
    print(f"Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    
    metrics = {
        "group_cv_auc_mean": mean_auc,
        "group_cv_auc_std": std_auc,
        "group_cv_acc_mean": mean_acc,
        "model_name": model_name
    }
    
    if out_json:
        write_json(out_json, metrics)
        print(f"Saved to {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    
    validate_split(args.npz, args.out)
