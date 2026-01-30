import argparse, json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from src.eval.metrics import compute_metrics

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main(npz_path, jsonl_path, seed=42, test_size=0.3):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)

    rows = read_jsonl(jsonl_path)
    assert len(rows) == len(y), f"JSONL rows={len(rows)} but y={len(y)} (mismatch)"

    groups = np.array([r["pair_id"] for r in rows])

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(X[train_idx], y[train_idx])

    probs = clf.predict_proba(X[test_idx])[:, 1]
    m = compute_metrics(y[test_idx], probs)
    m.update({"split": "group_by_pair_id", "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
    print(m)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.3)
    args = ap.parse_args()
    main(args.npz, args.jsonl, seed=args.seed, test_size=args.test_size)
