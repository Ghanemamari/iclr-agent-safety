


# src/baselines/text_tfidf.py
import json
import argparse
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.eval.metrics import compute_metrics


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def labels_to_int(labels):
    # accepte: 0/1, "benign"/"injected", "safe"/"unsafe"
    out = []
    for y in labels:
        if isinstance(y, (int, np.integer)):
            out.append(int(y))
        else:
            s = str(y).lower()
            if s in ["1", "true", "unsafe", "malicious", "injected"]:
                out.append(1)
            else:
                out.append(0)
    return np.array(out, dtype=int)


def train_and_eval(texts_train, y_train, texts_test, y_test, seed=0):
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True,
    )
    X_train = vec.fit_transform(texts_train)
    X_test = vec.transform(texts_test)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, probs)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group_by_pair_id", action="store_true",
                    help="Use GroupShuffleSplit by pair_id (prevents paired leakage)")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--cv", type=int, default=0,
                    help="Stratified K-fold CV on text baseline (0 disables). NOTE: CV is not group-aware.")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    texts = [r["prompt"] for r in rows]
    y = labels_to_int([r["label"] for r in rows])

    if args.cv and args.cv > 1:
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        aucs = []
        for tr, te in skf.split(np.zeros(len(y)), y):
            m = train_and_eval(
                [texts[i] for i in tr], y[tr],
                [texts[i] for i in te], y[te],
                seed=args.seed
            )
            aucs.append(m["auroc"])
        print({"cv_folds": args.cv, "cv_auroc_mean": float(np.mean(aucs)), "cv_auroc_std": float(np.std(aucs))})
        return

    if args.group_by_pair_id:
        groups = [r.get("pair_id", i) for i, r in enumerate(rows)]
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(gss.split(texts, y, groups=groups))
        metrics = train_and_eval(
            [texts[i] for i in tr_idx], y[tr_idx],
            [texts[i] for i in te_idx], y[te_idx],
            seed=args.seed
        )
        metrics["split"] = "group_by_pair_id"
        metrics["n_train"] = int(len(tr_idx))
        metrics["n_test"] = int(len(te_idx))
        print(metrics)
        return

    # random split standard
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    metrics = train_and_eval(X_tr, y_tr, X_te, y_te, seed=args.seed)
    metrics["split"] = "random"
    metrics["n_train"] = int(len(y_tr))
    metrics["n_test"] = int(len(y_te))
    print(metrics)


if __name__ == "__main__":
    main()
