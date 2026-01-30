import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from src.eval.metrics import compute_metrics
from src.utils.io import write_json

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def train_probe(npz_path: str, out_json: str = None, test_size: float = 0.3, seed: int = 42):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    model_name = str(data["model_name"][0]) if "model_name" in data else "unknown"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    #clf = LogisticRegression(max_iter=5000, n_jobs=-1, class_weight="balanced")
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, probs)

    metrics.update({
        "model_name": model_name,
        "npz": npz_path,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X.shape[1]),
    })

    print(metrics)

    if out_json:
        out_dir = os.path.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        write_json(out_json, metrics)
        print(f"Saved metrics to {out_json}")

    return metrics


def layer_sweep(npz_path: str, test_size: float = 0.3, seed: int = 42):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    layers = data["layers"]
    model_name = str(data["model_name"][0]) if "model_name" in data else "unknown"

    n_layers = len(layers)
    hidden_size = X.shape[1] // n_layers

    results = []

    for i, layer in enumerate(layers):
        Xi = X[:, i*hidden_size:(i+1)*hidden_size]

        X_train, X_test, y_train, y_test = train_test_split(
            Xi, y,
            test_size=test_size,
            random_state=seed,
            stratify=y
        )

        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced"
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, probs)

        metrics.update({
            "layer": int(layer),
            "hidden_size": int(hidden_size),
            "model_name": model_name
        })

        results.append(metrics)

    return results



def cv_probe_auc(X, y, seed=42, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], probs))
    return float(np.mean(aucs)), float(np.std(aucs))



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None, help="Path to save metrics json (optional)")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", action="store_true", help="Run layer-wise probe sweep")
    ap.add_argument("--cv", type=int, default=0, help="Run stratified K-fold CV (e.g. --cv 5)")
    args = ap.parse_args()




    train_probe(args.npz, out_json=args.out, test_size=args.test_size, seed=args.seed)



    if args.sweep:
        sweep_results = layer_sweep(
            args.npz,
            test_size=args.test_size,
            seed=args.seed
        )
        for r in sweep_results:
            print(r)

    from src.utils.io import write_json

    if args.sweep:
        sweep_path = args.npz.replace("_feats.npz", "_layer_sweep.json")
        write_json(sweep_path, {"results": sweep_results})
        print(f"Saved layer sweep to {sweep_path}")



    # Cross-validation
    if args.cv and args.cv > 1:
        data = np.load(args.npz, allow_pickle=True)
        X = data["X"]
        y = data["y"]

        mean_auc, std_auc = cv_probe_auc(X, y, seed=args.seed, k=args.cv)
        print({
            "cv_folds": args.cv,
            "cv_auroc_mean": mean_auc,
            "cv_auroc_std": std_auc
        })
