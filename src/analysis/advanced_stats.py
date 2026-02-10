import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

def bootstrap_auc(y_true, y_scores, n_bootstraps=2000, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue

        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95% CI
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    return confidence_lower, confidence_upper

def compute_separation_stats(X, y):
    # Train a quick probe to get the direction
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X, y)
    
    # Project data onto the weight vector (signed distance)
    # w is (1, d), X is (N, d) -> scores (N,)
    scores = X.dot(clf.coef_.T).flatten() + clf.intercept_
    
    benign_scores = scores[y == 0]
    injected_scores = scores[y == 1]
    
    # Cohen's d
    # d = (mean1 - mean2) / pooled_std
    n1, n2 = len(benign_scores), len(injected_scores)
    var1, var2 = np.var(benign_scores, ddof=1), np.var(injected_scores, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(injected_scores) - np.mean(benign_scores)) / pooled_std
    
    # Mann-Whitney U test (non-parametric significance)
    # We want to see if Injected is stochastically greater than Benign
    stat, p_value = mannwhitneyu(injected_scores, benign_scores, alternative='two-sided')
    
    return {
        "cohens_d": cohens_d,
        "p_value": p_value,
        "mean_diff": np.mean(injected_scores) - np.mean(benign_scores),
        "scores": scores
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    print(f"Loading {args.npz}...")
    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    
    print(f"Computing advanced stats on N={len(y)} samples...")
    stats = compute_separation_stats(X, y)
    
    print("\n--- Advanced Stats ---")
    print(f"P-value (Mann-Whitney): {stats['p_value']:.4e}")
    print(f"Cohen's d: {stats['cohens_d']:.4f}")
    
    if stats['p_value'] < 0.001:
        print("Result is Statistically Significant (p < 0.001)")
    
    if abs(stats['cohens_d']) > 0.8:
        print("Effect Size is LARGE (d > 0.8)")
        
    # Bootstrap CI for AUC using CV predictions
    # We need to run CV to get unbiased predictions for the whole dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(y))
    
    # Simple standardized CV
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
        
    global_auc = roc_auc_score(y, preds)
    ci_lower, ci_upper = bootstrap_auc(y, preds)
    
    print(f"\nGlobal 5-Fold AUC: {global_auc:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Save results
    import json
    results = {
        "p_value": float(stats["p_value"]),
        "cohens_d": float(stats["cohens_d"]),
        "auc_mean": float(global_auc),
        "auc_ci_lower": float(ci_lower),
        "auc_ci_upper": float(ci_upper),
        "n_samples": int(len(y))
    }
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved stats to {args.out}")

if __name__ == "__main__":
    main()
