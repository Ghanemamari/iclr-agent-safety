import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.eval.metrics import compute_metrics

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main(path, test_size=0.3, seed=42):
    rows = read_jsonl(path)
    texts = [r["prompt"] for r in rows]
    y = [0 if r["label"] == "benign" else 1 for r in rows]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=seed, stratify=y
    )

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(Xtr, y_train)

    probs = clf.predict_proba(Xte)[:, 1]
    print(compute_metrics(y_test, probs))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    main(args.input)