"""
Train a lightweight classifier to score candidate assets.
"""
import argparse, pickle
import pandas as pd
from .features import make_training_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--horizon_ms", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.trace)
    X, y = make_training_data(df, horizon_ms=args.horizon_ms)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
    print({"auc": auc})
    with open(args.out, "wb") as f:
        pickle.dump(clf, f)

if __name__ == "__main__":
    main()
