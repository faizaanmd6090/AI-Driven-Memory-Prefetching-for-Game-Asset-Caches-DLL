"""
Feature engineering and labeling utilities.
"""
import pandas as pd

def make_training_data(df: pd.DataFrame, horizon_ms: int = 500):
    df = df.sort_values("t_ms").reset_index(drop=True)
    X_rows, y_rows = [], []
    for i, row in df.iterrows():
        t = int(row["t_ms"])
        recent = df.iloc[max(0, i-1):i]["asset_id"].tolist()
        last = recent[-1] if recent else ""
        window = df[(df["t_ms"] > t) & (df["t_ms"] <= t + horizon_ms)]
        # positives: assets used within horizon
        for cand in window["asset_id"].unique():
            X_rows.append([t % 1000, hash(last) % 100000, hash(cand) % 100000])
            y_rows.append(1)
        # negatives: a few not in window
        negs = list(set(df["asset_id"].unique()) - set(window["asset_id"].unique()))
        for cand in negs[:5]:
            X_rows.append([t % 1000, hash(last) % 100000, hash(cand) % 100000])
            y_rows.append(0)
    import numpy as np
    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    return X, y
