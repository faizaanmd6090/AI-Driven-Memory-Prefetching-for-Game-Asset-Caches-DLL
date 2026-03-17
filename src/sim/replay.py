"""
Trace replay driver: loads a CSV trace, runs cache + policy, computes metrics.
"""
import argparse
import pandas as pd
from .cache import LRUCache
from .policies import NoPrefetch, NGramPrefetch, MLPrefetch

def run(trace_path, cache_mb=512, policy_name="noprefetch", k=8, horizon_ms=500, model_path=None):
    df = pd.read_csv(trace_path)
    df = df.sort_values("t_ms").reset_index(drop=True)

    cache = LRUCache(capacity_bytes=cache_mb * 1024 * 1024)

    if policy_name == "noprefetch":
        policy = NoPrefetch()
    elif policy_name == "ngram":
        policy = NGramPrefetch(n=2)
    elif policy_name == "ml":
        assert model_path, "ml policy requires --model"
        policy = MLPrefetch(model_path)
    else:
        raise ValueError("unknown policy")

    late_loads = 0
    wasted_io = 0

    for i, row in df.iterrows():
        t = int(row["t_ms"])
        asset = row["asset_id"]
        size = int(row.get("bytes", 262144))

        # N-gram learner observes actual sequence
        if isinstance(policy, NGramPrefetch):
            policy.observe(asset)

        # Horizon window as oracle for candidates (keeps starter simple)
        window = df[(df["t_ms"] > t) & (df["t_ms"] <= t + horizon_ms)]
        candidates = list(window["asset_id"].unique())
        context = {
            "t_ms": t,
            "recent_assets": list(df.iloc[max(0, i-3):i]["asset_id"].values),
            "candidates": candidates,
        }

        recs = policy.recommend(context, k=k) if k > 0 else []

        # Prefetch
        for r in recs:
            if not cache.contains(r):
                loaded = cache.prefetch(r, size, t_ms=t)
                if loaded and r not in candidates:
                    wasted_io += size

        # Serve actual request
        hit = cache.get(asset, size, t_ms=t)
        if not hit:
            late_loads += 1

    total = len(df)
    hit_rate = cache.stats.hits / total if total else 0.0

    return {
        "requests": total,
        "hit_rate": hit_rate,
        "late_loads": late_loads,
        "bytes_loaded": cache.stats.bytes_loaded,
        "bytes_evicted": cache.stats.bytes_evicted,
        "wasted_io_bytes": wasted_io,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--cache_mb", type=int, default=512)
    ap.add_argument("--policy", choices=["noprefetch", "ngram", "ml"], default="noprefetch")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--horizon_ms", type=int, default=500)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    metrics = run(args.trace, cache_mb=args.cache_mb, policy_name=args.policy, k=args.k, horizon_ms=args.horizon_ms, model_path=args.model)
    print(metrics)

if __name__ == "__main__":
    main()
