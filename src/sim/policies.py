"""
Prefetch policies: NoPrefetch, NGram baseline, MLPrefetch wrapper.
"""
from collections import defaultdict, deque
import heapq
import pickle

class NoPrefetch:
    def recommend(self, context, k=0):
        return []

class NGramPrefetch:
    def __init__(self, n=2):
        self.n = n
        self.freq = defaultdict(lambda: defaultdict(int))
        self.buffer = deque(maxlen=n)

    def observe(self, asset_id):
        if len(self.buffer) == self.n:
            key = tuple(self.buffer)
            self.freq[key][asset_id] += 1
        self.buffer.append(asset_id)

    def recommend(self, context, k=8):
        recent = context.get("recent_assets", [])
        key = tuple(recent[-self.n:])
        cand = self.freq.get(key, {})
        top = heapq.nlargest(k, cand.items(), key=lambda x: x[1])
        return [a for a,_ in top]

class MLPrefetch:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def _features(self, context, candidate):
        import numpy as np
        last = context.get("recent_assets", [-1])[-1]
        last_h = hash(last) % 100000 if isinstance(last, str) else int(last)
        cand_h = hash(candidate) % 100000 if isinstance(candidate, str) else int(candidate)
        dist = context.get("candidate_distances", {}).get(candidate, 0.0)
        t_mod = context.get("t_ms", 0) % 1000
        return np.array([t_mod, last_h, cand_h, float(dist)], dtype=float)

    def recommend(self, context, k=8):
        candidates = context.get("candidates", [])
        if not candidates:
            return []
        import numpy as np
        X = np.vstack([self._features(context, c) for c in candidates])
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(X)[:, 1]
        else:
            scores = self.model.predict(X)
        top_idx = scores.argsort()[-k:][::-1]
        return [candidates[i] for i in top_idx]
