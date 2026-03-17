"""
Simple LRU cache with prefetch hooks for asset caching.
"""
from collections import OrderedDict
from dataclasses import dataclass

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    bytes_loaded: int = 0
    bytes_evicted: int = 0

class LRUCache:
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.used = 0
        self.store = OrderedDict()   # asset_id -> (size_bytes, t_insert_ms)
        self.stats = CacheStats()

    def contains(self, asset_id: str) -> bool:
        return asset_id in self.store

    def touch(self, asset_id: str):
        size, t_insert = self.store.pop(asset_id)
        self.store[asset_id] = (size, t_insert)

    def get(self, asset_id: str, size_bytes: int, t_ms: int):
        if self.contains(asset_id):
            self.stats.hits += 1
            self.touch(asset_id)
            return True
        self.stats.misses += 1
        self._ensure_space(size_bytes)
        self.store[asset_id] = (size_bytes, t_ms)
        self.used += size_bytes
        self.stats.bytes_loaded += size_bytes
        return False

    def prefetch(self, asset_id: str, size_bytes: int, t_ms: int):
        if self.contains(asset_id):
            self.touch(asset_id)
            return False  # nothing loaded
        self._ensure_space(size_bytes)
        self.store[asset_id] = (size_bytes, t_ms)
        self.used += size_bytes
        self.stats.bytes_loaded += size_bytes
        return True  # loaded

    def evict(self, asset_id: str):
        if asset_id in self.store:
            size, _ = self.store.pop(asset_id)
            self.used -= size
            self.stats.bytes_evicted += size

    def _ensure_space(self, needed: int):
        while self.used + needed > self.capacity and len(self.store) > 0:
            old_id, (size, _) = self.store.popitem(last=False)  # pop LRU
            self.used -= size
            self.stats.bytes_evicted += size
