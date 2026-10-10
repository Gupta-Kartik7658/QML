"""Persistent, content-addressed cache for quantum features.

Key = hash(prepared-data key [dataset fingerprint + split + seed + preprocessing],
           input-data hash, selected feature names, feature-map parameters, observables,
           architecture + noise parameters, shots, transpiler seed, qiskit/aer/numpy versions).
Any change in any of these produces a new key, so stale features are never reused.
Writes are atomic (temp file + os.replace), so a crash never leaves a half-written entry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import array_hash, load_npz, quantum_versions, save_npz, stable_hash

CACHE_VERSION = 2


class FeatureCache:
    def __init__(self, cache_dir: str | Path, logger=None):
        self.dir = Path(cache_dir) / "features"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.logger = logger

    def key(self, **parts) -> str:
        parts = dict(parts)
        parts["_v"] = CACHE_VERSION
        parts["_versions"] = quantum_versions()
        return stable_hash(parts, 24)

    def get_or_compute(self, key: str, compute, expect: dict | None = None) -> dict:
        path = self.dir / f"{key}.npz"
        if path.exists():
            try:
                d = load_npz(path)
                if self._valid(d, expect):
                    self.hits += 1
                    return d
                if self.logger:
                    self.logger.warning(f"[cache] invalid entry {path.name} -> recomputing")
            except Exception as e:  # corrupted file
                if self.logger:
                    self.logger.warning(f"[cache] unreadable {path.name} ({e}) -> recomputing")
        self.misses += 1
        d = compute()
        save_npz(path, compress=False, **d)
        return d

    def get(self, key: str, expect: dict | None = None):
        """Cached entry or None (never computes)."""
        path = self.dir / f"{key}.npz"
        if not path.exists():
            return None
        try:
            d = load_npz(path)
            return d if self._valid(d, expect) else None
        except Exception:
            return None

    @staticmethod
    def _valid(d: dict, expect: dict | None) -> bool:
        for k, v in d.items():
            if v.dtype.kind == "f" and not np.isfinite(v).all():
                return False
        if expect:
            for k, shape in expect.items():
                if k not in d or tuple(d[k].shape) != tuple(shape):
                    return False
        return True

    def stats(self) -> dict:
        return {"cache_hits": self.hits, "cache_misses": self.misses}


def input_hash(X: np.ndarray) -> str:
    return array_hash(np.asarray(X, dtype=np.float32))
