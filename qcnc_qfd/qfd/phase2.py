"""Shared helpers for the Phase-2 experiments (hybrid features, shot budget, theory check, IBM hardware).

Everything is read from the SAME caches as the main run (prepared splits, SFS subsets, ideal and noisy
features), so Phase-2 numbers are computed on exactly the rows / subsets / features of the main tables.
"""
from __future__ import annotations

import numpy as np

from .cache import FeatureCache
from .featuremaps import fmap_params, observable_matrices
from .pipeline import _ideal_cached, _noisy_cached, get_sfs
from .preprocess import load_prepared

PARTS = ("train", "val", "test")


def load_task(cfg: dict, prep_dir: str, dataset: str, fmap: str, seed: int, logger=None) -> dict:
    """Prepared arrays + SFS subset (cached) for one (dataset, feature map, seed)."""
    A, meta = load_prepared(prep_dir)
    names = meta["feature_names"]
    fm = fmap_params(cfg, fmap)
    st = get_sfs(cfg, A, meta, dataset, fmap, seed, logger)
    cols = [names.index(f) for f in st["best_subset"]]
    return {"A": A, "meta": meta, "fm": fm, "sfs": st, "cols": cols, "names": names, "k": len(cols)}


def ideal_z(cfg: dict, T: dict, logger=None) -> dict:
    """Ideal <Z_i> features (train/val/test, full classifier rows) from the shared feature cache."""
    A = T["A"]
    cache = FeatureCache(cfg["paths"]["cache_dir"], logger)
    X3 = {p: A[f"Xq_{p}"] for p in PARTS}
    F = _ideal_cached(cache, T["meta"]["key"], X3, T["cols"], T["names"], T["fm"], extended=True)
    return {p: np.asarray(F[p][:, :T["k"]], dtype=np.float64) for p in PARTS}


def paired_rows(T: dict) -> dict:
    """The paired noisy-simulation subsets (positions inside train/val/test), labels and weights."""
    A = T["A"]
    return {"idx": {"train": A["noisy_tr"], "val": A["noisy_va"], "test": A["noisy_te"]},
            "y": {"train": A["y_train"][A["noisy_tr"]], "val": A["y_val"][A["noisy_va"]],
                  "test": A["y_test"][A["noisy_te"]]},
            "w": {"train": None, "val": A["w_noisy_val"], "test": A["w_noisy_test"]}}


def noisy_probs(cfg: dict, T: dict, arch: str, allow_compute: bool = False, logger=None):
    """Noisy probabilities on the paired rows (exactly the main run's cache entry), or None if not cached."""
    A = T["A"]
    R = paired_rows(T)
    Xn = {p: A[f"Xq_{p}"][R["idx"][p]] for p in PARTS}
    cache = FeatureCache(cfg["paths"]["cache_dir"], logger)
    return _noisy_cached(cache, T["meta"]["key"], Xn, T["cols"], T["names"], T["fm"], arch,
                         cfg["architectures"][arch], cfg["noise"], allow_compute=allow_compute)


def ideal_probs_paired(T: dict) -> dict:
    from .featuremaps import ideal_probabilities
    A = T["A"]
    R = paired_rows(T)
    return {p: ideal_probabilities(A[f"Xq_{p}"][R["idx"][p]][:, T["cols"]], T["fm"]) for p in PARTS}


def z_from_probs(P: dict, k: int) -> dict:
    M, _ = observable_matrices(k, False)
    return {p: np.asarray(P[p]) @ M for p in PARTS}


def affine_fit(ideal: np.ndarray, noisy: np.ndarray) -> list[dict]:
    """Per-feature least-squares fit noisy_i = alpha_i + beta_i * ideal_i, with max residual and R^2."""
    out = []
    for i in range(ideal.shape[1]):
        X = np.c_[np.ones(len(ideal)), ideal[:, i]]
        c, *_ = np.linalg.lstsq(X, noisy[:, i], rcond=None)
        r = noisy[:, i] - X @ c
        ss = ((noisy[:, i] - noisy[:, i].mean()) ** 2).sum()
        out.append({"feature": i, "alpha": float(c[0]), "beta": float(c[1]), "max_abs_residual": float(np.abs(r).max()),
                    "residual_sd": float(r.std(ddof=2)) if len(r) > 2 else float("nan"),
                    "r2": float(1 - (r ** 2).sum() / ss) if ss > 0 else float("nan")})
    return out
