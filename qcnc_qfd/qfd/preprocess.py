"""Leakage-free splitting, preprocessing and subsampling.

Order of operations for one (dataset, seed):
  1. split all rows into train / validation / test (stratified or chronological)
  2. FIT every learned transform on the TRAINING partition only
       signed-log -> (train-only, out-of-fold) target encoding -> median imputation
       -> constant-feature removal -> {StandardScaler (classical view), quantile/minmax (angle view)}
  3. subsample: training rows (training-only undersampling), validation and test via
     case-control sampling of genuine rows with inverse-probability weights, so weighted
     metrics are unbiased estimates of the metric at the ORIGINAL prevalence.
  4. save float32 arrays + metadata to cache/prepared/<dataset>/<key>/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .datasets import RawDataset, load_dataset
from .utils import get_logger, load_npz, read_json, save_npz, stable_hash, write_json

PREP_VERSION = 4  # bump when this module's behaviour changes -> invalidates prepared caches


# ----------------------------------------------------------------------------- split
def make_split(y: np.ndarray, time: np.ndarray | None, mode: str, fractions, seed: int):
    n = len(y)
    idx = np.arange(n)
    f_tr, f_va, f_te = fractions
    if mode == "stratified":
        tr, rest = train_test_split(idx, train_size=f_tr, stratify=y, random_state=seed)
        va, te = train_test_split(rest, train_size=f_va / (f_va + f_te), stratify=y[rest],
                                  random_state=seed + 1)
    elif mode == "temporal":
        if time is None:
            raise ValueError("temporal split requested but dataset has no time column")
        order = np.lexsort((idx, time))  # stable: time, then original order
        a, b = int(round(f_tr * n)), int(round((f_tr + f_va) * n))
        tr, va, te = order[:a], order[a:b], order[b:]
    else:
        raise ValueError(mode)
    tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
    assert len(np.intersect1d(tr, va)) == 0 and len(np.intersect1d(tr, te)) == 0 \
        and len(np.intersect1d(va, te)) == 0, "split overlap"
    return tr, va, te


# ----------------------------------------------------------------------------- transforms
def signed_log1p(a):
    return np.sign(a) * np.log1p(np.abs(a))


class Preprocessor:
    """All learned transforms; fit() sees training rows only."""

    def __init__(self, log_cols, te_cols, angle_scaling="quantile", te_smoothing=20.0, seed=0):
        self.log_cols, self.te_cols = list(log_cols), list(te_cols)
        self.angle_scaling = angle_scaling
        self.m, self.seed = te_smoothing, seed

    # target encoding ---------------------------------------------------------
    def _te_fit_map(self, cats, y):
        prior = float(y.mean())
        df = pd.DataFrame({"c": cats, "y": y}).groupby("c")["y"].agg(["sum", "count"])
        enc = (df["sum"] + self.m * prior) / (df["count"] + self.m)
        return enc.to_dict(), prior

    def _te_apply(self, cats, mapping, prior):
        return pd.Series(cats).map(mapping).fillna(prior).to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: np.ndarray, te_raw: dict | None):
        X = X.copy()
        self.columns_in = list(X.columns)
        for c in self.log_cols:
            X[c] = signed_log1p(X[c].to_numpy())
        self.te_maps, self.te_oof = {}, {}
        for c in self.te_cols:
            cats = te_raw[c]
            self.te_maps[c] = self._te_fit_map(cats, y)
            # out-of-fold encoding for the training rows themselves (no self-leakage)
            oof = np.empty(len(y))
            for a, b in KFold(5, shuffle=True, random_state=self.seed).split(cats):
                mp, pr = self._te_fit_map(cats[a], y[a])
                oof[b] = self._te_apply(cats[b], mp, pr)
            X[c] = oof
            self.te_oof[c] = oof
        self.medians = X.median(numeric_only=True)
        X = X.fillna(self.medians)
        std = X.std(ddof=0)
        self.constant_cols = [c for c in X.columns if not np.isfinite(std[c]) or std[c] == 0.0]
        self.columns = [c for c in X.columns if c not in self.constant_cols]
        A = X[self.columns].to_numpy(dtype=float)
        self.std_scaler = StandardScaler().fit(A)
        self.binary = np.array([set(np.unique(A[:, j])).issubset({0.0, 1.0}) for j in range(A.shape[1])])
        if self.angle_scaling == "quantile":
            self.qt = QuantileTransformer(n_quantiles=min(1000, len(A)), output_distribution="uniform",
                                          subsample=200_000, random_state=self.seed).fit(A)
        elif self.angle_scaling == "minmax":
            self.min_, self.max_ = A.min(0), A.max(0)
        else:
            raise ValueError(self.angle_scaling)
        return self

    def transform(self, X: pd.DataFrame, te_raw: dict | None, is_train_rows: np.ndarray | None = None):
        """is_train_rows: positions (into X) that are training rows -> use their OOF target encoding."""
        X = X.copy()
        for c in self.log_cols:
            X[c] = signed_log1p(X[c].to_numpy())
        for c in self.te_cols:
            mapping, prior = self.te_maps[c]
            X[c] = self._te_apply(te_raw[c], mapping, prior)
            if is_train_rows is not None:
                X.loc[X.index[is_train_rows["pos"]], c] = self.te_oof[c][is_train_rows["oof_pos"]]
        X = X.fillna(self.medians)
        A = X[self.columns].to_numpy(dtype=float)
        Xc = self.std_scaler.transform(A)
        if self.angle_scaling == "quantile":
            U = self.qt.transform(A)
        else:
            U = (A - self.min_) / np.where(self.max_ > self.min_, self.max_ - self.min_, 1.0)
        U = np.where(self.binary[None, :], (A > 0.5).astype(float), U)
        U = np.clip(U, 0.0, 1.0)  # unseen extremes (val/test) are clipped, never refitted
        return Xc.astype(np.float32), U.astype(np.float32)  # U in [0,1]; feature map sets angles

    def describe(self) -> dict:
        return {"log_cols": self.log_cols, "target_encoded": self.te_cols,
                "te_smoothing": self.m, "constant_cols_dropped": self.constant_cols,
                "binary_cols": [c for c, b in zip(self.columns, self.binary) if b],
                "angle_scaling": f"{self.angle_scaling} -> [0,1] (feature map applies its input_range)",
                "classical_scaling": "StandardScaler (train-fitted)", "imputation": "train median"}


# ----------------------------------------------------------------------------- subsampling
def _case_control(pos, neg, neg_cap, rng):
    """all positives + <= neg_cap negatives; weights make metrics unbiased for the full set."""
    if neg_cap is None or neg_cap >= len(neg):
        keep_neg, w_neg = neg, 1.0
    else:
        keep_neg = rng.choice(neg, int(neg_cap), replace=False)
        w_neg = len(neg) / float(neg_cap)
    keep = np.sort(np.concatenate([pos, keep_neg]))
    return keep, w_neg


def subsample_plan(y, tr, va, te, s: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed + 777)
    # -- training rows (training-only undersampling of the genuine class)
    tp, tn = tr[y[tr] == 1], tr[y[tr] == 0]
    n_pos = min(len(tp), int(s["train_cap"] * s["train_max_pos_frac"]))
    n_neg = min(len(tn), s["train_cap"] - n_pos)
    tr_sub = np.sort(np.concatenate([rng.choice(tp, n_pos, replace=False), rng.choice(tn, n_neg, replace=False)]))
    # -- validation and test: case-control with weights
    va_sub, wv = _case_control(va[y[va] == 1], va[y[va] == 0], s["val_neg_cap"], rng)
    te_sub, wt = _case_control(te[y[te] == 1], te[y[te] == 0], s["test_neg_cap"], rng)
    w_val = np.where(y[va_sub] == 1, 1.0, wv)
    w_test = np.where(y[te_sub] == 1, 1.0, wt)
    # -- SFS evaluator rows: stratified subset of training rows
    if s["sfs_train_cap"] < len(tr_sub):
        sfs_pos, _ = train_test_split(np.arange(len(tr_sub)), train_size=s["sfs_train_cap"],
                                      stratify=y[tr_sub], random_state=seed + 3)
        sfs_pos = np.sort(sfs_pos)
    else:
        sfs_pos = np.arange(len(tr_sub))
    # -- noisy-simulation subsets (positions inside the *_sub arrays), paired with ideal
    ytr = y[tr_sub]
    p_ = np.flatnonzero(ytr == 1)
    n_ = np.flatnonzero(ytr == 0)
    npos = min(len(p_), s["noisy_train_max_pos"], s["noisy_train_cap"] // 2)
    nneg = min(len(n_), s["noisy_train_cap"] - npos)
    noisy_tr = np.sort(np.concatenate([rng.choice(p_, npos, replace=False), rng.choice(n_, nneg, replace=False)]))
    yv, yt = y[va_sub], y[te_sub]
    nv_keep, _ = _case_control(np.flatnonzero(yv == 1), np.flatnonzero(yv == 0), s["noisy_val_neg"], rng)
    nt_keep, _ = _case_control(np.flatnonzero(yt == 1), np.flatnonzero(yt == 0), s["noisy_test_neg"], rng)
    # weights relative to the FULL validation/test partitions
    full_val_neg, full_test_neg = int((y[va] == 0).sum()), int((y[te] == 0).sum())
    w_noisy_val = np.where(yv[nv_keep] == 1, 1.0, full_val_neg / max(1, int((yv[nv_keep] == 0).sum())))
    w_noisy_test = np.where(yt[nt_keep] == 1, 1.0, full_test_neg / max(1, int((yt[nt_keep] == 0).sum())))
    return dict(tr_sub=tr_sub, va_sub=va_sub, te_sub=te_sub, w_val=w_val, w_test=w_test, sfs_pos=sfs_pos,
                noisy_tr=noisy_tr, noisy_va=nv_keep, noisy_te=nt_keep,
                w_noisy_val=w_noisy_val, w_noisy_test=w_noisy_test)


# ----------------------------------------------------------------------------- prepare
def prepared_key(ds: RawDataset, cfg: dict, seed: int) -> str:
    return stable_hash({
        "v": PREP_VERSION, "fp": ds.fingerprint, "ds_cfg": cfg["datasets"][ds.name],
        "split": cfg["split"], "sampling": cfg["sampling"], "prep": cfg["preprocessing"],
        "seed": seed, "limit": cfg["run"].get("sample_limit")})


def prepare_one(ds: RawDataset, cfg: dict, seed: int, logger=None) -> Path:
    key = prepared_key(ds, cfg, seed)
    out = Path(cfg["paths"]["cache_dir"]) / "prepared" / ds.name / f"seed{seed}_{key}"
    if (out / "meta.json").exists() and (out / "arrays.npz").exists():
        if logger:
            logger.info(f"[prepare] cache hit {ds.name} seed={seed} ({out.name})")
        return out
    y = ds.y
    tr, va, te = make_split(y, ds.time, cfg["split"]["mode"], cfg["split"]["fractions"], seed)
    te_raw = ds.meta.get("_te_raw", {})
    te_cols = ds.te_cols if cfg["preprocessing"].get("target_encode", True) else []
    X = ds.X if te_cols else ds.X.drop(columns=ds.te_cols, errors="ignore")
    pp = Preprocessor(ds.log_cols, te_cols, cfg["preprocessing"]["angle_scaling"], seed=seed)
    pp.fit(X.iloc[tr].reset_index(drop=True), y[tr], {c: te_raw[c][tr] for c in te_cols})
    plan = subsample_plan(y, tr, va, te, cfg["sampling"], seed)

    def tf(rows, train_rows=False):
        Xr = X.iloc[rows].reset_index(drop=True)
        tr_info = None
        if train_rows and te_cols:  # map rows -> their position in `tr` (for OOF encodings)
            tr_info = {"pos": np.arange(len(rows)), "oof_pos": np.searchsorted(tr, rows)}
        return pp.transform(Xr, {c: te_raw[c][rows] for c in te_cols}, tr_info)

    Xc_tr, Xq_tr = tf(plan["tr_sub"], True)
    Xc_va, Xq_va = tf(plan["va_sub"])
    Xc_te, Xq_te = tf(plan["te_sub"])
    names = pp.columns
    mi = mutual_info_classif(Xc_tr, y[plan["tr_sub"]], random_state=seed)
    mi_rank = [names[i] for i in np.argsort(-mi, kind="stable")]

    arrays = dict(
        Xq_train=Xq_tr, Xq_val=Xq_va, Xq_test=Xq_te, Xc_train=Xc_tr, Xc_val=Xc_va, Xc_test=Xc_te,
        y_train=y[plan["tr_sub"]].astype(np.int8), y_val=y[plan["va_sub"]].astype(np.int8),
        y_test=y[plan["te_sub"]].astype(np.int8),
        w_val=plan["w_val"].astype(np.float64), w_test=plan["w_test"].astype(np.float64),
        rows_train=plan["tr_sub"], rows_val=plan["va_sub"], rows_test=plan["te_sub"],
        sfs_pos=plan["sfs_pos"], noisy_tr=plan["noisy_tr"], noisy_va=plan["noisy_va"],
        noisy_te=plan["noisy_te"], w_noisy_val=plan["w_noisy_val"], w_noisy_test=plan["w_noisy_test"])
    save_npz(out / "arrays.npz", compress=False, **arrays)

    def counts(rows):
        return {"rows": int(len(rows)), "positives": int(y[rows].sum()), "prevalence": float(y[rows].mean())}

    meta = {k: v for k, v in ds.meta.items() if not k.startswith("_")}
    meta.update({
        "dataset": ds.name, "seed": seed, "key": key, "fingerprint_sha256": ds.fingerprint,
        "feature_names": names, "mi_ranking": mi_rank, "mi_values": {names[i]: float(mi[i]) for i in range(len(names))},
        "split_mode": cfg["split"]["mode"], "split_fractions": cfg["split"]["fractions"],
        "partition_full": {"train": counts(tr), "val": counts(va), "test": counts(te)},
        "partition_used": {"train": counts(plan["tr_sub"]), "val": counts(plan["va_sub"]),
                           "test": counts(plan["te_sub"]), "sfs_train": int(len(plan["sfs_pos"])),
                           "noisy_train": int(len(plan["noisy_tr"])), "noisy_val": int(len(plan["noisy_va"])),
                           "noisy_test": int(len(plan["noisy_te"]))},
        "preprocessing": pp.describe(), "sampling": cfg["sampling"],
        "leakage_checks": {
            "train_val_overlap": int(len(np.intersect1d(tr, va))),
            "train_test_overlap": int(len(np.intersect1d(tr, te))),
            "val_test_overlap": int(len(np.intersect1d(va, te))),
            "transforms_fitted_on": "training partition only"},
    })
    if cfg["split"]["mode"] == "temporal" and ds.time is not None:
        meta["time_ranges"] = {k: [float(ds.time[r].min()), float(ds.time[r].max())]
                               for k, r in (("train", tr), ("val", va), ("test", te))}
    write_json(out / "meta.json", meta)
    if logger:
        pf = meta["partition_full"]
        logger.info(f"[prepare] {ds.name} seed={seed}: train {pf['train']} val {pf['val']} test {pf['test']}")
    return out


def prepare_all(cfg: dict, datasets, seeds, logger=None) -> dict:
    """Main-process only. Returns {(dataset, seed): prepared_dir}."""
    logger = logger or get_logger("qfd")
    out = {}
    for name in datasets:
        ds = None
        for seed in seeds:
            # cheap check first: the key needs the fingerprint, which needs the file
            if ds is None:
                logger.info(f"[prepare] loading {name} ...")
                ds = load_dataset(name, cfg, cfg["run"].get("sample_limit"), seed=0)
                logger.info(f"[prepare] {name}: {ds.meta['rows']} rows, {ds.meta['positives']} positives "
                            f"({100 * ds.meta['prevalence']:.3f}%), {len(ds.X.columns)} candidate features, "
                            f"schema check: {ds.meta['schema_check']['matches']}")
            out[(name, seed)] = str(prepare_one(ds, cfg, seed, logger))
        del ds
    return out


def load_prepared(path: str | Path) -> tuple[dict, dict]:
    path = Path(path)
    return load_npz(path / "arrays.npz"), read_json(path / "meta.json")
