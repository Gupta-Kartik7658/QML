"""Dataset loaders with schema validation.

Each loader returns a `RawDataset`: candidate input features (numeric DataFrame, fixed
column order = deterministic SFS tie-breaking), binary label, optional time column and
a metadata dict that goes straight into the paper's dataset table.

Only *stateless* transformations happen here (dropping identifiers / leakage columns,
fixed-vocabulary one-hot encoding, string cleaning). Anything that is *fitted* (imputation,
scaling, target encoding, quantile transforms) happens in preprocess.py on the training
partition only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import file_sha256


@dataclass
class RawDataset:
    name: str
    X: pd.DataFrame                 # candidate features, numeric
    y: np.ndarray                   # int8 {0,1}
    time: np.ndarray | None         # for temporal splits
    log_cols: list = field(default_factory=list)     # signed log1p before scaling
    te_cols: list = field(default_factory=list)      # target-encoded (fitted on train only)
    meta: dict = field(default_factory=dict)
    fingerprint: str = ""


class DatasetError(RuntimeError):
    pass


def _require(df: pd.DataFrame, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DatasetError(f"{name}: missing expected columns {missing}. Found: {list(df.columns)}")


def _clean_common(df: pd.DataFrame, label: str, drop_duplicates: bool, meta: dict) -> pd.DataFrame:
    meta["rows_raw"] = int(len(df))
    meta["positives_raw"] = int(df[label].sum())
    y = df[label]
    if not set(pd.unique(y)).issubset({0, 1}):
        raise DatasetError(f"label {label} is not binary: {pd.unique(y)[:10]}")
    n_missing_label = int(y.isna().sum())
    if n_missing_label:
        df = df[~y.isna()]
    meta["rows_missing_label_dropped"] = n_missing_label
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        meta["duplicate_rows_dropped"] = int(before - len(df))
    else:
        meta["duplicate_rows_dropped"] = 0
    num = df.select_dtypes(include=[np.number])
    meta["non_finite_values"] = int((~np.isfinite(num.to_numpy(dtype=float))).sum())
    return df.reset_index(drop=True)


def _finish(name, df, label, feat_cols, time_col, log_cols, te_cols, meta, fp, cfg_ds) -> RawDataset:
    X = df[feat_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y = df[label].astype(np.int8).to_numpy()
    meta.update(
        rows=int(len(df)), positives=int(y.sum()), prevalence=float(y.mean()),
        n_candidate_features=len(feat_cols), candidate_features=list(feat_cols),
        missing_values=int(X.isna().sum().sum()),
    )
    exp_r, exp_p = cfg_ds.get("expected_rows"), cfg_ds.get("expected_positives")
    meta["schema_check"] = {
        "expected_rows": exp_r, "observed_rows_raw": meta["rows_raw"],
        "expected_positives": exp_p, "observed_positives_raw": meta["positives_raw"],
        "matches": (exp_r is None or exp_r == meta["rows_raw"]) and (exp_p is None or exp_p == meta["positives_raw"]),
    }
    t = df[time_col].astype(float).to_numpy() if time_col else None
    return RawDataset(name, X.reset_index(drop=True), y, t, log_cols, te_cols, meta, fp)


# ----------------------------------------------------------------------------- creditcard
def load_creditcard(path: Path, cfg_ds: dict, fp_cache: Path | None = None) -> RawDataset:
    fp = file_sha256(path, fp_cache)
    df = pd.read_csv(path)
    _require(df, ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)], "creditcard")
    meta = {"source": "Kaggle mlg-ulb/creditcardfraud (Dal Pozzolo et al.), real anonymised transactions",
            "data_type": "real", "label": "Class", "dropped_columns": {}}
    df = _clean_common(df, "Class", cfg_ds.get("drop_duplicates", True), meta)
    feats = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    meta["notes"] = ("V1-V28 are PCA components supplied by the data owner; no further PCA applied. "
                     "Time = seconds since first transaction (2 days).")
    return _finish("creditcard", df, "Class", feats, "Time", ["Amount"], [], meta, fp, cfg_ds)


# ----------------------------------------------------------------------------- PaySim
PAYSIM_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_paysim(path: Path, cfg_ds: dict, fp_cache: Path | None = None) -> RawDataset:
    fp = file_sha256(path, fp_cache)
    dtypes = {"step": np.int32, "type": "category", "amount": np.float64, "nameOrig": str,
              "oldbalanceOrg": np.float64, "newbalanceOrig": np.float64, "nameDest": str,
              "oldbalanceDest": np.float64, "newbalanceDest": np.float64,
              "isFraud": np.int8, "isFlaggedFraud": np.int8}
    df = pd.read_csv(path, dtype=dtypes)
    _require(df, list(dtypes), "paysim")
    meta = {"source": "Kaggle ealaxi/paysim1 (Lopez-Rojas et al., PaySim mobile-money simulator)",
            "data_type": "synthetic (agent-based simulation calibrated on real mobile-money logs)",
            "label": "isFraud",
            "dropped_columns": {
                "nameOrig": "account identifier", "nameDest": "account identifier (only M/C prefix kept)",
                "isFlaggedFraud": "output of the simulator's own rule-based flag -> leakage-prone, not an input"}}
    df = _clean_common(df, "isFraud", cfg_ds.get("drop_duplicates", True), meta)  # before dropping ids
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(np.int8)
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])
    unknown = set(df["type"].astype(str).unique()) - set(PAYSIM_TYPES)
    if unknown:
        raise DatasetError(f"paysim: unexpected transaction types {unknown}")
    for t in PAYSIM_TYPES:  # fixed vocabulary -> no fitting needed
        df[f"type_{t}"] = (df["type"].astype(str) == t).astype(np.int8)
    df = df.drop(columns=["type"])
    money = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    feats = ["step"] + money + ["dest_is_merchant"] + [f"type_{t}" for t in PAYSIM_TYPES]
    if cfg_ds.get("engineered_balance_errors", False):
        df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
        df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
        feats += ["errorBalanceOrig", "errorBalanceDest"]
        money += ["errorBalanceOrig", "errorBalanceDest"]
    meta["fraud_by_type"] = {t: int(df.loc[df[f"type_{t}"] == 1, "isFraud"].sum()) for t in PAYSIM_TYPES}
    meta["notes"] = "step = simulated hour (1..743). Monetary columns are signed-log transformed."
    return _finish("paysim", df, "isFraud", feats, "step", money, [], meta, fp, cfg_ds)


# ----------------------------------------------------------------------------- BankSim
BANKSIM_GENDER = ["F", "M", "E", "U"]


def load_banksim(path: Path, cfg_ds: dict, fp_cache: Path | None = None) -> RawDataset:
    fp = file_sha256(path, fp_cache)
    df = pd.read_csv(path, dtype=str)
    _require(df, ["step", "customer", "age", "gender", "zipcodeOri", "merchant", "zipMerchant",
                  "category", "amount", "fraud"], "banksim")
    for c in df.columns:  # values are stored as 'quoted' strings
        df[c] = df[c].str.strip().str.strip("'")
    df["step"] = df["step"].astype(np.int32)
    df["amount"] = df["amount"].astype(float)
    df["fraud"] = df["fraud"].astype(np.int8)
    meta = {"source": "Kaggle ealaxi/banksim1 (Lopez-Rojas & Axelsson, BankSim payment simulator)",
            "data_type": "synthetic (agent-based simulation calibrated on aggregated real bank data)",
            "label": "fraud", "dropped_columns": {"customer": "customer identifier"}}
    df = _clean_common(df, "fraud", cfg_ds.get("drop_duplicates", True), meta)  # before dropping ids
    for z in ("zipcodeOri", "zipMerchant"):
        if df[z].nunique() == 1:
            meta["dropped_columns"][z] = f"constant ({df[z].iloc[0]})"
            df = df.drop(columns=[z])
    df = df.drop(columns=["customer"])
    age = pd.to_numeric(df["age"], errors="coerce")  # 'U' (unknown) -> NaN, imputed on train
    df["age_group"] = age
    df["age_unknown"] = age.isna().astype(np.int8)
    for g in BANKSIM_GENDER:
        df[f"gender_{g}"] = (df["gender"] == g).astype(np.int8)
    cats = sorted(df["category"].unique())  # schema vocabulary (label-free)
    for c in cats:
        df[f"cat_{c.replace('es_', '')}"] = (df["category"] == c).astype(np.int8)
    df["merchant_te"] = df["merchant"]  # replaced by train-only out-of-fold target encoding
    meta["n_merchants"] = int(df["merchant"].nunique())
    meta["categories"] = cats
    feats = (["step", "amount", "age_group", "age_unknown"] + [f"gender_{g}" for g in BANKSIM_GENDER]
             + [f"cat_{c.replace('es_', '')}" for c in cats] + ["merchant_te"])
    df = df.drop(columns=["age", "gender", "category"])
    meta["notes"] = ("step = simulated day. merchant id -> out-of-fold smoothed target encoding fitted on "
                     "training rows only; categories one-hot.")
    X_df = df[feats].copy()
    merchant = X_df.pop("merchant_te")
    X_df["merchant_te"] = 0.0
    ds = _finish("banksim", pd.concat([X_df, df[["fraud"]]], axis=1), "fraud", feats, "step",
                 ["amount"], ["merchant_te"], meta, fp, cfg_ds)
    ds.meta["_te_raw"] = {"merchant_te": merchant.to_numpy()}  # categorical codes for fitting later
    return ds


LOADERS = {"creditcard": load_creditcard, "paysim": load_paysim, "banksim": load_banksim}


def load_dataset(name: str, cfg: dict, sample_limit: int | None = None, seed: int = 0) -> RawDataset:
    cfg_ds = cfg["datasets"][name]
    path = Path(cfg["paths"]["data_dir"]) / cfg_ds["file"]
    if not path.exists():
        raise DatasetError(
            f"{name}: file not found at {path}. Download it (see README.md, 'Data') and place it there.")
    fp_cache = Path(cfg["paths"]["cache_dir"]) / "fingerprints.json"
    ds = LOADERS[name](path, cfg_ds, fp_cache)
    if sample_limit and sample_limit < len(ds.y):
        ds = _stratified_limit(ds, sample_limit, seed)
    return ds


def _stratified_limit(ds: RawDataset, n: int, seed: int) -> RawDataset:
    """PILOT ONLY: keep all positives (up to n/2) + random negatives."""
    rng = np.random.default_rng(10_000 + seed)
    pos = np.flatnonzero(ds.y == 1)
    neg = np.flatnonzero(ds.y == 0)
    pos = rng.choice(pos, min(len(pos), n // 2), replace=False)
    neg = rng.choice(neg, min(len(neg), n - len(pos)), replace=False)
    keep = np.sort(np.concatenate([pos, neg]))
    ds.X = ds.X.iloc[keep].reset_index(drop=True)
    ds.y = ds.y[keep]
    ds.time = ds.time[keep] if ds.time is not None else None
    if "_te_raw" in ds.meta:
        ds.meta["_te_raw"] = {k: v[keep] for k, v in ds.meta["_te_raw"].items()}
    ds.meta["PILOT_SAMPLE_LIMIT"] = n
    ds.meta["rows"] = int(len(keep))
    ds.meta["positives"] = int(ds.y.sum())
    ds.meta["prevalence"] = float(ds.y.mean())
    return ds
