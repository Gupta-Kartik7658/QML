"""Downstream classical classifiers: MLP, XGBoost, RBF-SVM (+ the fixed SFS evaluator).

All three receive exactly the same representation, training rows, validation rows and
the same-size hyper-parameter grid (6-8 configurations). Selection: validation ROC-AUC
(weighted), ties -> validation AP, then grid order. No refit on validation data.

The classical SVM here consumes an explicit numerical feature vector. It is NOT the
fidelity-kernel QSVM (see run_qsvm_reference.py for that).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .metrics import avg_precision, roc_auc


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def make_model(name: str, hp: dict, fixed: dict, seed: int, threads: int, ytr: np.ndarray):
    if name == "mlp":
        p = dict(fixed)
        p.update(hp)
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=seed, **p))
    if name == "xgb":
        p = dict(fixed)
        p.update(hp)
        if p.get("scale_pos_weight") == "balanced":
            p["scale_pos_weight"] = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
        return XGBClassifier(random_state=seed, n_jobs=threads, verbosity=0, **p)
    if name == "svm":
        p = dict(fixed)
        p.update(hp)
        return make_pipeline(StandardScaler(), SVC(random_state=seed, **p))
    raise ValueError(name)


def fit_model(name, model, Xtr, ytr, Xva, yva, wva):
    if name == "xgb":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)
    else:
        model.fit(Xtr, ytr)
    return model


def scores(name, model, X):
    if name == "svm":
        return model.decision_function(X).astype(np.float64)
    return model.predict_proba(X)[:, 1].astype(np.float64)


def iteration_log(name, model) -> dict:
    """Per-iteration training traces for the paper/appendix."""
    if name == "mlp":
        m = model[-1]
        out = {"loss_curve": list(map(float, m.loss_curve_)), "n_iter": int(m.n_iter_),
               "converged": bool(m.n_iter_ < m.max_iter)}
        if getattr(m, "validation_scores_", None) is not None:
            out["internal_val_accuracy"] = list(map(float, m.validation_scores_))
        return out
    if name == "xgb":
        ev = model.evals_result()
        key = next(iter(ev)) if ev else None
        return {"eval_metric_curve": ev[key] if key else {}, "best_iteration": int(getattr(model, "best_iteration", -1))}
    if name == "svm":
        m = model[-1]
        return {"n_support": list(map(int, m.n_support_)), "fit_status": int(m.fit_status_)}
    return {}


def tune_and_fit(name, cfg_clf: dict, Xtr, ytr, Xva, yva, wva, seed, threads, hp_metric="roc_auc"):
    """Grid search on the validation set. Returns (best_model, best_hp, search_rows)."""
    rows, best = [], None
    for i, hp in enumerate(expand_grid(cfg_clf["grid"])):
        t0 = time.perf_counter()
        model = make_model(name, hp, cfg_clf.get("fixed", {}), seed, threads, ytr)
        fit_model(name, model, Xtr, ytr, Xva, yva, wva)
        t_fit = time.perf_counter() - t0
        s = scores(name, model, Xva)
        auc, ap = roc_auc(yva, s, wva), avg_precision(yva, s, wva)
        rows.append({"config_id": i, **{f"hp_{k}": str(v) for k, v in hp.items()},
                     "val_roc_auc": auc, "val_avg_precision": ap, "fit_seconds": t_fit})
        key = (auc if hp_metric == "roc_auc" else ap, ap if hp_metric == "roc_auc" else auc)
        key = tuple(-1.0 if not np.isfinite(k) else k for k in key)
        if best is None or key > best[0]:
            best = (key, model, hp, i)
    return best[1], best[2], best[3], rows


# ----------------------------------------------------------------------------- SFS evaluator
def sfs_evaluator(kind: str, seed: int, threads: int):
    """Fixed, predefined evaluator used to rank SFS candidates (never tuned during SFS)."""
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                                  max_iter=2000, random_state=seed))
    if kind == "xgb_small":
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                             random_state=seed, n_jobs=threads, verbosity=0, tree_method="hist")
    raise ValueError(kind)
