"""Evaluation metrics for imbalanced fraud detection (sample-weight aware).

Weights: validation/test genuine rows may be case-control subsampled; their weight is
N_genuine_full / N_genuine_kept, so every metric below is an (unbiased) estimate of the metric
on the full partition at its ORIGINAL prevalence. Fraud rows always have weight 1.

Definitions (fraud = positive class):
  recall = TP/(TP+FN), precision = TP/(TP+FP), specificity = TN/(TN+FP), FPR = FP/(FP+TN)
  F1, balanced accuracy, MCC as usual. ROC-AUC and average precision use continuous scores.
The decision threshold maximises F1 on the VALIDATION set and is then applied unchanged to test.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)


def _w(y, w):
    return np.ones(len(y)) if w is None else np.asarray(w, dtype=float)


def roc_auc(y, s, w=None):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, sample_weight=_w(y, w)))


def avg_precision(y, s, w=None):
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s, sample_weight=_w(y, w)))


def choose_threshold(y, s, w=None, max_candidates: int = 2000) -> float:
    """Threshold maximising (weighted) F1 on validation data."""
    w = _w(y, w)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
    tp = np.cumsum(w_sorted * (y_sorted == 1))
    fp = np.cumsum(w_sorted * (y_sorted == 0))
    P = tp[-1]
    # only evaluate at the last index of each distinct score
    last = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s_sorted) - 1]
    if len(last) > max_candidates:
        last = last[np.linspace(0, len(last) - 1, max_candidates).astype(int)]
    prec = tp[last] / np.maximum(tp[last] + fp[last], 1e-12)
    rec = tp[last] / max(P, 1e-12)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return float(s_sorted[last[i]])  # predict fraud if score >= threshold


def threshold_metrics(y, s, thr, w=None) -> dict:
    w = _w(y, w)
    yp = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1], sample_weight=w).ravel()
    rtn, rfp, rfn, rtp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    spec = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {
        "threshold": float(thr), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "specificity": float(spec), "fpr": float(1 - spec),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp, sample_weight=w)),
        "mcc": float(matthews_corrcoef(y, yp, sample_weight=w)),
        "tp_w": float(tp), "fp_w": float(fp), "tn_w": float(tn), "fn_w": float(fn),
        "tp": int(rtp), "fp": int(rfp), "tn": int(rtn), "fn": int(rfn),
    }


def full_metrics(y, s, thr, w=None) -> dict:
    out = {"roc_auc": roc_auc(y, s, w), "avg_precision": avg_precision(y, s, w)}
    out.update(threshold_metrics(y, s, thr, w))
    w_ = _w(y, w)
    out["n"] = int(len(y))
    out["n_pos"] = int(y.sum())
    out["prevalence_est"] = float((w_ * y).sum() / w_.sum())
    return out


# ----------------------------------------------------------------------------- fast bootstrap
def _auc_ap_sorted(y, w, s_sorted_desc_groups):
    """Weighted ROC-AUC and AP for scores pre-sorted descending (ties grouped)."""
    starts = s_sorted_desc_groups
    wp = w * (y == 1)
    wn = w * (y == 0)
    gp = np.add.reduceat(wp, starts)
    gn = np.add.reduceat(wn, starts)
    P, N = gp.sum(), gn.sum()
    if P <= 0 or N <= 0:
        return np.nan, np.nan
    cn_before = np.cumsum(gn) - gn  # negatives with strictly higher score
    # AUC = P(score_pos > score_neg) + 0.5 P(tie)
    auc = ((gp * (N - cn_before - gn)).sum() + 0.5 * (gp * gn).sum()) / (P * N)
    tp = np.cumsum(gp)
    fp = np.cumsum(gn)
    prec = tp / np.maximum(tp + fp, 1e-300)
    ap = (gp * prec).sum() / P  # step-wise AP (= sklearn definition)
    return auc, ap


def bootstrap_ci(y, s, w=None, n_boot=200, seed=0, alpha=0.05) -> dict:
    """Poisson bootstrap (O(n) per replicate after one sort) 95% CIs for ROC-AUC and AP."""
    if n_boot <= 0:
        return {}
    w = _w(y, w)
    order = np.argsort(-s, kind="mergesort")
    y, w, s = y[order], w[order], s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    rng = np.random.default_rng(seed)
    aucs, aps = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        wb = w * rng.poisson(1.0, len(w))
        aucs[b], aps[b] = _auc_ap_sorted(y, wb, starts)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"roc_auc_ci_low": float(np.nanpercentile(aucs, lo)), "roc_auc_ci_high": float(np.nanpercentile(aucs, hi)),
            "avg_precision_ci_low": float(np.nanpercentile(aps, lo)),
            "avg_precision_ci_high": float(np.nanpercentile(aps, hi)), "n_bootstrap": int(n_boot)}


def paired_bootstrap_diff(y, s_a, s_b, w=None, n_boot=1000, seed=0, alpha=0.05) -> dict:
    """Paired Poisson bootstrap of (B - A) for ROC-AUC and AP on the SAME test rows and resamples.

    Returns point differences, percentile CIs and a one-sided bootstrap p-value P(diff <= 0)."""
    y = np.asarray(y)
    w = _w(y, w)
    prep = []
    for s in (np.asarray(s_a, float), np.asarray(s_b, float)):
        o = np.argsort(-s, kind="mergesort")
        prep.append((o, np.r_[0, np.flatnonzero(np.diff(s[o]) != 0) + 1]))
    full = [_auc_ap_sorted(y[o], w[o], st) for o, st in prep]
    rng = np.random.default_rng(seed)
    d_auc, d_ap = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        wb = w * rng.poisson(1.0, len(w))
        (a1, p1), (a2, p2) = (_auc_ap_sorted(y[o], wb[o], st) for o, st in prep)
        d_auc[b], d_ap[b] = a2 - a1, p2 - p1
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"auc_a": float(full[0][0]), "auc_b": float(full[1][0]), "ap_a": float(full[0][1]), "ap_b": float(full[1][1]),
            "d_auc": float(full[1][0] - full[0][0]), "d_auc_ci_low": float(np.nanpercentile(d_auc, lo)),
            "d_auc_ci_high": float(np.nanpercentile(d_auc, hi)), "d_auc_p_le0": float(np.nanmean(d_auc <= 0)),
            "d_ap": float(full[1][1] - full[0][1]), "d_ap_ci_low": float(np.nanpercentile(d_ap, lo)),
            "d_ap_ci_high": float(np.nanpercentile(d_ap, hi)), "d_ap_p_le0": float(np.nanmean(d_ap <= 0)),
            "n_bootstrap_paired": int(n_boot)}


def curve_points(y, s, w=None, max_points=400) -> dict:
    """Down-sampled ROC and PR curves (for later overlay plots without the raw scores)."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    w = _w(y, w)
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
    pr, rc, _ = precision_recall_curve(y, s, sample_weight=w)

    def ds(*a):
        n = len(a[0])
        if n <= max_points:
            return a
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
        return tuple(x[idx] for x in a)

    fpr, tpr = ds(fpr, tpr)
    pr, rc = ds(pr, rc)
    return {"roc_fpr": fpr.astype(np.float32), "roc_tpr": tpr.astype(np.float32),
            "pr_precision": pr.astype(np.float32), "pr_recall": rc.astype(np.float32)}
