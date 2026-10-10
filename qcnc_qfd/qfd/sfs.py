r"""Sequential Forward Selection (SFS) over quantum feature representations.

    S <- {}
    repeat:
        for each candidate f not in S (fixed dataset column order = deterministic tie-break):
            z = quantum features of x_{S + f}       (ideal simulation, <Z_i> observables)
            fit the FIXED evaluator on training rows, score validation rows
            record ROC-AUC (primary) and AP (secondary)
        add the best f (strictly greater score wins)
    until |S| = k_max (default: full path), or the evaluation budget is exhausted
          [optional early stop if the best addition does not improve the best-so-far by >= tol]
    return the best-validation subset encountered: smallest k with score >= best - tol

At most p + (p-1) + ... + (p-k+1) evaluations (p=30, k=7 -> 189).
Only training (fit) and validation (rank) rows are touched; the test set never is.
Selection uses the ideal simulation, so the selected subset is architecture-independent;
architecture/noise effects are then measured on that fixed subset (paired comparison).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .classifiers import scores as clf_scores
from .classifiers import sfs_evaluator
from .featuremaps import ideal_features
from .metrics import avg_precision, roc_auc
from .utils import read_json, stable_hash, write_json

SFS_VERSION = 2


def max_evaluations(p: int, k: int) -> int:
    return int(sum(p - i for i in range(min(k, p))))


def sfs_key(prep_key: str, fmap: dict, sfs_cfg: dict, feature_names, n_sfs_train: int) -> str:
    return stable_hash({"v": SFS_VERSION, "prep": prep_key, "fmap": fmap, "sfs": sfs_cfg,
                        "features": list(feature_names), "n": n_sfs_train})


def run_sfs(Xq_tr, ytr, Xq_va, yva, wva, feature_names, fmap: dict, sfs_cfg: dict, seed: int,
            threads: int, state_path: Path, logger=None, evaluator_kind: str | None = None):
    feature_names = list(feature_names)
    p = len(feature_names)
    kmax = min(int(sfs_cfg["max_features"]), p)
    tol = float(sfs_cfg.get("tol", 0.0))
    budget = sfs_cfg.get("budget_evals")
    kind = evaluator_kind or sfs_cfg.get("evaluator", "logreg")
    metric = sfs_cfg.get("metric", "roc_auc")

    state = {"history": [], "stages": [], "selected": [], "done": False, "version": SFS_VERSION}
    if state_path.exists():
        try:
            state = read_json(state_path)
            if logger:
                logger.info(f"[SFS] resuming from stage {len(state['stages'])} ({state_path.name})")
        except Exception:
            pass
    if state.get("done"):
        return state

    t_start = time.perf_counter()
    n_eval = len(state["history"])
    while len(state["selected"]) < kmax and not state["done"]:
        S = list(state["selected"])
        cand = [j for j in range(p) if j not in S]
        if not cand:
            state["stopping_reason"] = "exhausted"
            break
        stage = len(S) + 1
        t_stage = time.perf_counter()
        best = None
        for j in cand:
            if budget is not None and n_eval >= int(budget):
                state["stopping_reason"] = "budget"
                state["done"] = True
                break
            sub = S + [j]
            t0 = time.perf_counter()
            Ztr, _ = ideal_features(Xq_tr[:, sub], fmap)
            Zva, _ = ideal_features(Xq_va[:, sub], fmap)
            t_q = time.perf_counter() - t0
            model = sfs_evaluator(kind, seed, threads)
            if np.ptp(Ztr, axis=0).max() < 1e-9:  # degenerate representation
                auc, ap = 0.5, float((wva * yva).sum() / wva.sum())
            else:
                model.fit(Ztr, ytr)
                s = clf_scores("xgb" if kind == "xgb_small" else "mlp", model, Zva)
                auc, ap = roc_auc(yva, s, wva), avg_precision(yva, s, wva)
            n_eval += 1
            score = auc if metric == "roc_auc" else ap
            rec = {"stage": stage, "candidate": feature_names[j], "subset": [feature_names[i] for i in sub],
                   "val_roc_auc": auc, "val_avg_precision": ap, "quantum_seconds": t_q,
                   "eval_seconds": time.perf_counter() - t0}
            state["history"].append(rec)
            if best is None or score > best[0]:
                best = (score, j, auc, ap)
        if best is None:
            break
        prev_best = max([s["best_score"] for s in state["stages"]], default=-np.inf)
        state["selected"].append(best[1])
        state["stages"].append({"stage": stage, "added": feature_names[best[1]],
                                "subset": [feature_names[i] for i in state["selected"]],
                                "best_score": best[0], "val_roc_auc": best[2], "val_avg_precision": best[3],
                                "n_candidates": len(cand), "stage_seconds": time.perf_counter() - t_stage})
        if logger:
            logger.info(f"[SFS] stage {stage}: +{feature_names[best[1]]} -> val AUC {best[2]:.4f} "
                        f"AP {best[3]:.4f} ({len(cand)} candidates, {time.perf_counter() - t_stage:.1f}s)")
        if sfs_cfg.get("stop_on_no_improvement", True) and best[0] < prev_best + tol:
            state["stopping_reason"] = "no_improvement"
            state["done"] = True
        write_json(state_path, state)  # checkpoint after every stage
    if not state.get("stopping_reason"):
        state["stopping_reason"] = "max_features"
    # best subset encountered: smallest k whose score is within `tol` of the best score
    top = max(s["best_score"] for s in state["stages"])
    best_stage = min((s for s in state["stages"] if s["best_score"] >= top - tol), key=lambda s: s["stage"])
    state.update({
        "done": True, "best_subset": best_stage["subset"], "best_k": best_stage["stage"],
        "best_val_roc_auc": best_stage["val_roc_auc"], "best_val_avg_precision": best_stage["val_avg_precision"],
        "n_evaluations": n_eval, "max_possible_evaluations": max_evaluations(p, kmax),
        "evaluator": kind, "metric": metric, "tol": tol, "k_max": kmax, "p_candidates": p,
        "runtime_seconds": state.get("runtime_seconds", 0.0) + (time.perf_counter() - t_start),
    })
    write_json(state_path, state)
    return state
