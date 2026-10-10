"""Experiment orchestration shared by the six run_<classifier>_<architecture>.py scripts.

One script = one classifier x one architecture; it loops over
    datasets x feature maps x seeds  (+ a classical-all baseline task per dataset/seed).
Each (dataset, feature map, seed) is one *task*, executed in a worker process
(ProcessPoolExecutor, 'spawn' context, `workers` processes, `threads_per_worker` threads each).

Inside a task:
  1. SFS (ideal simulation; cached in cache/sfs, shared by all scripts on the machine)
  2. quantum features of the selected subset (cached in cache/features)
  3. representations:
       q_ideal_z          <Z_i> features, ideal simulation               [primary]
       q_ideal_zzz        <Z_i> + <Z_i Z_j>, ideal                       [observable ablation]
       q_ideal_mi_z       <Z_i> of the predefined top-k mutual-information subset [SFS ablation]
       c_selected         the SFS-selected raw features, no quantum encoding     [baseline]
       q_noisy_z          <Z_i> from the architecture's noisy simulation (exact expectation) [paired rows]
       q_ideal_paired_z   ideal <Z_i> on exactly the same rows as q_noisy_z                [paired]
       q_noisy_shots_z    noisy + finite-shot sampling (noise.shots, default 1024)         [paired]
       q_ideal_shots_z    ideal + finite-shot sampling                                      [paired]
       -> 2x2 decomposition: hardware-noise effect vs shot-noise effect
     classical task:
       c_all              all candidate features, no quantum encoding           [baseline]
  4. for each representation: grid search on validation -> threshold on validation ->
     test metrics + bootstrap CIs -> save model, scores, curves, logs, plots.
Everything is written incrementally and atomically; rerunning the same command resumes.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .cache import FeatureCache, input_hash
from .config import apply_cli, build_parser, load_config
from .featuremaps import fmap_params, ideal_features
from .metrics import bootstrap_ci, choose_threshold, curve_points, full_metrics
from .preprocess import load_prepared, prepare_all
from .utils import (Timer, atomic_write_text, close_logger, environment_info, get_logger,
                    limit_threadpools, peak_memory_mb, read_json, save_npz, stable_hash,
                    write_csv_df, write_json)

TASK_VERSION = 3
REP_INFO = {
    "q_ideal_z": ("quantum", "SFS", "ideal", "<Z_i>"),
    "q_ideal_zzz": ("quantum", "SFS", "ideal", "<Z_i>,<Z_iZ_j>"),
    "q_ideal_mi_z": ("quantum", "MI-top-k", "ideal", "<Z_i>"),
    "c_selected": ("classical", "SFS", "none", "raw selected"),
    "q_noisy_z": ("quantum", "SFS", "noisy", "<Z_i> exact"),
    "q_ideal_paired_z": ("quantum", "SFS", "ideal", "<Z_i> exact (paired rows)"),
    "q_noisy_shots_z": ("quantum", "SFS", "noisy", "<Z_i> finite shots"),
    "q_ideal_shots_z": ("quantum", "SFS", "ideal", "<Z_i> finite shots (paired rows)"),
    "c_all": ("classical", "none", "none", "raw all"),
}


# ============================================================================ feature helpers
def _ideal_cached(cache: FeatureCache, prep_key, X: dict, cols, names, fmap, extended=True):
    """Ideal features for train/val/test of a column subset; cached as one entry."""
    sub = [names[i] for i in cols]
    key = cache.key(kind="ideal", prep=prep_key, subset=sub, fmap=fmap, extended=extended,
                    inputs={p: input_hash(X[p][:, cols]) for p in X})

    def compute():
        out = {}
        t0 = time.perf_counter()
        for part, A in X.items():
            out[part], _ = ideal_features(A[:, cols], fmap, extended=extended)
        out["seconds"] = np.array([time.perf_counter() - t0])
        return out

    expect = {p: (len(X[p]), len(cols) + (len(cols) * (len(cols) - 1) // 2 if extended else 0)) for p in X}
    return cache.get_or_compute(key, compute, expect)


def _noisy_cached(cache, prep_key, X: dict, cols, names, fmap, arch, arch_cfg, noise_cfg, allow_compute=True):
    """Noisy computational-basis probabilities (exact density-matrix + exact readout) per partition.
    allow_compute=False -> return the cached entry or None."""
    from .noisy import noisy_probabilities
    sub = [names[i] for i in cols]
    ncfg = {k: v for k, v in noise_cfg.items() if k not in ("shots", "enabled", "batch_size")}
    key = cache.key(kind="noisy_probs", prep=prep_key, subset=sub, fmap=fmap, arch=arch, arch_cfg=arch_cfg,
                    noise=ncfg, inputs={p: input_hash(X[p][:, cols]) for p in X})

    def compute():
        out = {}
        t0 = time.perf_counter()
        stats = None
        for part, A in X.items():
            P, stats = noisy_probabilities(A[:, cols], fmap, arch, arch_cfg, noise_cfg)
            out[part] = P.astype(np.float64)
        out["seconds"] = np.array([time.perf_counter() - t0])
        out["stats_json"] = np.array(json.dumps(stats))
        return out

    if not allow_compute:
        return cache.get(key)
    return cache.get_or_compute(key, compute)


def get_sfs(cfg: dict, A: dict, meta: dict, dataset: str, fmap_name: str, seed: int, logger=None) -> dict:
    """SFS result for (dataset, feature map, seed) from the shared cache (computed if missing)."""
    from .sfs import run_sfs, sfs_key
    fm = fmap_params(cfg, fmap_name)
    sp = A["sfs_pos"]
    skey = sfs_key(meta["key"], fm, cfg["sfs"], meta["feature_names"], len(sp))
    spath = Path(cfg["paths"]["cache_dir"]) / "sfs" / dataset / f"{fmap_name}_seed{seed}_{skey}.json"
    spath.parent.mkdir(parents=True, exist_ok=True)
    return run_sfs(A["Xq_train"][sp], A["y_train"][sp], A["Xq_val"], A["y_val"], A["w_val"], meta["feature_names"],
                   fm, cfg["sfs"], seed, int(cfg["run"]["threads_per_worker"]), spath, logger)


# ============================================================================ one representation
def _train_eval_rep(rep, data, clf_name, cfg, seed, threads, rep_dir: Path, title, logger, save_preds, plots):
    from .classifiers import iteration_log, scores, tune_and_fit
    from .plotting import diagnostics_figure
    import joblib

    Xtr, ytr, Xva, yva, wva, Xte, yte, wte = (data[k] for k in ("Xtr", "ytr", "Xva", "yva", "wva", "Xte", "yte", "wte"))
    t0 = time.perf_counter()
    model, hp, hp_idx, rows = tune_and_fit(clf_name, cfg["classifiers"][clf_name], Xtr, ytr, Xva, yva, wva,
                                          seed, threads, cfg["evaluation"]["hp_metric"])
    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    s_va = scores(clf_name, model, Xva)
    s_te = scores(clf_name, model, Xte)
    t_pred = time.perf_counter() - t1
    thr = choose_threshold(yva, s_va, wva)
    m_te = full_metrics(yte, s_te, thr, wte)
    m_va = full_metrics(yva, s_va, thr, wva)
    ci = bootstrap_ci(yte, s_te, wte, int(cfg["evaluation"].get("n_bootstrap", 0)), seed=seed)
    cte, cva = curve_points(yte, s_te, wte), curve_points(yva, s_va, wva)

    rep_dir.mkdir(parents=True, exist_ok=True)
    write_csv_df(rep_dir / "hp_search.csv", pd.DataFrame(rows))
    write_json(rep_dir / "iterations.json", iteration_log(clf_name, model))
    save_npz(rep_dir / "curves.npz", **{f"test_{k}": v for k, v in cte.items()}, **{f"val_{k}": v for k, v in cva.items()})
    if save_preds:
        save_npz(rep_dir / "predictions.npz", y_test=yte, score_test=s_te.astype(np.float32), w_test=wte,
                 y_val=yva, score_val=s_va.astype(np.float32), w_val=wva, threshold=np.array([thr]))
    try:
        joblib.dump(model, rep_dir / "model.joblib", compress=3)
    except Exception as e:  # never fail a run because a model could not be pickled
        logger.warning(f"could not save model: {e}")
    if plots:
        try:
            diagnostics_figure(rep_dir / "diagnostics.png", title, yte, s_te, wte, yva, s_va, wva, thr, m_te, cte, cva)
        except Exception as e:
            logger.warning(f"plot failed: {e}")
    out = {"best_hp": json.dumps(hp), "best_hp_index": hp_idx, "n_hp_configs": len(rows),
           "train_seconds": t_train, "predict_seconds": t_pred, "threshold": thr,
           "repr_dim": int(Xtr.shape[1]), "n_train": int(len(ytr)), "n_train_pos": int(ytr.sum()),
           "n_val": int(len(yva)), "n_val_pos": int(yva.sum()), "n_test": int(len(yte)), "n_test_pos": int(yte.sum())}
    out.update({f"test_{k}": v for k, v in m_te.items()})
    out.update({f"val_{k}": v for k, v in m_va.items()})
    out.update({f"test_{k}": v for k, v in ci.items()})
    write_json(rep_dir / "metrics.json", out)
    logger.info(f"   {rep:18s} {clf_name}: test AUC {m_te['roc_auc']:.4f} AP {m_te['avg_precision']:.4f} "
                f"F1 {m_te['f1']:.3f} recall {m_te['recall']:.3f} FPR {m_te['fpr']:.4f} | val AUC {m_va['roc_auc']:.4f} "
                f"| hp {hp} | {t_train:.1f}s")
    return out


# ============================================================================ one task (worker)
def run_task(task: dict, cfg: dict, clf_name: str, arch: str, out_dir: str) -> list[dict]:
    threads = int(cfg["run"]["threads_per_worker"])
    limit_threadpools(threads)
    out_dir = Path(out_dir)
    tdir = out_dir / "tasks" / task["task_id"]
    tdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"task.{task['task_id']}.{os.getpid()}", tdir / "task.log", cfg["run"]["log_level"])
    timer = Timer()
    t_task = time.perf_counter()
    base = {"run_tag": cfg["run"]["run_tag"], "script": f"{clf_name}_{arch}", "classifier": clf_name,
            "arch_script": arch, "dataset": task["dataset"], "feature_map": task["fmap"], "seed": task["seed"],
            "task_id": task["task_id"], "pilot": bool(cfg.get("_pilot", False))}
    rows = []
    try:
        A, meta = load_prepared(task["prep_dir"])
        names = meta["feature_names"]
        prep_key = meta["key"]
        base.update({"dataset_fingerprint": meta["fingerprint_sha256"][:16], "prep_key": prep_key,
                     "split_mode": meta["split_mode"], "test_prevalence_full": meta["partition_full"]["test"]["prevalence"],
                     "test_positives_full": meta["partition_full"]["test"]["positives"]})
        cache = FeatureCache(cfg["paths"]["cache_dir"], logger)
        ytr, yva, yte = A["y_train"], A["y_val"], A["y_test"]
        wva, wte = A["w_val"], A["w_test"]
        reps: dict[str, dict] = {}
        extra: dict[str, dict] = {}
        seed = task["seed"]
        logger.info(f"=== task {task['task_id']} | {clf_name} | {arch} | train {len(ytr)} ({ytr.sum()} fraud) "
                    f"val {len(yva)} ({yva.sum()}) test {len(yte)} ({yte.sum()})")

        if task["fmap"] == "classical":
            reps["c_all"] = dict(Xtr=A["Xc_train"], ytr=ytr, Xva=A["Xc_val"], yva=yva, wva=wva,
                                 Xte=A["Xc_test"], yte=yte, wte=wte)
            extra["c_all"] = {"selected_features": ";".join(names), "n_selected": len(names)}
        else:
            from .plotting import sfs_figure
            fm = fmap_params(cfg, task["fmap"])
            base["fmap_params"] = json.dumps(fm)
            # ---- 1. SFS (shared across scripts on this machine)
            with timer("sfs"):
                state = get_sfs(cfg, A, meta, task["dataset"], task["fmap"], seed, logger)
            write_json(tdir / "sfs_summary.json", {k: v for k, v in state.items() if k != "history"})
            write_csv_df(tdir / "sfs_history.csv", pd.DataFrame(state["history"]))
            if cfg["run"]["task_plots"]:
                sfs_figure(tdir / "sfs_curve.png", state, f"SFS {task['dataset']} {task['fmap']} seed {seed}")
            sel = state["best_subset"]
            cols = [names.index(f) for f in sel]
            k = len(cols)
            base.update({"sfs_runtime_seconds": state["runtime_seconds"], "sfs_evaluations": state["n_evaluations"],
                         "sfs_max_evaluations": state["max_possible_evaluations"], "sfs_stopping": state["stopping_reason"],
                         "sfs_best_val_roc_auc": state["best_val_roc_auc"], "sfs_evaluator": state["evaluator"]})
            sel_info = {"selected_features": ";".join(sel), "n_selected": k}
            X3 = {"train": A["Xq_train"], "val": A["Xq_val"], "test": A["Xq_test"]}
            # ---- 2. ideal quantum features
            with timer("ideal_features"):
                Fi = _ideal_cached(cache, prep_key, X3, cols, names, fm, extended=True)
            t_ideal = float(Fi["seconds"][0])
            zc = slice(0, k)
            reps["q_ideal_z"] = dict(Xtr=Fi["train"][:, zc], ytr=ytr, Xva=Fi["val"][:, zc], yva=yva, wva=wva,
                                     Xte=Fi["test"][:, zc], yte=yte, wte=wte)
            extra["q_ideal_z"] = {**sel_info, "quantum_extract_seconds": t_ideal}
            if cfg["observables"].get("extended", True) and k > 1:
                reps["q_ideal_zzz"] = dict(Xtr=Fi["train"], ytr=ytr, Xva=Fi["val"], yva=yva, wva=wva,
                                           Xte=Fi["test"], yte=yte, wte=wte)
                extra["q_ideal_zzz"] = {**sel_info, "quantum_extract_seconds": t_ideal}
            if cfg["baselines"].get("classical_selected", True):
                reps["c_selected"] = dict(Xtr=A["Xc_train"][:, cols], ytr=ytr, Xva=A["Xc_val"][:, cols], yva=yva,
                                          wva=wva, Xte=A["Xc_test"][:, cols], yte=yte, wte=wte)
                extra["c_selected"] = dict(sel_info)
            if cfg["baselines"].get("mi_fixed_subset", True):
                mi_sel = meta["mi_ranking"][:k]
                mcols = [names.index(f) for f in mi_sel]
                with timer("ideal_features_mi"):
                    Fm = _ideal_cached(cache, prep_key, X3, mcols, names, fm, extended=True)
                reps["q_ideal_mi_z"] = dict(Xtr=Fm["train"][:, zc], ytr=ytr, Xva=Fm["val"][:, zc], yva=yva, wva=wva,
                                            Xte=Fm["test"][:, zc], yte=yte, wte=wte)
                extra["q_ideal_mi_z"] = {"selected_features": ";".join(mi_sel), "n_selected": k,
                                         "quantum_extract_seconds": float(Fm["seconds"][0])}
            # ---- 3. architecture: compiled-circuit statistics + noisy features (paired subsets)
            from .noisy import compile_for_arch
            _, _, cstats = compile_for_arch(fm, k, arch, cfg["architectures"][arch], cfg["noise"])
            write_json(tdir / f"circuit_{arch}.json", cstats)
            base.update({"compiled_depth": cstats["compiled"]["depth"], "compiled_2q": cstats["compiled"]["two_qubit_gates"],
                         "logical_depth": cstats["logical_u_cx"]["depth"], "logical_2q": cstats["logical_u_cx"]["two_qubit_gates"],
                         "swap_overhead_2q": cstats["swap_overhead_2q"]})
            if cfg["noise"].get("enabled", True):
                from .featuremaps import ideal_probabilities, observable_matrices
                from .noisy import sample_shots
                ntr, nva, nte = A["noisy_tr"], A["noisy_va"], A["noisy_te"]
                Xn = {"train": A["Xq_train"][ntr], "val": A["Xq_val"][nva], "test": A["Xq_test"][nte]}
                logger.info(f"   noisy {arch} simulation on {sum(len(v) for v in Xn.values())} samples (k={k}) ...")
                with timer("noisy_features"):
                    Pn = _noisy_cached(cache, prep_key, Xn, cols, names, fm, arch, cfg["architectures"][arch], cfg["noise"])
                Pi = {p: ideal_probabilities(Xn[p][:, cols], fm) for p in Xn}  # paired ideal (cheap)
                Mz, _ = observable_matrices(k, False)
                wnv, wnt = A["w_noisy_val"], A["w_noisy_test"]
                shots = cfg["noise"].get("shots")
                nsel = {**sel_info, "noise_config": json.dumps({"arch": arch, **cfg["architectures"][arch]})}

                def add(rep, P, shots_=None, sseed=0, info=None):
                    F = {}
                    for i, part in enumerate(("train", "val", "test")):
                        Pp = P[part] if not shots_ else sample_shots(P[part], shots_, 1000 * seed + 10 * sseed + i)
                        F[part] = (Pp @ Mz).astype(np.float32)
                    reps[rep] = dict(Xtr=F["train"], ytr=ytr[ntr], Xva=F["val"], yva=yva[nva], wva=wnv,
                                     Xte=F["test"], yte=yte[nte], wte=wnt)
                    extra[rep] = {**(info or sel_info), "shots": shots_ or 0}

                add("q_noisy_z", Pn, info={**nsel, "quantum_extract_seconds": float(Pn["seconds"][0])})
                add("q_ideal_paired_z", Pi)
                if shots:
                    add("q_noisy_shots_z", Pn, int(shots), 1, info={**nsel, "quantum_extract_seconds": float(Pn["seconds"][0])})
                    add("q_ideal_shots_z", Pi, int(shots), 2)
                # noise-induced feature distortion (diagnostic for the paper)
                base["noisy_feature_mae"] = float(np.abs((Pn["test"] - Pi["test"]) @ Mz).mean())
                base["noisy_tvd_mean"] = float(0.5 * np.abs(Pn["test"] - Pi["test"]).sum(1).mean())

        # ---- 4. train / evaluate every representation
        if cfg["run"].get("arch_only"):
            reps = {r: v for r, v in reps.items() if r in ("q_noisy_z", "q_noisy_shots_z")}
        for rep, data in reps.items():
            rep_hash = stable_hash({"v": TASK_VERSION, "task": task, "rep": rep, "clf": cfg["classifiers"][clf_name],
                                    "eval": cfg["evaluation"], "arch": cfg["architectures"][arch] if ("noisy" in rep or "paired" in rep or "shots" in rep) else None,
                                    "noise": cfg["noise"] if ("noisy" in rep or "paired" in rep or "shots" in rep) else None,
                                    "extra": extra.get(rep, {}).get("selected_features")})
            rep_dir = tdir / rep
            mpath = rep_dir / "metrics.json"
            kind, selection, sim, obs = REP_INFO[rep]
            row = dict(base)
            row.update({"representation": rep, "repr_kind": kind, "selection": selection, "simulation": sim,
                        "observables": obs, "architecture": arch if (sim == "noisy" or rep in ("q_ideal_paired_z", "q_ideal_shots_z")) else ("ideal" if kind == "quantum" else "none")})
            row.update(extra.get(rep, {}))
            if cfg["run"].get("resume", True) and mpath.exists():
                m = read_json(mpath)
                if m.get("rep_hash") == rep_hash:
                    row.update(m)
                    row["status"] = "completed"
                    rows.append(row)
                    logger.info(f"   {rep}: already completed (resume)")
                    continue
            title = f"{clf_name.upper()} | {task['dataset']} | {task['fmap']} | {rep} | {arch} | seed {seed}"
            m = _train_eval_rep(rep, data, clf_name, cfg, seed, threads, rep_dir, title, logger,
                                cfg["run"].get("save_predictions", True), cfg["run"]["task_plots"])
            m["rep_hash"] = rep_hash
            write_json(mpath, m)
            row.update(m)
            row["status"] = "completed"
            rows.append(row)
        status = "completed"
        err = ""
    except Exception as e:  # worker-level error handling: record, never crash the whole run
        status = "failed"
        err = f"{e.__class__.__name__}: {e}"
        tb = traceback.format_exc()
        atomic_write_text(tdir / "error.txt", tb)
        logger.error(tb)
        rows.append({**base, "status": "failed", "error": err})
    total = time.perf_counter() - t_task
    for r in rows:
        r.update({"task_seconds": total, "peak_memory_mb": peak_memory_mb(),
                  **{f"time_{k}": v for k, v in timer.t.items()}})
        if "cache" in locals():
            r.update(cache.stats())
    write_json(tdir / "task_status.json", {"status": status, "error": err, "task": task, "seconds": total,
                                           "task_hash": task["task_hash"], "rows": rows})
    logger.info(f"=== task {task['task_id']} {status} in {total / 60:.1f} min; peak mem {peak_memory_mb():.0f} MB")
    close_logger(logger)
    return rows


# ============================================================================ orchestration (main process)
def config_fingerprint(cfg: dict, clf_name: str, arch: str) -> str:
    """Everything that can change a task's results (excludes workers/threads/logging/resume flags)."""
    keep = {k: cfg[k] for k in ("datasets", "split", "sampling", "preprocessing", "feature_maps", "observables",
                                "sfs", "noise", "evaluation", "baselines") if k in cfg}
    keep["classifier"] = cfg["classifiers"][clf_name]
    keep["architecture"] = cfg["architectures"][arch]
    keep["sample_limit"] = cfg["run"].get("sample_limit")
    if cfg["run"].get("arch_only"):          # only added when set -> existing task hashes unchanged
        keep["arch_only"] = True
    return stable_hash(keep)


def _task_list(cfg, prepared, include_classical, cfg_fp=""):
    tasks = []
    order = {"paysim": 0, "banksim": 1, "creditcard": 2}  # heavy first -> better load balancing
    for dname in sorted(cfg["run"]["datasets"], key=lambda d: order.get(d, 9)):
        for seed in cfg["run"]["seeds"]:
            fmaps = list(cfg["run"]["feature_maps"]) + (["classical"] if include_classical else [])
            for fm in fmaps:
                t = {"dataset": dname, "fmap": fm, "seed": seed, "prep_dir": prepared[(dname, seed)],
                     "task_id": f"{dname}__{fm}__seed{seed}"}
                t["task_hash"] = stable_hash({"t": t, "v": TASK_VERSION, "cfg": cfg_fp})
                tasks.append(t)
    return tasks


def collect_results(out_dir: Path) -> pd.DataFrame:
    rows = []
    for st in sorted((out_dir / "tasks").glob("*/task_status.json")):
        try:
            rows.extend(read_json(st)["rows"])
        except Exception:
            pass
    return pd.DataFrame(rows)


def _summaries(out_dir: Path, df: pd.DataFrame, title: str):
    if df.empty:
        return
    write_csv_df(out_dir / "results.csv", df)
    ok = df[df["status"] == "completed"] if "status" in df else df
    if ok.empty or "test_roc_auc" not in ok:
        return
    metrics = ["test_roc_auc", "test_avg_precision", "test_f1", "test_recall", "test_precision", "test_fpr",
               "test_mcc", "test_balanced_accuracy", "val_roc_auc", "train_seconds"]
    metrics = [m for m in metrics if m in ok]
    g = ok.groupby(["dataset", "feature_map", "representation", "architecture"])[metrics]
    summ = g.agg(["mean", "std", "count"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    write_csv_df(out_dir / "summary_mean_std.csv", summ.reset_index())
    try:
        from .plotting import script_summary_figure
        script_summary_figure(out_dir / "summary.png", ok, title)
    except Exception:
        pass


def run_script(clf_name: str, arch: str, argv=None):
    parser = build_parser(f"Quantum feature extraction + {clf_name.upper()} on the {arch} configuration "
                          f"(all datasets x feature maps x seeds).")
    args = parser.parse_args(argv)
    cfg = apply_cli(load_config(args.config), args)
    out_dir = Path(cfg["paths"]["results_dir"]) / cfg["run"]["run_tag"] / f"{clf_name}_{arch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"main.{clf_name}.{arch}", out_dir / "run.log", cfg["run"]["log_level"])
    mode = cfg["baselines"].get("classical_all", "auto")
    include_classical = (mode == "on") or (mode == "auto" and arch == "superconducting")
    atomic_write_text(out_dir / "run_config.yaml", yaml.safe_dump(cfg, sort_keys=False))
    write_json(out_dir / "environment.json", environment_info())
    logger.info(f"#### {clf_name} x {arch} | workers={cfg['run']['workers']} threads/worker="
                f"{cfg['run']['threads_per_worker']} | datasets={cfg['run']['datasets']} fmaps={cfg['run']['feature_maps']} "
                f"seeds={cfg['run']['seeds']} | noise={cfg['noise']['enabled']} | out={out_dir}")
    if cfg.get("_pilot"):
        logger.warning("PILOT MODE: small caps, NOT for paper numbers")
    t0 = time.perf_counter()
    prepared = prepare_all(cfg, cfg["run"]["datasets"], cfg["run"]["seeds"], logger)
    tasks = _task_list(cfg, prepared, include_classical, config_fingerprint(cfg, clf_name, arch))
    todo = []
    for t in tasks:
        st = out_dir / "tasks" / t["task_id"] / "task_status.json"
        if cfg["run"].get("resume", True) and st.exists():
            s = read_json(st)
            if s.get("task_hash") == t["task_hash"]:
                if s["status"] == "completed":
                    continue
                if not cfg["run"].get("retry_failed"):
                    logger.info(f"skip previously failed {t['task_id']} (use --retry-failed)")
                    continue
        todo.append(t)
    logger.info(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run")
    workers = max(1, min(int(cfg["run"]["workers"]), len(todo) or 1))
    done = 0
    attempts = 0
    while todo and attempts < 2:
        attempts += 1
        remaining = list(todo)
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
                futs = {ex.submit(run_task, t, cfg, clf_name, arch, str(out_dir)): t for t in remaining}
                try:
                    for fut in as_completed(futs):
                        t = futs[fut]
                        try:
                            rows = fut.result()
                            ok = all(r.get("status") == "completed" for r in rows)
                            done += 1
                            todo.remove(t)
                            logger.info(f"[{done}/{len(tasks)}] {t['task_id']}: {'OK' if ok else 'FAILED'} "
                                        f"({(time.perf_counter() - t0) / 60:.1f} min elapsed)")
                        except BrokenProcessPool:
                            raise
                        except Exception as e:
                            logger.error(f"{t['task_id']} crashed: {e}")
                            todo.remove(t)
                        _summaries(out_dir, collect_results(out_dir), f"{clf_name.upper()} / {arch}")
                except KeyboardInterrupt:
                    logger.warning("Interrupted: cancelling pending tasks; completed results are saved. "
                                   "Re-run the same command to resume.")
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
        except BrokenProcessPool:
            logger.error("A worker process died (out of memory?). Restarting the pool for remaining tasks; "
                         "consider --workers 4.")
    df = collect_results(out_dir)
    _summaries(out_dir, df, f"{clf_name.upper()} / {arch}")
    n_fail = int((df["status"] == "failed").sum()) if (not df.empty and "status" in df) else 0
    logger.info(f"#### finished {clf_name} x {arch} in {(time.perf_counter() - t0) / 3600:.2f} h; "
                f"{len(df)} result rows, {n_fail} failed. Results: {out_dir / 'results.csv'}")
    close_logger(logger)
    return df
