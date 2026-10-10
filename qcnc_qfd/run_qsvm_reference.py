"""Reference experiment: the ORIGINAL fidelity-kernel QSVM vs the revised feature-extraction approach.

For every dataset x feature map x seed (same SFS subset as the main experiments):
  qsvm_fidelity      SVC on the precomputed quantum kernel K_ij = |<psi(x_i)|psi(x_j)>|^2 (exact, ideal)
  svm_rbf_on_z       classical RBF-SVM on the <Z_i> features          } same training rows,
  svm_rbf_raw        classical RBF-SVM on the raw selected features   } same grid size, same val/test
Kernel methods scale as O(n^2), so all three use the same `--qsvm-train` rows (default 2000:
all available frauds up to half, rest genuine). Validation and test are the same weighted sets as
in the main experiments (original prevalence). Test kernels are evaluated in chunks.

    python run_qsvm_reference.py --pilot
    python run_qsvm_reference.py --qsvm-train 2000
"""
import qfd.threads  # noqa: F401

import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.config import apply_cli, build_parser, load_config
from qfd.featuremaps import fmap_params, ideal_features, statevectors
from qfd.metrics import bootstrap_ci, choose_threshold, full_metrics
from qfd.pipeline import get_sfs
from qfd.preprocess import load_prepared, prepare_all
from qfd.utils import get_logger, limit_threadpools, peak_memory_mb, write_csv_df, write_json

C_GRID = [0.1, 1.0, 10.0]
CW_GRID = [None, "balanced"]


def kernel(Pa, Pb):
    return np.abs(Pa @ Pb.conj().T) ** 2


def chunked_scores(model, Pe, Ptr, chunk=20000):
    return np.concatenate([model.decision_function(kernel(Pe[s:s + chunk], Ptr)) for s in range(0, len(Pe), chunk)])


def fit_select(make, Xtr, ytr, score_va, yva, wva):
    from qfd.metrics import avg_precision, roc_auc
    best, rows = None, []
    for C in C_GRID:
        for cw in CW_GRID:
            t0 = time.perf_counter()
            m = make(C, cw).fit(Xtr, ytr)
            s = score_va(m)
            auc, ap = roc_auc(yva, s, wva), avg_precision(yva, s, wva)
            rows.append({"C": C, "class_weight": str(cw), "val_roc_auc": auc, "val_avg_precision": ap,
                         "fit_seconds": time.perf_counter() - t0})
            if best is None or (auc, ap) > best[0]:
                best = ((auc, ap), m, {"C": C, "class_weight": cw})
    return best[1], best[2], rows


def run_one(task, cfg, out_dir, n_train):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    limit_threadpools(int(cfg["run"]["threads_per_worker"]))
    tdir = Path(out_dir) / "tasks" / f"{task['dataset']}__{task['fmap']}__seed{task['seed']}"
    tdir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"qsvm.{tdir.name}", tdir / "task.log")
    rows = []
    try:
        A, meta = load_prepared(task["prep_dir"])
        seed = task["seed"]
        state = get_sfs(cfg, A, meta, task["dataset"], task["fmap"], seed, log)
        names = meta["feature_names"]
        cols = [names.index(f) for f in state["best_subset"]]
        fm = fmap_params(cfg, task["fmap"])
        rng = np.random.default_rng(seed + 31)
        y = A["y_train"]
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        npos = min(len(pos), n_train // 2)
        rows_tr = np.sort(np.concatenate([rng.choice(pos, npos, replace=False),
                                          rng.choice(neg, min(len(neg), n_train - npos), replace=False)]))
        ytr, yva, yte, wva, wte = y[rows_tr], A["y_val"], A["y_test"], A["w_val"], A["w_test"]
        Utr, Uva, Ute = A["Xq_train"][rows_tr][:, cols], A["Xq_val"][:, cols], A["Xq_test"][:, cols]
        res = {}
        # --- fidelity-kernel QSVM
        t0 = time.perf_counter()
        Ptr, Pva, Pte = statevectors(Utr, fm), statevectors(Uva, fm), statevectors(Ute, fm)
        Ktr = kernel(Ptr, Ptr)
        t_k = time.perf_counter() - t0
        m, hp, grid = fit_select(lambda C, cw: SVC(kernel="precomputed", C=C, class_weight=cw), Ktr, ytr,
                                 lambda mm: chunked_scores(mm, Pva, Ptr), yva, wva)
        res["qsvm_fidelity"] = (chunked_scores(m, Pva, Ptr), chunked_scores(m, Pte, Ptr), hp, grid, t_k)
        # kernel heat-map data (100 genuine + 100 fraud training rows, sorted by class)
        sel = np.r_[np.flatnonzero(ytr == 0)[:100], np.flatnonzero(ytr == 1)[:100]]
        np.save(tdir / "kernel_sample_200.npy", Ktr[np.ix_(sel, sel)].astype(np.float32))
        # --- classical SVMs on the same rows
        Ztr, Zva, Zte = (ideal_features(U, fm)[0] for U in (Utr, Uva, Ute))
        for name, (a, b, c) in {"svm_rbf_on_z": (Ztr, Zva, Zte),
                                "svm_rbf_raw": (A["Xc_train"][rows_tr][:, cols], A["Xc_val"][:, cols], A["Xc_test"][:, cols])}.items():
            m, hp, grid = fit_select(lambda C, cw: make_pipeline(StandardScaler(), SVC(C=C, class_weight=cw, gamma="scale")),
                                     a, ytr, lambda mm, b=b: mm.decision_function(b), yva, wva)
            res[name] = (m.decision_function(b), m.decision_function(c), hp, grid, 0.0)
        for name, (s_va, s_te, hp, grid, t_k) in res.items():
            thr = choose_threshold(yva, s_va, wva)
            mt = full_metrics(yte, s_te, thr, wte)
            ci = bootstrap_ci(yte, s_te, wte, int(cfg["evaluation"]["n_bootstrap"]), seed)
            write_csv_df(tdir / f"{name}_grid.csv", pd.DataFrame(grid))
            row = {"dataset": task["dataset"], "feature_map": task["fmap"], "seed": seed, "model": name,
                   "selected_features": ";".join(state["best_subset"]), "n_qubits": len(cols), "n_train": len(ytr),
                   "n_train_pos": int(ytr.sum()), "best_hp": str(hp), "kernel_seconds": t_k, "status": "completed",
                   "peak_memory_mb": peak_memory_mb(), "pilot": bool(cfg.get("_pilot", False))}
            row.update({f"test_{k}": v for k, v in {**mt, **ci}.items()})
            row["val_roc_auc"] = max(g["val_roc_auc"] for g in grid)
            rows.append(row)
            log.info(f"{name:15s} test AUC {mt['roc_auc']:.4f} AP {mt['avg_precision']:.4f} | hp {hp}")
    except Exception as e:
        (tdir / "error.txt").write_text(traceback.format_exc())
        rows.append({"dataset": task["dataset"], "feature_map": task["fmap"], "seed": task["seed"],
                     "status": "failed", "error": str(e)})
    write_json(tdir / "rows.json", rows)
    return rows


def main():
    p = build_parser(__doc__)
    p.add_argument("--qsvm-train", type=int, default=2000, help="training rows for all kernel models")
    args = p.parse_args()
    cfg = apply_cli(load_config(args.config), args)
    n_train = 300 if cfg.get("_pilot") else args.qsvm_train
    out = Path(cfg["paths"]["results_dir"]) / cfg["run"]["run_tag"] / "qsvm_reference"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("qsvm", out / "run.log")
    prepared = prepare_all(cfg, cfg["run"]["datasets"], cfg["run"]["seeds"], log)
    tasks = [{"dataset": d, "fmap": f, "seed": s, "prep_dir": prepared[(d, s)]}
             for d in cfg["run"]["datasets"] for s in cfg["run"]["seeds"] for f in cfg["run"]["feature_maps"]]
    rows = []
    with ProcessPoolExecutor(int(cfg["run"]["workers"]), mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(run_one, t, cfg, str(out), n_train) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            write_csv_df(out / "results.csv", pd.DataFrame(rows))
            log.info(f"[{i}/{len(tasks)}] done")
    df = pd.DataFrame(rows)
    write_csv_df(out / "results.csv", df)
    if "test_roc_auc" in df:
        s = df[df["status"] == "completed"].groupby(["dataset", "feature_map", "model"])[
            ["test_roc_auc", "test_avg_precision", "test_f1", "kernel_seconds"]].agg(["mean", "std"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        write_csv_df(out / "summary.csv", s.reset_index())
        print(s.round(4).to_string())
    _heatmaps(out)


def _heatmaps(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted((out / "tasks").glob("*__seed0/kernel_sample_200.npy"))
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(2.6 * len(files), 2.6), squeeze=False)
    for ax, f in zip(axes[0], files):
        ax.imshow(np.load(f), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f.parent.name.replace("__seed0", "").replace("__", " "), fontsize=7)
        ax.axhline(99.5, color="w", lw=0.6)
        ax.axvline(99.5, color="w", lw=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Fidelity kernels (rows/cols: 100 genuine | 100 fraud training samples)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_kernel_heatmaps.png", dpi=250)
    fig.savefig(out / "fig_kernel_heatmaps.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
