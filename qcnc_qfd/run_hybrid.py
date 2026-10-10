"""Phase 2, experiment 1: do quantum features add information on top of ALL classical features?

For every dataset x seed x classifier (same rows, splits, tuning and threshold rule as the main run):
  c_all                 all classical features                                          [reference]
  c_all+q_<map>         c_all + ideal <Z_i> of that map's SFS subset (k features)        [hybrid]
  c_all+rff_<map>       c_all + k random Fourier features cos(w.u + b) of the SAME inputs [classical control]
The control answers the obvious reviewer question: is any gain "quantum", or just extra non-linear features
of the same inputs with the same dimension?

Statistics: paired Poisson bootstrap of the test ROC-AUC / AP difference (same test rows and resamples),
per seed, plus mean +- s.d. over seeds.

    python run_hybrid.py --pilot
    python run_hybrid.py                               # after the main runs (reuses SFS + feature caches)
Outputs: results/<run_tag>/hybrid/{results.csv, paired_tests.csv, summary.csv, table_hybrid.tex,
                                    fig_hybrid_delta.png/.pdf, <clf>/<dataset>__seed<k>/<rep>/...}
"""
import qfd.threads  # noqa: F401

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.config import apply_cli, build_parser, load_config
from qfd.phase2 import PARTS, ideal_z, load_task
from qfd.preprocess import load_prepared, prepare_all
from qfd.utils import get_logger, limit_threadpools, load_npz, read_json, stable_hash, write_csv_df, write_json

VERSION = 1
CLFS = ("mlp", "xgb", "svm")


def rff(U: dict, k: int, seed: int) -> dict:
    """k random Fourier features of [0,1]-scaled inputs; bandwidth comparable to the feature-map angles."""
    rng = np.random.default_rng(seed)
    d = U["train"].shape[1]
    W = rng.normal(0.0, np.pi, size=(d, k))
    b = rng.uniform(0, 2 * np.pi, size=k)
    return {p: np.cos(U[p] @ W + b) for p in PARTS}


def run_job(job, cfg, out_root):
    from qfd.pipeline import _train_eval_rep
    limit_threadpools(int(cfg["run"]["threads_per_worker"]))
    rep_dir = Path(out_root) / job["clf"] / f"{job['dataset']}__seed{job['seed']}" / job["rep"]
    h = stable_hash({"v": VERSION, "job": {k: v for k, v in job.items() if k != "prep_dir"},
                     "clf": cfg["classifiers"][job["clf"]], "eval": cfg["evaluation"], "sfs": cfg["sfs"],
                     "fm": cfg["feature_maps"], "prep": Path(job["prep_dir"]).name})
    mpath = rep_dir / "metrics.json"
    row = {k: v for k, v in job.items() if k != "prep_dir"}
    if mpath.exists():
        m = read_json(mpath)
        if m.get("job_hash") == h:
            return {**row, **m, "status": "completed", "resumed": True}
    A, meta = load_prepared(job["prep_dir"])
    Xc = {p: A[f"Xc_{p}"].astype(np.float64) for p in PARTS}
    extra = {"n_classical": int(Xc["train"].shape[1]), "added_features": ""}
    if job["rep"] != "c_all":
        kind, fmap = job["rep"].split("+")[1].split("_", 1)
        T = load_task(cfg, job["prep_dir"], job["dataset"], fmap, job["seed"])
        if kind == "q":
            add = ideal_z(cfg, T)
        else:
            U = {p: A[f"Xq_{p}"][:, T["cols"]].astype(np.float64) for p in PARTS}
            add = rff(U, T["k"], seed=1000 * job["seed"] + {"Z": 1, "ZZ": 2, "Pauli": 3}[fmap])
        Xc = {p: np.hstack([Xc[p], add[p]]) for p in PARTS}
        extra.update({"added_features": ";".join(T["sfs"]["best_subset"]), "n_added": T["k"], "feature_map": fmap,
                      "added_kind": "quantum <Z_i> (ideal)" if kind == "q" else "random Fourier (classical control)"})
    data = dict(Xtr=Xc["train"], ytr=A["y_train"], Xva=Xc["val"], yva=A["y_val"], wva=A["w_val"],
                Xte=Xc["test"], yte=A["y_test"], wte=A["w_test"])
    import logging
    logger = logging.getLogger(f"hybrid.{job['clf']}")
    t0 = time.perf_counter()
    m = _train_eval_rep(job["rep"], data, job["clf"], cfg, job["seed"], int(cfg["run"]["threads_per_worker"]), rep_dir,
                        f"hybrid {job['rep']} | {job['clf']} | {job['dataset']} seed {job['seed']}", logger,
                        save_preds=True, plots=bool(cfg["run"]["task_plots"]))
    m.update(extra)
    m["job_hash"] = h
    m["job_seconds"] = time.perf_counter() - t0
    from qfd.utils import write_json as wj
    wj(mpath, m)
    return {**row, **m, "status": "completed", "resumed": False}


def _paired_group(clf, ds, seed, reps, out_root, n_boot):
    """All paired comparisons for one (classifier, dataset, seed) -- runs in a worker process."""
    from qfd.metrics import paired_bootstrap_diff
    limit_threadpools(1)
    base_dir = Path(out_root) / clf / f"{ds}__seed{seed}"
    try:
        ref = load_npz(base_dir / "c_all" / "predictions.npz")
    except Exception:
        return []
    rows = []
    for fmap in ("Z", "ZZ", "Pauli"):
        pairs = [(f"c_all+q_{fmap}", "c_all", "quantum vs classical"),
                 (f"c_all+rff_{fmap}", "c_all", "RFF control vs classical"),
                 (f"c_all+q_{fmap}", f"c_all+rff_{fmap}", "quantum vs RFF control")]
        for b, a, label in pairs:
            if a not in reps or b not in reps:
                continue
            pa = ref if a == "c_all" else load_npz(base_dir / a / "predictions.npz")
            pb = load_npz(base_dir / b / "predictions.npz")
            assert np.array_equal(pa["y_test"], pb["y_test"])
            r = paired_bootstrap_diff(pa["y_test"], pa["score_test"], pb["score_test"], pa["w_test"],
                                      n_boot=n_boot, seed=seed)
            rows.append({"classifier": clf, "dataset": ds, "seed": seed, "feature_map": fmap, "comparison": label,
                         "A": a, "B": b, "n_test": int(len(pa["y_test"])), **r})
    return rows


def paired_tests(df, out_root, n_boot, log, workers=1):
    """Paired bootstrap tests, parallel over (classifier, dataset, seed) groups, with progress + ETA logging."""
    groups = [(clf, ds, int(seed), set(g["rep"])) for (clf, ds, seed), g in df.groupby(["clf", "dataset", "seed"])]
    log.info(f"paired bootstrap tests: {len(groups)} groups x up to 9 comparisons, {n_boot} resamples each, "
             f"{workers} workers (PaySim groups are the slow ones: ~400k test rows)")
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max(1, workers), mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(_paired_group, clf, ds, seed, reps, str(out_root), n_boot): (clf, ds, seed)
                for clf, ds, seed, reps in groups}
        for i, fu in enumerate(as_completed(futs), 1):
            clf, ds, seed = futs[fu]
            try:
                r = fu.result()
                rows.extend(r)
            except Exception as e:
                log.error(f"  paired tests {clf} {ds} seed{seed} FAILED: {e}")
                continue
            el = time.perf_counter() - t0
            log.info(f"  [{i}/{len(groups)}] {clf} {ds} seed{seed}: {len(r)} comparisons | "
                     f"{el / 60:.1f} min elapsed, ~{el / i * (len(groups) - i) / 60:.1f} min left")
    log.info(f"{len(rows)} paired comparisons in {(time.perf_counter() - t0) / 60:.1f} min")
    return pd.DataFrame(rows)


def summarise(pt: pd.DataFrame, out: Path):
    if pt.empty:
        return pt
    g = pt.groupby(["comparison", "dataset", "classifier", "feature_map"])
    s = g.agg(d_auc_mean=("d_auc", "mean"), d_auc_sd=("d_auc", "std"), d_ap_mean=("d_ap", "mean"), d_ap_sd=("d_ap", "std"),
              n_seeds=("seed", "nunique"),
              seeds_auc_sig_better=("d_auc_ci_low", lambda v: int((v > 0).sum())),
              seeds_auc_sig_worse=("d_auc_ci_high", lambda v: int((v < 0).sum())),
              seeds_ap_sig_better=("d_ap_ci_low", lambda v: int((v > 0).sum())),
              seeds_ap_sig_worse=("d_ap_ci_high", lambda v: int((v < 0).sum()))).reset_index()
    write_csv_df(out / "summary.csv", s)
    # LaTeX: quantum vs classical and quantum vs RFF, d_AUC / d_AP (mean +- sd), significant seeds
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize",
             "\\caption{Hybrid features: change in test ROC-AUC / AP when $k$ features are added to all classical "
             "features (mean $\\pm$ s.d. over seeds; [b/w] = seeds whose paired 95\\% bootstrap CI is entirely above/below 0).}",
             "\\label{tab:hybrid}", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lllll}", "\\hline",
             "Dataset & Clf. & Map & $\\Delta$AUC (Q$-$C) & $\\Delta$AUC (Q$-$RFF) \\\\", "\\hline"]
    for (ds, clf, fm), _ in s.groupby(["dataset", "classifier", "feature_map"]):
        cells = []
        for comp in ("quantum vs classical", "quantum vs RFF control"):
            r = s[(s["dataset"] == ds) & (s["classifier"] == clf) & (s["feature_map"] == fm) & (s["comparison"] == comp)]
            if len(r):
                r = r.iloc[0]
                cells.append(f"{r['d_auc_mean']:+.4f} $\\pm$ {0 if np.isnan(r['d_auc_sd']) else r['d_auc_sd']:.4f} "
                             f"[{r['seeds_auc_sig_better']}/{r['seeds_auc_sig_worse']}]")
            else:
                cells.append("--")
        lines.append(f"{ds} & {clf.upper()} & {fm} & {cells[0]} & {cells[1]} \\\\")
    lines += ["\\hline", "\\end{tabular}}", "\\end{table}", ""]
    (out / "table_hybrid.tex").write_text("\n".join(lines), encoding="utf-8")
    return s


def plot(s: pd.DataFrame, out: Path):
    if s.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dss = sorted(s["dataset"].unique())
    fig, axes = plt.subplots(2, len(dss), figsize=(3.2 * len(dss), 4.4), squeeze=False)
    for col, ds in enumerate(dss):
        for row, (metric, lab) in enumerate((("d_auc", "ROC-AUC"), ("d_ap", "AP"))):
            ax = axes[row, col]
            d = s[(s["dataset"] == ds) & (s["comparison"].isin(["quantum vs classical", "RFF control vs classical"]))]
            labels = [f"{c.upper()}-{f}" for c in CLFS for f in ("Z", "ZZ", "Pauli")]
            x = np.arange(len(labels))
            for j, (comp, color) in enumerate((("quantum vs classical", "C0"), ("RFF control vs classical", "C7"))):
                vals, errs = [], []
                for c in CLFS:
                    for f in ("Z", "ZZ", "Pauli"):
                        r = d[(d["classifier"] == c) & (d["feature_map"] == f) & (d["comparison"] == comp)]
                        vals.append(r[f"{metric}_mean"].iloc[0] if len(r) else np.nan)
                        errs.append(r[f"{metric}_sd"].iloc[0] if len(r) else np.nan)
                ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, yerr=np.nan_to_num(errs), color=color, capsize=1.5,
                       label="+ quantum $\\langle Z\\rangle$" if j == 0 else "+ random Fourier (control)")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
            ax.set_title(f"{ds}: $\\Delta$ test {lab} vs all-classical", fontsize=7)
            ax.tick_params(axis="y", labelsize=6)
    axes[0, 0].legend(fontsize=5.5)
    fig.tight_layout()
    fig.savefig(out / "fig_hybrid_delta.png", dpi=250)
    fig.savefig(out / "fig_hybrid_delta.pdf")
    plt.close(fig)


def main():
    p = build_parser(__doc__)
    p.add_argument("--classifiers", nargs="+", default=list(CLFS), choices=list(CLFS))
    p.add_argument("--no-rff", action="store_true", help="skip the random-Fourier-feature control")
    p.add_argument("--n-boot-paired", type=int, default=500)
    p.add_argument("--tests-only", action="store_true", help="skip training; read results.csv and run the paired tests")
    args = p.parse_args()
    cfg = apply_cli(load_config(args.config), args)
    out = Path(cfg["paths"]["results_dir"]) / cfg["run"]["run_tag"] / "hybrid"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("hybrid", out / "run.log")
    prepared = prepare_all(cfg, cfg["run"]["datasets"], cfg["run"]["seeds"], log)
    reps = ["c_all"] + [f"c_all+q_{f}" for f in cfg["run"]["feature_maps"]]
    if not args.no_rff:
        reps += [f"c_all+rff_{f}" for f in cfg["run"]["feature_maps"]]
    order = {"paysim": 0, "banksim": 1, "creditcard": 2}
    jobs = [{"clf": c, "dataset": d, "seed": s, "rep": r, "prep_dir": prepared[(d, s)]}
            for d in sorted(cfg["run"]["datasets"], key=lambda x: order.get(x, 9))
            for c in sorted(args.classifiers, key=lambda x: {"svm": 0, "mlp": 1, "xgb": 2}[x])
            for s in cfg["run"]["seeds"] for r in reps]
    log.info(f"{len(jobs)} model runs ({len(reps)} representations) | workers={cfg['run']['workers']}")
    rows, t0 = [], time.perf_counter()
    if args.tests_only:
        jobs = []
        rows = pd.read_csv(out / "results.csv").to_dict("records")
        log.info(f"--tests-only: {len(rows)} finished model runs read from results.csv")
    with ProcessPoolExecutor(int(cfg["run"]["workers"]), mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(run_job, j, cfg, str(out)): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]
            try:
                r = fu.result()
                rows.append(r)
                log.info(f"[{i}/{len(jobs)}] {j['clf']} {j['dataset']} seed{j['seed']} {j['rep']:16s} "
                         f"AUC {r['test_roc_auc']:.4f} AP {r['test_avg_precision']:.4f}"
                         f"{' (resumed)' if r.get('resumed') else ''} | {(time.perf_counter() - t0) / 60:.1f} min")
            except Exception as e:
                rows.append({**{k: v for k, v in j.items() if k != "prep_dir"}, "status": "failed", "error": str(e)})
                log.error(f"[{i}/{len(jobs)}] {j['clf']} {j['dataset']} seed{j['seed']} {j['rep']} FAILED: {e}")
            if i % 10 == 0:
                write_csv_df(out / "results.csv", pd.DataFrame(rows))
    df = pd.DataFrame(rows)
    write_csv_df(out / "results.csv", df)
    ok = df[df["status"] == "completed"]
    pt = paired_tests(ok, out, args.n_boot_paired, log, int(cfg["run"]["workers"]))
    write_csv_df(out / "paired_tests.csv", pt)
    s = summarise(pt, out)
    plot(s, out)
    write_json(out / "summary.json", {"n_runs": int(len(df)), "n_failed": int((df["status"] != "completed").sum()),
                                     "pilot": bool(cfg.get("_pilot")), "hours": (time.perf_counter() - t0) / 3600})
    if len(s):
        q = s[s["comparison"] == "quantum vs classical"]
        log.info("quantum vs classical (d_AUC mean, seeds sig. better/worse):\n" +
                 q[["dataset", "classifier", "feature_map", "d_auc_mean", "seeds_auc_sig_better", "seeds_auc_sig_worse",
                    "d_ap_mean"]].to_string(index=False))
    log.info(f"DONE in {(time.perf_counter() - t0) / 3600:.2f} h -> {out}")


if __name__ == "__main__":
    main()
