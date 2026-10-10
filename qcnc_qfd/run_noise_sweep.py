"""One-at-a-time noise-parameter sweeps for each architecture configuration (cf. the sweeps of the earlier version on the main branch).

For each architecture x feature map x noise parameter x strength factor, the <Z_i> features of the
SFS-selected subset are recomputed under the modified noise model (all other parameters at baseline),
a classifier (default XGBoost, fixed best-practice grid) is trained/tuned on the noisy training/validation
rows and evaluated on the noisy test rows. Exact expectation values and finite shots are both reported.
Strength factors multiply error probabilities and gate times, and DIVIDE T1/T2 (factor > 1 = worse).
The readout parameter now has a real effect (it is applied to measured probabilities).

    python run_noise_sweep.py --pilot
    python run_noise_sweep.py --arch superconducting
    python run_noise_sweep.py --arch trapped_ion                # may run on another computer
"""
import qfd.threads  # noqa: F401

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.config import apply_cli, build_parser, load_config
from qfd.featuremaps import fmap_params, observable_matrices
from qfd.metrics import choose_threshold, full_metrics
from qfd.noisy import noisy_probabilities, sample_shots
from qfd.pipeline import get_sfs
from qfd.preprocess import load_prepared, prepare_all
from qfd.utils import get_logger, limit_threadpools, write_csv_df

PARAMS = {"superconducting": ["T1_us", "T2_us", "gate_time_1q_ns", "gate_time_2q_ns", "p_dep_1q", "p_dep_2q", "p_readout"],
          "trapped_ion": ["p_dep_1q", "p_dep_2q", "p_amp_damp", "p_phase_damp", "p_readout"]}
FACTORS = [0.1, 0.3, 1.0, 3.0, 10.0]


def _subset(y, pos_cap, neg_cap, rng):
    p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    p = rng.choice(p, min(len(p), pos_cap), replace=False)
    n = rng.choice(n, min(len(n), neg_cap), replace=False)
    return np.sort(np.r_[p, n]), len(n)


def run_point(job, cfg):
    from qfd.classifiers import scores, tune_and_fit
    limit_threadpools(1)
    A, meta = load_prepared(job["prep_dir"])
    fm = fmap_params(cfg, job["fmap"])
    cols = [meta["feature_names"].index(f) for f in job["subset"]]
    k = len(cols)
    a = dict(cfg["architectures"][job["arch"]])
    base = a[job["param"]]
    val = base / job["factor"] if job["param"].startswith("T") else min(base * job["factor"], 0.5 if job["param"].startswith("p_") else 1e9)
    if job["param"] == "T2_us":
        val = min(val, 2 * a["T1_us"])
    rng = np.random.default_rng(job["seed"] + 5)
    caps = job["caps"]
    tr, _ = _subset(A["y_train"], caps["train_pos"], caps["train_neg"], rng)
    va, nva = _subset(A["y_val"], 10**9, caps["val_neg"], rng)
    te, nte = _subset(A["y_test"], 10**9, caps["test_neg"], rng)
    wva = np.where(A["y_val"][va] == 1, 1.0, (A["y_val"] == 0).sum() / max(nva, 1))
    wte = np.where(A["y_test"][te] == 1, 1.0, (A["y_test"] == 0).sum() / max(nte, 1))
    M, _ = observable_matrices(k, False)
    t0 = time.perf_counter()
    P = {part: noisy_probabilities(A[f"Xq_{part}"][rows][:, cols], fm, job["arch"], a, cfg["noise"], {job["param"]: val})[0]
         for part, rows in (("train", tr), ("val", va), ("test", te))}
    t_sim = time.perf_counter() - t0
    out = []
    for mode in ("exact", "shots"):
        F = {}
        for i, (part, Pp) in enumerate(P.items()):
            if mode == "shots":
                Pp = sample_shots(Pp, int(cfg["noise"].get("shots") or 1024), job["seed"] * 10 + i)
            F[part] = Pp @ M
        m, hp, _, _ = tune_and_fit(job["clf"], cfg["classifiers"][job["clf"]], F["train"], A["y_train"][tr],
                                   F["val"], A["y_val"][va], wva, job["seed"], 1)
        s_va, s_te = scores(job["clf"], m, F["val"]), scores(job["clf"], m, F["test"])
        thr = choose_threshold(A["y_val"][va], s_va, wva)
        mt = full_metrics(A["y_test"][te], s_te, thr, wte)
        out.append({**{k_: v for k_, v in job.items() if k_ not in ("prep_dir", "caps", "subset")},
                    "subset": ";".join(job["subset"]), "value": val, "baseline": base, "mode": mode,
                    "sim_seconds": t_sim, **{f"test_{k_}": v for k_, v in mt.items()}})
    return out


def main():
    p = build_parser(__doc__)
    p.add_argument("--arch", nargs="+", default=["superconducting", "trapped_ion"])
    p.add_argument("--classifier", default="xgb", choices=["mlp", "xgb", "svm"])
    p.add_argument("--sweep-dataset", default="creditcard")
    p.add_argument("--plot-only", action="store_true", help="merge results_*.csv (e.g. from two computers) and plot")
    args = p.parse_args()
    cfg = apply_cli(load_config(args.config), args)
    if args.plot_only:
        out = Path(cfg["paths"]["results_dir"]) / cfg["run"]["run_tag"] / "noise_sweep"
        merged = pd.concat([pd.read_csv(f) for f in out.glob("results_*.csv")], ignore_index=True)
        write_csv_df(out / "results.csv", merged)
        plot(merged, out)
        return
    seed = cfg["run"]["seeds"][0]
    cfg["run"]["datasets"] = [args.sweep_dataset]
    caps = dict(train_pos=300, train_neg=1200, val_neg=1000, test_neg=2000)
    factors = FACTORS
    if cfg.get("_pilot"):
        caps = dict(train_pos=60, train_neg=150, val_neg=100, test_neg=150)
        factors = [0.1, 1.0, 10.0]
    out = Path(cfg["paths"]["results_dir"]) / cfg["run"]["run_tag"] / "noise_sweep"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("noise_sweep", out / "run.log")
    prepared = prepare_all(cfg, [args.sweep_dataset], [seed], log)
    prep = prepared[(args.sweep_dataset, seed)]
    A, meta = load_prepared(prep)
    jobs = []
    for fmn in cfg["run"]["feature_maps"]:
        st = get_sfs(cfg, A, meta, args.sweep_dataset, fmn, seed, log)
        for arch in args.arch:
            for prm in PARAMS[arch]:
                for f in factors:
                    jobs.append({"arch": arch, "fmap": fmn, "param": prm, "factor": f, "seed": seed, "clf": args.classifier,
                                 "dataset": args.sweep_dataset, "prep_dir": prep, "caps": caps, "subset": st["best_subset"]})
    log.info(f"{len(jobs)} sweep points")
    rows = []
    csv = out / f"results_{'_'.join(args.arch)}.csv"
    with ProcessPoolExecutor(int(cfg["run"]["workers"]), mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(run_point, j, cfg) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                rows.extend(fu.result())
            except Exception as e:
                log.error(f"sweep point failed: {e}")
            write_csv_df(csv, pd.DataFrame(rows))
            if i % 5 == 0 or i == len(jobs):
                log.info(f"[{i}/{len(jobs)}]")
    df = pd.DataFrame(rows)
    write_csv_df(csv, df)
    # merged file for analyze_results.py
    merged = pd.concat([pd.read_csv(f) for f in out.glob("results_*.csv")], ignore_index=True)
    write_csv_df(out / "results.csv", merged)
    plot(merged, out)


def plot(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for arch in sorted(df["arch"].unique()):
        d = df[df["arch"] == arch]
        prms = PARAMS[arch]
        fig, axes = plt.subplots(2, (len(prms) + 1) // 2, figsize=(2.6 * ((len(prms) + 1) // 2), 4.6), squeeze=False)
        for ax, prm in zip(axes.flat, prms):
            for j, fm in enumerate(["Z", "ZZ", "Pauli"]):
                for mode, ls in (("exact", "-"), ("shots", "--")):
                    s = d[(d["param"] == prm) & (d["fmap"] == fm) & (d["mode"] == mode)].sort_values("value")
                    if len(s):
                        ax.plot(s["value"], s["test_roc_auc"], ls, marker="o", ms=3, color=f"C{j}",
                                label=f"{fm} ({mode})")
            ax.set_xscale("log")
            ax.set_title(prm, fontsize=7)
            ax.set_ylabel("Test ROC-AUC", fontsize=7)
        for ax in axes.flat[len(prms):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=5)
        fig.suptitle(f"Noise sensitivity of <Z> features, {arch} configuration (one parameter varied at a time)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"fig_noise_sweep_{arch}.png", dpi=250)
        fig.savefig(out / f"fig_noise_sweep_{arch}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
