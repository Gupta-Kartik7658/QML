"""Phase 2, experiment 2: shot budget. How many measurements does fraud detection with <Z_i> features need?

For every dataset x feature map x seed and every simulation (ideal, superconducting-noisy,
trapped-ion-noisy), the main run's cached exact probability vectors on the paired rows are re-sampled with
S shots (S = 16 ... 8192, plus the exact limit). A classifier (default XGBoost, the main run's grid) is
tuned on validation and evaluated on test (original-prevalence weights).

Test of Corollary 2 (theory/NOISE_INVARIANCE.md): for the Z map, noisy S-shot features should behave like
ideal features with S_eff = beta^2 * S shots, where beta is fitted on the same rows. The figure overlays the
noisy curves against S_eff.

    python run_shot_budget.py --pilot
    python run_shot_budget.py                    # needs the main runs' noisy caches (both architectures)
Outputs: results/<run_tag>/shot_budget/{results.csv, shots_to_target.csv, fig_shot_budget.png/.pdf,
                                         fig_shot_budget_effective.png/.pdf, summary.json}
"""
import qfd.threads  # noqa: F401

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.config import apply_cli, build_parser, load_config
from qfd.featuremaps import observable_matrices
from qfd.noisy import sample_shots
from qfd.phase2 import PARTS, affine_fit, ideal_probs_paired, load_task, noisy_probs, paired_rows
from qfd.preprocess import prepare_all
from qfd.utils import get_logger, limit_threadpools, write_csv_df, write_json

SHOTS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SIMS = ("ideal", "superconducting", "trapped_ion")


def run_job(job, cfg):
    from qfd.classifiers import scores, tune_and_fit
    from qfd.metrics import choose_threshold, full_metrics
    limit_threadpools(1)
    T = load_task(cfg, job["prep_dir"], job["dataset"], job["fmap"], job["seed"])
    k = T["k"]
    M, _ = observable_matrices(k, False)
    Pi = ideal_probs_paired(T)
    if job["sim"] == "ideal":
        P = Pi
        beta = np.ones(k)
    else:
        P = noisy_probs(cfg, T, job["sim"], allow_compute=False)
        if P is None:
            return []
        P = {p: np.asarray(P[p]) for p in PARTS}
        fits = affine_fit(np.concatenate([Pi[p] @ M for p in PARTS]), np.concatenate([P[p] @ M for p in PARTS]))
        beta = np.array([f["beta"] for f in fits])
    R = paired_rows(T)
    rows = []
    sim_id = SIMS.index(job["sim"])
    for S in job["shots"]:
        for draw in range(job["draws"]):
            F = {}
            for i, p in enumerate(PARTS):
                Pp = P[p] if S == 0 else sample_shots(P[p], S, 10_000_000 * job["seed"] + 100_000 * draw + 10 * S + 3 * i + sim_id)
                F[p] = Pp @ M
            t0 = time.perf_counter()
            m, hp, _, _ = tune_and_fit(job["clf"], cfg["classifiers"][job["clf"]], F["train"], R["y"]["train"],
                                       F["val"], R["y"]["val"], R["w"]["val"], job["seed"], 1)
            s_va, s_te = scores(job["clf"], m, F["val"]), scores(job["clf"], m, F["test"])
            thr = choose_threshold(R["y"]["val"], s_va, R["w"]["val"])
            mt = full_metrics(R["y"]["test"], s_te, thr, R["w"]["test"])
            rows.append({"dataset": job["dataset"], "fmap": job["fmap"], "seed": job["seed"], "sim": job["sim"],
                         "clf": job["clf"], "k": k, "shots": S if S else np.inf, "draw": draw,
                         "beta_mean": float(beta.mean()), "beta_sq_mean": float((beta ** 2).mean()),
                         "effective_shots": float((beta ** 2).mean() * S) if S else np.inf,
                         "fit_seconds": time.perf_counter() - t0, "best_hp": str(hp),
                         **{f"test_{a}": b for a, b in mt.items() if a in ("roc_auc", "avg_precision", "f1", "recall",
                                                                           "precision", "fpr", "mcc")}})
    return rows


def shots_to_target(df, tol=0.005):
    """Smallest S from which the mean test AUC stays within tol of the same configuration's exact-limit AUC."""
    out = []
    for (ds, fm, sim), g in df.groupby(["dataset", "fmap", "sim"]):
        m = g.groupby("shots")["test_roc_auc"].mean()
        if np.inf not in m.index:
            continue
        exact = m.loc[np.inf]
        fin = [s for s in sorted(m.index) if np.isfinite(s)]
        ok = [s for i, s in enumerate(fin) if all(m.loc[t] >= exact - tol for t in fin[i:])]  # stays within tol
        out.append({"dataset": ds, "fmap": fm, "sim": sim, "exact_auc": exact, "tolerance": tol,
                    "min_shots_within_tol": ok[0] if ok else np.nan,
                    "auc_at_1024": m.get(1024, np.nan), "auc_at_64": m.get(64, np.nan)})
    return pd.DataFrame(out)


def plot(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dss = sorted(df["dataset"].unique())
    fms = [f for f in ("Z", "ZZ", "Pauli") if f in set(df["fmap"])]
    for metric, fname in (("test_roc_auc", "fig_shot_budget"), ("test_avg_precision", "fig_shot_budget_ap")):
        fig, axes = plt.subplots(len(fms), len(dss), figsize=(2.9 * len(dss), 2.3 * len(fms)), squeeze=False)
        for r, fm in enumerate(fms):
            for c, ds in enumerate(dss):
                ax = axes[r, c]
                for j, sim in enumerate(SIMS):
                    g = df[(df["dataset"] == ds) & (df["fmap"] == fm) & (df["sim"] == sim)]
                    if g.empty:
                        continue
                    fin = g[np.isfinite(g["shots"])].groupby("shots")[metric].agg(["mean", "std"])
                    ax.errorbar(fin.index, fin["mean"], yerr=fin["std"].fillna(0), marker="o", ms=2.5, lw=1,
                                capsize=1.5, color=f"C{j}", label=sim.replace("_", "-"))
                    ex = g[~np.isfinite(g["shots"])][metric].mean()
                    ax.axhline(ex, color=f"C{j}", ls=":", lw=0.8)
                ax.set_xscale("log", base=2)
                ax.set_title(f"{ds} | {fm} map", fontsize=7)
                ax.tick_params(labelsize=6)
                if c == 0:
                    ax.set_ylabel(f"test {'ROC-AUC' if metric == 'test_roc_auc' else 'AP'}", fontsize=7)
                if r == len(fms) - 1:
                    ax.set_xlabel("shots per sample (dotted: exact limit)", fontsize=7)
        axes[0, 0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / f"{fname}.png", dpi=250)
        fig.savefig(out / f"{fname}.pdf")
        plt.close(fig)
    # Corollary 2: Z map, AUC vs effective shots beta^2 S
    z = df[(df["fmap"] == "Z") & np.isfinite(df["shots"])]
    if z.empty:
        return
    fig, axes = plt.subplots(1, len(dss), figsize=(2.9 * len(dss), 2.5), squeeze=False)
    for c, ds in enumerate(dss):
        ax = axes[0, c]
        for j, sim in enumerate(SIMS):
            g = z[(z["dataset"] == ds) & (z["sim"] == sim)]
            if g.empty:
                continue
            m = g.groupby("shots").agg(x=("effective_shots", "mean"), y=("test_roc_auc", "mean"))
            ax.plot(m["x"], m["y"], marker="o", ms=2.5, lw=1, color=f"C{j}",
                    label=f"{sim.replace('_', '-')} ($\\beta^2$={g['beta_sq_mean'].mean():.3f})")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{ds} | Z map", fontsize=7)
        ax.set_xlabel(r"effective shots $\beta^2 S$", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5)
    axes[0, 0].set_ylabel("test ROC-AUC", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_shot_budget_effective.png", dpi=250)
    fig.savefig(out / "fig_shot_budget_effective.pdf")
    plt.close(fig)


def main():
    p = build_parser(__doc__)
    p.add_argument("--classifier", default="xgb", choices=["mlp", "xgb", "svm"])
    p.add_argument("--shot-levels", type=int, nargs="+", default=SHOTS)
    p.add_argument("--draws", type=int, default=1, help="independent shot-sampling draws per level")
    p.add_argument("--sims", nargs="+", default=list(SIMS), choices=list(SIMS))
    args = p.parse_args()
    cfg = apply_cli(load_config(args.config), args)
    out = Path(cfg["paths"]["results_dir"]) / cfg["run"]["run_tag"] / "shot_budget"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("shot_budget", out / "run.log")
    shots = [16, 128, 1024] if cfg.get("_pilot") else list(args.shot_levels)
    shots = shots + [0]  # 0 = exact expectation values
    prepared = prepare_all(cfg, cfg["run"]["datasets"], cfg["run"]["seeds"], log)
    jobs = [{"dataset": d, "seed": s, "fmap": f, "sim": sim, "clf": args.classifier, "shots": shots, "draws": args.draws,
             "prep_dir": prepared[(d, s)]}
            for d in cfg["run"]["datasets"] for s in cfg["run"]["seeds"] for f in cfg["run"]["feature_maps"] for sim in args.sims]
    log.info(f"{len(jobs)} jobs x {len(shots)} shot levels x {args.draws} draws | classifier {args.classifier}")
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(int(cfg["run"]["workers"]), mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(run_job, j, cfg): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]
            r = []
            try:
                r = fu.result()
                if not r:
                    log.info(f"[{i}/{len(jobs)}] {j['dataset']} {j['fmap']} seed{j['seed']} {j['sim']}: noisy cache missing -> skipped")
                rows.extend(r)
            except Exception as e:
                log.error(f"[{i}/{len(jobs)}] {j['dataset']} {j['fmap']} seed{j['seed']} {j['sim']} FAILED: {e}")
            write_csv_df(out / "results.csv", pd.DataFrame(rows))
            if r:
                log.info(f"[{i}/{len(jobs)}] {j['dataset']} {j['fmap']} seed{j['seed']} {j['sim']} done "
                         f"({(time.perf_counter() - t0) / 60:.1f} min)")
    df = pd.DataFrame(rows)
    write_csv_df(out / "results.csv", df)
    if df.empty:
        log.error("no results")
        return
    st = shots_to_target(df)
    write_csv_df(out / "shots_to_target.csv", st)
    plot(df, out)
    write_json(out / "summary.json", {"n_rows": int(len(df)), "pilot": bool(cfg.get("_pilot")),
                                     "hours": (time.perf_counter() - t0) / 3600, "classifier": args.classifier})
    log.info("shots needed to come within 0.005 AUC of the exact limit:\n" + st.to_string(index=False))
    log.info(f"DONE -> {out}")


if __name__ == "__main__":
    main()
