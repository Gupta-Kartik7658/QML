"""Numerical check of the affine noise law for product (Z) feature maps (theory/NOISE_INVARIANCE.md).

Part A: synthetic inputs, Qiskit Aer density-matrix simulation.
  For each architecture x feature map x noise strength (all noise parameters scaled together):
  fit z_noisy = alpha + beta * z_ideal per feature. Report the max residual, and for the superconducting
  Z map compare beta with the closed form. ZZ / Pauli are the negative control (entangling gates).

Part B: real data, the main run's cached noisy features (exactly the paired rows of the main tables).
  1. Affine fit residual on the real rows.
  2. Pipeline invariance: train MLP / SVM / XGBoost (one fixed configuration) on ideal vs noisy exact
     <Z> features of the paired rows. Compare test scores and test AUC. The proposition predicts identical
     results for Z and differences for ZZ / Pauli.

    python run_theory_check.py --pilot
    python run_theory_check.py                         # after the main runs (uses their cache)
Outputs: results/<run_tag>/theory/{affine_synthetic.csv, affine_real.csv, pipeline_invariance.csv,
                                   summary.json, fig_affine_law.png/.pdf}
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
from qfd.featuremaps import fmap_params, ideal_probabilities, observable_matrices
from qfd.phase2 import (PARTS, affine_fit, ideal_probs_paired, load_task, noisy_probs, paired_rows,
                        z_from_probs)
from qfd.preprocess import prepare_all
from qfd.utils import get_logger, limit_threadpools, write_csv_df, write_json

ARCHS = ("superconducting", "trapped_ion")


def scaled_arch(a: dict, factor: float) -> dict:
    """All noise parameters x factor (T1/T2 divided), same convention as run_noise_sweep.py."""
    a = dict(a)
    for k, v in list(a.items()):
        if k in ("T1_us", "T2_us"):
            a[k] = v / factor
        elif k.startswith("p_"):
            a[k] = min(v * factor, 0.5)
        elif k.startswith("gate_time"):
            a[k] = v * factor
    if "T2_us" in a:
        a["T2_us"] = min(a["T2_us"], 2 * a["T1_us"])
    return a


def closed_form_beta_sc(a: dict) -> float:
    t, T1, T2 = a["gate_time_1q_ns"] * 1e-3, a["T1_us"], min(a["T2_us"], 2 * a["T1_us"])
    return float((1 - 2 * a["p_readout"]) * np.exp(-t / T1) * np.exp(-t / T2) * (1 - a["p_dep_1q"]) ** 2)


def synthetic_point(job, cfg):
    from qfd.noisy import noisy_probabilities
    limit_threadpools(1)
    fm = fmap_params(cfg, job["fmap"])
    rng = np.random.default_rng(job["k"] * 7 + 1)
    U = rng.random((job["n"], job["k"]))
    M, _ = observable_matrices(job["k"], False)
    zi = ideal_probabilities(U, fm) @ M  # float64 (ideal_features returns float32)
    a = scaled_arch(cfg["architectures"][job["arch"]], job["factor"])
    t0 = time.perf_counter()
    P, _ = noisy_probabilities(U, fm, job["arch"], a, cfg["noise"])
    zn = P @ M
    fits = affine_fit(np.asarray(zi, float), zn)
    row = {**{k: v for k, v in job.items()}, "sim_seconds": time.perf_counter() - t0,
           "max_abs_residual": max(f["max_abs_residual"] for f in fits),
           "min_r2": min(f["r2"] for f in fits),
           "beta_mean": float(np.mean([f["beta"] for f in fits])), "beta_min": min(f["beta"] for f in fits),
           "beta_max": max(f["beta"] for f in fits), "alpha_mean": float(np.mean([f["alpha"] for f in fits])),
           "ideal_is_cos_phi_max_err": float(np.abs(zi - np.cos(2 * (fm["input_range"][0] + (fm["input_range"][1] - fm["input_range"][0]) * U))).max())
           if job["fmap"] == "Z" else np.nan}
    if job["arch"] == "superconducting" and job["fmap"] == "Z":
        row["beta_closed_form"] = closed_form_beta_sc(a)
    sample = {"z_ideal": np.asarray(zi[:, 0], float).tolist(), "z_noisy": zn[:, 0].tolist()} if job["factor"] == 1.0 else None
    return row, sample


def real_point(job, cfg):
    from qfd.classifiers import make_model, fit_model, scores, expand_grid
    from qfd.metrics import roc_auc, avg_precision
    limit_threadpools(1)
    T = load_task(cfg, job["prep_dir"], job["dataset"], job["fmap"], job["seed"])
    Pn = noisy_probs(cfg, T, job["arch"], allow_compute=False)
    if Pn is None:
        return None
    Pi = ideal_probs_paired(T)
    zn, zi = z_from_probs(Pn, T["k"]), z_from_probs(Pi, T["k"])
    R = paired_rows(T)
    fits = affine_fit(np.concatenate([zi[p] for p in PARTS]), np.concatenate([zn[p] for p in PARTS]))
    base = {"dataset": job["dataset"], "fmap": job["fmap"], "seed": job["seed"], "arch": job["arch"], "k": T["k"],
            "selected": ";".join(T["sfs"]["best_subset"]), "n_rows": int(sum(len(R["y"][p]) for p in PARTS)),
            "max_abs_residual": max(f["max_abs_residual"] for f in fits), "min_r2": min(f["r2"] for f in fits),
            "beta_mean": float(np.mean([f["beta"] for f in fits])), "beta_min": min(f["beta"] for f in fits)}
    inv = []
    y, w = R["y"]["test"], R["w"]["test"]
    # round-off control: the ideal features perturbed at the 1e-12 level. Any ideal-vs-noisy difference of the
    # same size as ideal-vs-control is floating-point amplification in training, not information loss.
    rng = np.random.default_rng(job["seed"] + 99)
    zc = {p: zi[p] * (1.0 + 1e-12 * rng.standard_normal(zi[p].shape)) for p in PARTS}
    for clf in ("logreg", "svm", "xgb", "mlp"):
        out = {}
        for tag, Z in (("ideal", zi), ("noisy", zn), ("control", zc)):
            if clf == "logreg":  # convex + deterministic: the cleanest exact-invariance check
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", tol=1e-12, max_iter=10000))
                m.fit(Z["train"], R["y"]["train"])
                out[tag] = m.decision_function(Z["test"])
                hp = {"C": 1.0, "class_weight": "balanced"}
            else:
                c = cfg["classifiers"][clf]
                hp = expand_grid(c["grid"])[-1]  # one fixed configuration (no tuning -> isolates the representation)
                m = make_model(clf, hp, c.get("fixed", {}), job["seed"], 1, R["y"]["train"])
                fit_model(clf, m, Z["train"], R["y"]["train"], Z["val"], R["y"]["val"], R["w"]["val"])
                out[tag] = scores(clf, m, Z["test"])
        a_i, a_n, a_c = (roc_auc(y, out[t], w) for t in ("ideal", "noisy", "control"))
        inv.append({**{k: base[k] for k in ("dataset", "fmap", "seed", "arch", "k")}, "classifier": clf, "hp": json.dumps(hp),
                    "test_auc_ideal": a_i, "test_auc_noisy": a_n, "test_auc_roundoff_control": a_c,
                    "test_ap_ideal": avg_precision(y, out["ideal"], w), "test_ap_noisy": avg_precision(y, out["noisy"], w),
                    "auc_diff": a_n - a_i, "auc_diff_roundoff_control": a_c - a_i,
                    "max_abs_score_diff": float(np.abs(out["ideal"] - out["noisy"]).max()),
                    "max_abs_score_diff_roundoff_control": float(np.abs(out["ideal"] - out["control"]).max()),
                    "score_rank_corr": float(pd.Series(out["ideal"]).corr(pd.Series(out["noisy"]), method="spearman"))})
    return base, inv


def main():
    p = build_parser(__doc__)
    p.add_argument("--synthetic-rows", type=int, default=400)
    p.add_argument("--factors", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    p.add_argument("--skip-real", action="store_true")
    p.add_argument("--skip-synthetic", action="store_true", help="reuse affine_synthetic.csv from a previous run")
    args = p.parse_args()
    cfg = apply_cli(load_config(args.config), args)
    out = Path(cfg["paths"]["results_dir"]) / cfg["run"]["run_tag"] / "theory"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("theory", out / "run.log")
    n = 60 if cfg.get("_pilot") else args.synthetic_rows
    factors = [1.0, 10.0] if cfg.get("_pilot") else args.factors
    jobs = [{"arch": a, "fmap": f, "k": k, "factor": fac, "n": n}
            for a in ARCHS for f in cfg["run"]["feature_maps"] for fac in factors
            for k in ((3,) if cfg.get("_pilot") else (3, 7 if f == "Z" else 5))]
    rows, samples = [], {}
    if args.skip_synthetic and (out / "affine_synthetic.csv").exists():
        jobs = []
        rows = pd.read_csv(out / "affine_synthetic.csv").to_dict("records")
    log.info(f"Part A: {len(jobs)} synthetic configurations")
    with ProcessPoolExecutor(int(cfg["run"]["workers"]), mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(synthetic_point, j, cfg): j for j in jobs}
        for fu in as_completed(futs):
            r, s = fu.result()
            rows.append(r)
            if s is not None:
                samples[(r["arch"], r["fmap"], r["k"])] = s
            log.info(f"  {r['arch']:15s} {r['fmap']:5s} k={r['k']} x{r['factor']:<4g} beta={r['beta_mean']:.5f} "
                     f"max|resid|={r['max_abs_residual']:.1e} R2min={r['min_r2']:.6f}"
                     + (f" closed-form beta={r['beta_closed_form']:.5f}" if "beta_closed_form" in r else ""))
    syn = pd.DataFrame(rows).sort_values(["fmap", "arch", "k", "factor"])
    write_csv_df(out / "affine_synthetic.csv", syn)

    real, inv = [], []
    if not args.skip_real:
        prepared = prepare_all(cfg, cfg["run"]["datasets"], cfg["run"]["seeds"], log)
        rjobs = [{"dataset": d, "seed": s, "fmap": f, "arch": a, "prep_dir": prepared[(d, s)]}
                 for d in cfg["run"]["datasets"] for s in cfg["run"]["seeds"] for f in cfg["run"]["feature_maps"] for a in ARCHS]
        log.info(f"Part B: {len(rjobs)} real-data configurations (cached noisy features only)")
        with ProcessPoolExecutor(int(cfg["run"]["workers"]), mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(real_point, j, cfg): j for j in rjobs}
            for fu in as_completed(futs):
                j = futs[fu]
                try:
                    res = fu.result()
                except Exception as e:
                    log.error(f"  {j['dataset']} {j['fmap']} seed{j['seed']} {j['arch']}: {e}")
                    continue
                if res is None:
                    log.info(f"  {j['dataset']} {j['fmap']} seed{j['seed']} {j['arch']}: noisy features not cached -> skipped")
                    continue
                real.append(res[0])
                inv.extend(res[1])
        if real:
            write_csv_df(out / "affine_real.csv", pd.DataFrame(real).sort_values(["fmap", "arch", "dataset", "seed"]))
            write_csv_df(out / "pipeline_invariance.csv", pd.DataFrame(inv).sort_values(["fmap", "arch", "dataset", "seed", "classifier"]))

    summary = {"synthetic": {}, "real": {}}
    for (f, a), g in syn.groupby(["fmap", "arch"]):
        summary["synthetic"][f"{f}/{a}"] = {"max_abs_residual": float(g["max_abs_residual"].max()),
                                           "beta_at_factor": {str(k): float(v) for k, v in g.groupby("factor")["beta_mean"].mean().items()}}
        if "beta_closed_form" in g and g["beta_closed_form"].notna().any():
            summary["synthetic"][f"{f}/{a}"]["max_abs_beta_minus_closed_form"] = float((g["beta_mean"] - g["beta_closed_form"]).abs().max())
    if inv:
        I = pd.DataFrame(inv)
        for (f, a), g in I.groupby(["fmap", "arch"]):
            summary["real"][f"{f}/{a}"] = {
                "n": int(len(g)), "max_abs_auc_diff": float(g["auc_diff"].abs().max()),
                "mean_abs_auc_diff": float(g["auc_diff"].abs().mean()),
                "mean_abs_auc_diff_roundoff_control": float(g["auc_diff_roundoff_control"].abs().mean()),
                "per_classifier": {c: {"mean_abs_auc_diff": float(h["auc_diff"].abs().mean()),
                                       "mean_abs_auc_diff_roundoff_control": float(h["auc_diff_roundoff_control"].abs().mean()),
                                       "max_abs_score_diff": float(h["max_abs_score_diff"].max())}
                                   for c, h in g.groupby("classifier")}}
        R_ = pd.DataFrame(real)
        for (f, a), g in R_.groupby(["fmap", "arch"]):
            summary["real"][f"{f}/{a}"].update({"affine_max_abs_residual": float(g["max_abs_residual"].max()),
                                               "beta_mean": float(g["beta_mean"].mean())})
    summary["pilot"] = bool(cfg.get("_pilot"))
    write_json(out / "summary.json", summary)
    if samples:
        plot(samples, syn, out)
    log.info(json.dumps(summary, indent=1))


def plot(samples, syn, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fmaps = [f for f in ("Z", "ZZ", "Pauli") if any(k[1] == f for k in samples)]
    fig, axes = plt.subplots(1, len(fmaps) + 1, figsize=(3.0 * (len(fmaps) + 1), 2.8))
    for ax, f in zip(axes, fmaps):
        for j, a in enumerate(ARCHS):
            ks = [k for k in samples if k[0] == a and k[1] == f]
            if ks:
                s = samples[sorted(ks, key=lambda t: t[2])[0]]
                ax.scatter(s["z_ideal"], s["z_noisy"], s=4, alpha=0.6, color=f"C{j}", label=a.replace("_", "-"))
        ax.plot([-1, 1], [-1, 1], "k:", lw=0.8)
        ax.set_title(f"{f} map, baseline noise", fontsize=8)
        ax.set_xlabel(r"ideal $\langle Z_1\rangle$", fontsize=8)
        ax.set_ylabel(r"noisy $\langle Z_1\rangle$", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6)
    ax = axes[-1]
    for j, f in enumerate(fmaps):
        for ls, a in zip(("-", "--"), ARCHS):
            g = syn[(syn["fmap"] == f) & (syn["arch"] == a)].groupby("factor")["max_abs_residual"].max()
            if len(g):
                ax.plot(g.index, np.maximum(g.values, 1e-16), ls, marker="o", ms=3, color=f"C{j}", label=f"{f} {a[:5]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("noise strength factor", fontsize=8)
    ax.set_ylabel("max |affine-fit residual|", fontsize=8)
    ax.set_title("Affine law: Z exact, entangling maps not", fontsize=8)
    ax.legend(fontsize=5)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_affine_law.png", dpi=250)
    fig.savefig(out / "fig_affine_law.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
