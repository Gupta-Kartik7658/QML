"""Phase 2, experiment 3: real-hardware validation on IBM Heron (experiment-venv side).

The IBM part needs Qiskit 2.x + qiskit-ibm-runtime, which must NOT be installed in this venv (pinned
Qiskit 1.3.1). The workflow therefore has three steps:

  1) here (experiment .venv):  python run_ibm_hardware.py prepare
        -> results/<tag>/ibm_hardware/bundle/  (angles, labels, weights, ideal probabilities, circuits as QPY)
  2) IBM venv (see ibm/README.md):  python ibm/ibm_hardware.py submit|fetch ...   (see that file)
        -> results/<tag>/ibm_hardware/hw_<backend>_<jobid>.npz   (measured <Z_i> per row)
  3) here:  python run_ibm_hardware.py analyze
        -> affine-law test on hardware, classifiers trained/tested on hardware features, sim-to-hardware
           transfer with affine calibration, tables + figures.

Rows: a small stratified subset of the main run's paired noisy rows (dataset/seed configurable), the SAME rows
for every feature map, so hardware, ideal and simulated-noisy features are compared row by row.
"""
import qfd.threads  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.config import apply_cli, build_parser, load_config
from qfd.featuremaps import build_circuit, observable_matrices, to_angles
from qfd.phase2 import PARTS, affine_fit, ideal_probs_paired, load_task, noisy_probs, paired_rows
from qfd.preprocess import prepare_all
from qfd.utils import get_logger, write_csv_df, write_json

ARCHS = ("superconducting", "trapped_ion")


# ============================================================================ prepare
def choose_rows(T, caps, seed):
    """Positions inside the paired rows: train (pos/neg caps), val/test (all frauds + neg cap), with IPW weights."""
    rng = np.random.default_rng(seed + 4242)
    R = paired_rows(T)
    meta = T["meta"]
    sel, w = {}, {}
    for p in PARTS:
        y = R["y"][p]
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        npos = min(len(pos), caps[f"{p}_pos"])
        nneg = min(len(neg), caps[f"{p}_neg"])
        idx = np.sort(np.r_[rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)])
        sel[p] = idx
        if p == "train":
            w[p] = np.ones(len(idx))
        else:
            full = meta["partition_full"][p]
            full_neg, full_pos = full["rows"] - full["positives"], full["positives"]
            w[p] = np.where(y[idx] == 1, full_pos / max(npos, 1), full_neg / max(nneg, 1))
    return sel, w


def prepare(cfg, args, out, log):
    from qiskit import qpy, transpile
    ds, seed = args.dataset, args.seed
    prepared = prepare_all(cfg, [ds], [seed], log)
    caps = dict(train_pos=args.train_pos, train_neg=args.train_neg, val_pos=args.val_pos, val_neg=args.val_neg,
                test_pos=args.test_pos, test_neg=args.test_neg)
    bdir = out / "bundle"
    bdir.mkdir(parents=True, exist_ok=True)
    arrays, meta_out, sel = {}, {"dataset": ds, "seed": seed, "feature_maps": {}, "caps": caps}, None
    for fmap in args.fmaps:
        T = load_task(cfg, prepared[(ds, seed)], ds, fmap, seed, log)
        if args.k:  # depth scan: the first k features of the SFS path instead of the selected subset
            sub = T["sfs"]["stages"][args.k - 1]["subset"]
            T["cols"] = [T["names"].index(f) for f in sub]
            T["k"] = len(sub)
            T["sfs"] = dict(T["sfs"], best_subset=list(sub))
        if sel is None:
            sel, w = choose_rows(T, caps, seed)
            R = paired_rows(T)
            for p in PARTS:
                arrays[f"y_{p}"] = R["y"][p][sel[p]].astype(np.int8)
                arrays[f"w_{p}"] = w[p]
                arrays[f"paired_pos_{p}"] = sel[p]
        A = T["A"]
        R = paired_rows(T)
        Pi = ideal_probs_paired(T)
        k = T["k"]
        for p in PARTS:
            U = A[f"Xq_{p}"][R["idx"][p][sel[p]]][:, T["cols"]]
            arrays[f"{fmap}_angles_{p}"] = to_angles(U, T["fm"])
            arrays[f"{fmap}_ideal_probs_{p}"] = Pi[p][sel[p]]
        for arch in ARCHS:
            Pn = noisy_probs(cfg, T, arch, allow_compute=False)
            if Pn is not None:
                for p in PARTS:
                    arrays[f"{fmap}_sim_{arch}_probs_{p}"] = np.asarray(Pn[p])[sel[p]]
            elif args.simulate_missing:  # e.g. depth-scan subsets: simulate only the bundle rows
                from qfd.noisy import noisy_probabilities
                for p in PARTS:
                    U = A[f"Xq_{p}"][R["idx"][p][sel[p]]][:, T["cols"]]
                    arrays[f"{fmap}_sim_{arch}_probs_{p}"] = noisy_probabilities(U, T["fm"], arch, cfg["architectures"][arch], cfg["noise"])[0]
                log.info(f"{fmap}: simulated {arch} noise for the bundle rows")
        logical = transpile(build_circuit(T["fm"], k), basis_gates=["h", "p", "cx", "rz", "sx", "x", "ry", "rx"],
                            optimization_level=0)
        with open(bdir / f"circuit_{fmap}.qpy", "wb") as f:
            qpy.dump(logical, f)
        meta_out["feature_maps"][fmap] = {"k": k, "selected": T["sfs"]["best_subset"], "fmap_params": T["fm"],
                                          "logical_ops": {str(a): int(b) for a, b in logical.count_ops().items()}}
        log.info(f"{fmap}: k={k} selected={T['sfs']['best_subset']} logical ops {dict(logical.count_ops())}")
    n = {p: int(len(arrays[f"y_{p}"])) for p in PARTS}
    meta_out["rows"] = n
    meta_out["frauds"] = {p: int(arrays[f"y_{p}"].sum()) for p in PARTS}
    np.savez_compressed(bdir / "bundle.npz", **arrays)
    write_json(bdir / "meta.json", meta_out)
    log.info(f"bundle written to {bdir}: rows {n}, frauds {meta_out['frauds']} -> copy nothing, the IBM script reads it in place")


# ============================================================================ analyze
def analyze(cfg, args, out, log):
    from qfd.classifiers import scores, tune_and_fit
    from qfd.metrics import bootstrap_ci, choose_threshold, full_metrics
    from qfd.noisy import sample_shots
    bdir = out / "bundle"
    B = dict(np.load(bdir / "bundle.npz"))
    meta = json.loads((bdir / "meta.json").read_text())
    hw_files = sorted(out.glob("hw_*.npz")) if not args.hw_file else [Path(args.hw_file)]
    if not hw_files:
        raise SystemExit(f"no hardware result files hw_*.npz in {out}")
    y = {p: B[f"y_{p}"] for p in PARTS}
    w = {p: B[f"w_{p}"] for p in PARTS}
    rows_fit, rows_clf, rows_q, rows_shots = [], [], [], []
    scatter = {}
    for hf in hw_files:
        H = dict(np.load(hf, allow_pickle=True))
        hmeta = json.loads(str(H["meta_json"]))
        backend = hmeta["backend"]
        for fmap in meta["feature_maps"]:
            if f"{fmap}_z_train" not in H:
                continue
            k = meta["feature_maps"][fmap]["k"]
            M, _ = observable_matrices(k, False)
            shots = int(hmeta["shots"])
            reps = {"ideal (exact)": {p: B[f"{fmap}_ideal_probs_{p}"] @ M for p in PARTS},
                    f"ideal + {shots} shots (sim)": {p: sample_shots(B[f"{fmap}_ideal_probs_{p}"], shots, 11 + i) @ M
                                                     for i, p in enumerate(PARTS)}}
            for arch in ARCHS:
                if f"{fmap}_sim_{arch}_probs_train" in B:
                    reps[f"{arch} sim + {shots} shots"] = {p: sample_shots(B[f"{fmap}_sim_{arch}_probs_{p}"], shots, 21 + i) @ M
                                                           for i, p in enumerate(PARTS)}
            hw = {p: H[f"{fmap}_z_{p}"].astype(float) for p in PARTS}
            keep = {p: np.isfinite(hw[p]).all(1) for p in PARTS}  # rehearsal runs may measure only some rows
            if not all(keep[p].all() for p in PARTS):
                log.warning(f"{hf.name} {fmap}: only {sum(int(keep[p].sum()) for p in PARTS)} rows measured -> using those")
            hw = {p: hw[p][keep[p]] for p in PARTS}
            reps = {n_: {p: F[p][keep[p]] for p in PARTS} for n_, F in reps.items()}
            yk = {p: y[p][keep[p]] for p in PARTS}
            wk = {p: w[p][keep[p]] for p in PARTS}
            if any(len(np.unique(yk[p])) < 2 for p in PARTS):
                log.warning(f"{hf.name} {fmap}: a partition has a single class -> classifier part skipped")
            reps[f"hardware {backend}"] = hw
            # ---- affine law on hardware (fit on ALL rows; report per feature)
            zi = np.concatenate([reps["ideal (exact)"][p] for p in PARTS])
            zh = np.concatenate([hw[p] for p in PARTS])
            refs = {n_: affine_fit(zi, np.concatenate([F[p] for p in PARTS])) for n_, F in reps.items()
                    if "shots" in n_}  # same fit on simulated data: what shot noise (+ sim noise) alone gives
            for f in affine_fit(zi, zh):
                j = f["feature"]
                sd_shot = float(np.sqrt(np.mean(1 - zh[:, j] ** 2) / shots))
                rows_fit.append({"backend": backend, "fmap": fmap, **f,
                                 "feature_name": meta["feature_maps"][fmap]["selected"][j],
                                 "shot_noise_sd_expected": sd_shot,
                                 "residual_sd_over_shot_sd": f["residual_sd"] / sd_shot if sd_shot > 0 else float("nan"),
                                 **{f"r2_reference[{n_}]": rf[j]["r2"] for n_, rf in refs.items()}})
            scatter[(backend, fmap)] = (zi[:, 0], zh[:, 0])
            # ---- per PHYSICAL qubit: the proposition is a per-qubit law (each copy sits on different qubits)
            pl = hmeta["plan"].get(fmap, {})
            if pl and all(pl["rows_per_part"][p] == pl["full_rows_per_part"][p] for p in PARTS) and all(keep[p].all() for p in PARTS):
                cp = np.arange(len(zi)) % pl["copies"]
                phys = np.array(pl["physical_qubits"]).reshape(pl["copies"], k)
                for c in range(pl["copies"]):
                    m = cp == c
                    for f in affine_fit(zi[m], zh[m]):
                        j = f["feature"]
                        sd = float(np.sqrt(np.mean(1 - zh[m, j] ** 2) / shots))
                        rows_q.append({"backend": backend, "fmap": fmap, "copy": c, "feature": j,
                                       "feature_name": meta["feature_maps"][fmap]["selected"][j],
                                       "physical_qubit": int(phys[c, j]), "n_rows": int(m.sum()), **{a: b for a, b in f.items() if a != "feature"},
                                       "shot_noise_sd_expected": sd, "residual_sd_over_shot_sd": f["residual_sd"] / sd if sd > 0 else np.nan})
            # ---- hardware shot budget: re-use the first S of the recorded shots (no extra QPU time)
            if f"{fmap}_raw_bits" in H and pl and all(keep[p].all() for p in PARTS) and \
                    all(pl["rows_per_part"][p] == pl["full_rows_per_part"][p] for p in PARTS):
                raw = H[f"{fmap}_raw_bits"]
                nb = pl["copies"] * k
                bits_all = np.unpackbits(raw, axis=-1)[..., ::-1][..., :nb]  # (n_exec, shots, nbits)
                bnd = np.cumsum([0] + [len(y[p]) for p in PARTS])
                for S in [s_ for s_ in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if s_ <= bits_all.shape[1]]:
                    zS = (1.0 - 2.0 * bits_all[:, :S].astype(np.float64)).mean(axis=1)
                    zS = zS.reshape(zS.shape[0], pl["copies"], k).reshape(-1, k)[:bnd[-1]]
                    for src, F in (("hardware", {p: zS[bnd[i]:bnd[i + 1]] for i, p in enumerate(PARTS)}),
                                   ("ideal sim", {p: sample_shots(B[f"{fmap}_ideal_probs_{p}"], S, 31 + S + i) @ M
                                                  for i, p in enumerate(PARTS)})):
                        m, hp, _, _ = tune_and_fit("xgb", cfg["classifiers"]["xgb"], F["train"], y["train"], F["val"],
                                                   y["val"], w["val"], args.seed, 1)
                        thr = choose_threshold(y["val"], scores("xgb", m, F["val"]), w["val"])
                        mt = full_metrics(y["test"], scores("xgb", m, F["test"]), thr, w["test"])
                        rows_shots.append({"backend": backend, "fmap": fmap, "k": k, "shots": S, "source": src,
                                           "test_roc_auc": mt["roc_auc"], "test_avg_precision": mt["avg_precision"]})
            # ---- sim->hardware transfer: train on ideal exact, calibrate (alpha, beta) on TRAIN rows only
            cal = affine_fit(reps["ideal (exact)"]["train"], hw["train"])
            a_ = np.array([c["alpha"] for c in cal])
            b_ = np.array([c["beta"] for c in cal])
            b_ = np.where(np.abs(b_) < 1e-3, 1e-3, b_)
            reps["transfer: ideal-trained, hw calibrated"] = {p: (hw[p] - a_) / b_ for p in PARTS}
            for name, F in reps.items():
                if any(len(np.unique(yk[p])) < 2 for p in PARTS):
                    break
                for clf in args.classifiers:
                    if name.startswith("transfer"):
                        Ftr, Fva = reps["ideal (exact)"]["train"], reps["ideal (exact)"]["val"]
                    else:
                        Ftr, Fva = F["train"], F["val"]
                    m, hp, _, _ = tune_and_fit(clf, cfg["classifiers"][clf], Ftr, yk["train"], Fva, yk["val"], wk["val"], args.seed, 1)
                    thr = choose_threshold(yk["val"], scores(clf, m, F["val"]), wk["val"])
                    s_te = scores(clf, m, F["test"])
                    mt = full_metrics(yk["test"], s_te, thr, wk["test"])
                    ci = bootstrap_ci(yk["test"], s_te, wk["test"], 1000, seed=args.seed)
                    rows_clf.append({"backend": backend, "fmap": fmap, "representation": name, "classifier": clf,
                                     "n_test": int(len(yk["test"])), "best_hp": str(hp), **{f"test_{a}": b for a, b in mt.items()},
                                     **{f"test_{a}": b for a, b in ci.items()}})
                    log.info(f"{backend} {fmap:5s} {name:40s} {clf}: AUC {mt['roc_auc']:.4f} "
                             f"[{ci['roc_auc_ci_low']:.3f},{ci['roc_auc_ci_high']:.3f}] AP {mt['avg_precision']:.4f}")
    fit = pd.DataFrame(rows_fit)
    clf_df = pd.DataFrame(rows_clf)
    write_csv_df(out / "hardware_affine_fit.csv", fit)
    write_csv_df(out / "hardware_classifiers.csv", clf_df)
    summ = fit.groupby(["backend", "fmap"]).agg(beta_mean=("beta", "mean"), r2_min=("r2", "min"), r2_mean=("r2", "mean"),
                                                 max_abs_residual=("max_abs_residual", "max"),
                                                 resid_sd_over_shot_sd=("residual_sd_over_shot_sd", "mean")).reset_index()
    write_csv_df(out / "hardware_affine_summary.csv", summ)
    if rows_shots:
        SB = pd.DataFrame(rows_shots)
        write_csv_df(out / "hardware_shot_budget.csv", SB)
        log.info("hardware shot budget (xgb test AUC):\n" +
                 SB.pivot_table(index=["fmap", "shots"], columns="source", values="test_roc_auc").round(4).to_string())
    qsum = None
    if rows_q:
        Q = pd.DataFrame(rows_q)
        write_csv_df(out / "hardware_affine_per_qubit.csv", Q)
        qsum = Q.groupby(["backend", "fmap"]).agg(
            n_qubits=("physical_qubit", "size"), beta_median=("beta", "median"), beta_min=("beta", "min"),
            beta_max=("beta", "max"), alpha_min=("alpha", "min"), alpha_max=("alpha", "max"), r2_median=("r2", "median"),
            r2_min=("r2", "min"), resid_over_shot_median=("residual_sd_over_shot_sd", "median"),
            resid_over_shot_p90=("residual_sd_over_shot_sd", lambda v: float(np.percentile(v, 90))),
            resid_over_shot_max=("residual_sd_over_shot_sd", "max")).reset_index()
        write_csv_df(out / "hardware_affine_per_qubit_summary.csv", qsum)
        log.info("per-physical-qubit affine law:\n" + qsum.to_string(index=False))
        plot_qubits(Q, out)
    write_json(out / "summary.json", {"affine": summ.to_dict("records"),
                                     "affine_per_qubit": qsum.to_dict("records") if qsum is not None else None,
                                     "classifiers": clf_df[["backend", "fmap", "representation", "classifier", "test_roc_auc",
                                                            "test_avg_precision"]].to_dict("records")})
    plot(scatter, out)
    log.info("\n" + summ.to_string(index=False))


def plot_qubits(Q, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (backend, fmap), g in Q.groupby(["backend", "fmap"]):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
        axes[0].hist(g["beta"], bins=20, color="C0")
        axes[0].set_xlabel(r"fitted $\beta$ per physical qubit", fontsize=8)
        axes[0].set_ylabel("qubits", fontsize=8)
        axes[1].hist(g["residual_sd_over_shot_sd"], bins=20, color="C1")
        axes[1].axvline(1.0, color="k", ls=":", lw=0.8)
        axes[1].set_xlabel("affine-fit residual s.d. / shot-noise s.d.", fontsize=8)
        for ax in axes:
            ax.tick_params(labelsize=7)
        fig.suptitle(f"{fmap} map on {backend}: affine law per physical qubit ({len(g)} qubits)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"fig_hardware_per_qubit_{fmap}.png", dpi=250)
        fig.savefig(out / f"fig_hardware_per_qubit_{fmap}.pdf")
        plt.close(fig)


def plot(scatter, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not scatter:
        return
    fig, axes = plt.subplots(1, len(scatter), figsize=(3.0 * len(scatter), 2.8), squeeze=False)
    for ax, ((backend, fmap), (zi, zh)) in zip(axes[0], scatter.items()):
        ax.scatter(zi, zh, s=3, alpha=0.5)
        c = np.polyfit(zi, zh, 1)
        xs = np.linspace(-1, 1, 50)
        ax.plot(xs, np.polyval(c, xs), "r-", lw=1, label=f"affine fit: {c[1]:+.3f} + {c[0]:.3f} z")
        ax.plot([-1, 1], [-1, 1], "k:", lw=0.8)
        ax.set_title(f"{fmap} map on {backend} (qubit 1)", fontsize=8)
        ax.set_xlabel(r"ideal $\langle Z_1\rangle$", fontsize=8)
        ax.set_ylabel(r"hardware $\langle Z_1\rangle$", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_hardware_affine.png", dpi=250)
    fig.savefig(out / "fig_hardware_affine.pdf")
    plt.close(fig)


def main():
    p = build_parser(__doc__)
    p.add_argument("step", choices=["prepare", "analyze"])
    p.add_argument("--dataset", default="creditcard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fmaps", nargs="+", default=["Z", "ZZ"])
    p.add_argument("--train-pos", type=int, default=250)
    p.add_argument("--train-neg", type=int, default=550)
    p.add_argument("--val-neg", type=int, default=300)
    p.add_argument("--test-neg", type=int, default=400)
    p.add_argument("--classifiers", nargs="+", default=["xgb", "svm", "mlp"])
    p.add_argument("--hw-file", default=None)
    p.add_argument("--subdir", default="", help="separate experiment folder under ibm_hardware/ (e.g. paysim_Z)")
    p.add_argument("--k", type=int, default=0, help="use the first k features of the SFS path (depth scan)")
    p.add_argument("--val-pos", type=int, default=10**6)
    p.add_argument("--test-pos", type=int, default=10**6)
    p.add_argument("--simulate-missing", action="store_true", help="simulate noisy features for the bundle rows if not cached")
    args = p.parse_args()
    cfg = apply_cli(load_config(args.config), args)
    out = Path(cfg["paths"]["results_dir"]) / cfg["run"]["run_tag"] / "ibm_hardware"
    if args.subdir:
        out = out / args.subdir
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("ibm_hardware", out / "run.log")
    if args.step == "prepare":
        prepare(cfg, args, out, log)
    else:
        analyze(cfg, args, out, log)


if __name__ == "__main__":
    main()
