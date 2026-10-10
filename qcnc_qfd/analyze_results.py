"""Aggregate the results of all experiment scripts into paper tables and figures.

If the two architecture configurations were run on different computers, first copy the
results/<tag>/<classifier>_trapped_ion/ folders into the same results/<tag>/ folder, then:

    python analyze_results.py                      # results/main -> results/main/_paper
    python analyze_results.py --run-tag pilot

Nothing is re-simulated: everything is read from the saved results, curves and logs.
Missing experiments are reported as missing, never filled in.
"""
import qfd.threads  # noqa: F401

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qfd.config import ROOT  # noqa: E402
from qfd.utils import load_npz, read_json, write_csv_df, write_json  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight", "font.family": "serif"})
CLF_ORDER = ["mlp", "xgb", "svm"]
CLF_NAME = {"mlp": "MLP", "xgb": "XGBoost", "svm": "SVM (RBF)"}
FM_ORDER = ["Z", "ZZ", "Pauli"]
DS_ORDER = ["creditcard", "paysim", "banksim"]
DS_NAME = {"creditcard": "European credit card (real)", "paysim": "PaySim (synthetic)", "banksim": "BankSim (synthetic)"}
REP_NAME = {"c_all": "Classical, all features", "c_selected": "Classical, SFS-selected",
            "q_ideal_z": "Quantum <Z> (ideal)", "q_ideal_zzz": "Quantum <Z>,<ZZ> (ideal)",
            "q_ideal_mi_z": "Quantum <Z>, MI top-k", "q_ideal_paired_z": "Ideal (exact)",
            "q_ideal_shots_z": "Ideal + shots", "q_noisy_z": "Noisy (exact)", "q_noisy_shots_z": "Noisy + shots"}
IDEAL_REPS = ["c_all", "c_selected", "q_ideal_z", "q_ideal_zzz", "q_ideal_mi_z"]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def ms(series, digits=4):
    s = series.dropna()
    if s.empty:
        return "--"
    if len(s) == 1:
        return f"{s.iloc[0]:.{digits}f}"
    return f"{s.mean():.{digits}f} $\\pm$ {s.std():.{digits}f}"


def to_latex(df, path, caption, label):
    """Minimal booktabs-free IEEE-friendly LaTeX table (no jinja2 dependency)."""
    def cell(v):
        if isinstance(v, float):
            return "--" if np.isnan(v) else f"{v:.4g}"
        return str(v).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    cols = [cell(c) for c in df.columns]
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\caption{" + caption + "}",
             "\\label{" + label + "}", "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline", " & ".join(cols) + " \\\\", "\\hline"]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(cell(v) if not (isinstance(v, str) and "$" in v) else v for v in row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}}", "\\end{table}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ============================================================================ load
def load_all(run_dir: Path):
    frames = []
    main_dirs = {f"{c}_{a}" for c in ("mlp", "xgb", "svm") for a in ("superconducting", "trapped_ion")}
    for f in sorted(run_dir.glob("*_*/results.csv")):
        if f.parent.name not in main_dirs:   # skip _paper, sfs_vs_exhaustive, noise_sweep, qsvm_reference
            continue
        d = pd.read_csv(f)
        d["source_dir"] = f.parent.name
        frames.append(d)
    if not frames:
        raise SystemExit(f"no results.csv found under {run_dir}")
    df = pd.concat(frames, ignore_index=True)
    failed = df[df["status"] != "completed"] if "status" in df else df.iloc[:0]
    df = df[df["status"] == "completed"].copy()
    return df, failed


def dedupe(df):
    """Architecture-independent rows appear in both architecture scripts -> keep one, check equality."""
    ideal = df[df["representation"].isin(IDEAL_REPS)]
    other = df[~df["representation"].isin(IDEAL_REPS)]
    keys = ["classifier", "dataset", "feature_map", "seed", "representation"]
    chk = ideal.groupby(keys).agg(n=("test_roc_auc", "size"), auc_min=("test_roc_auc", "min"),
                                  auc_max=("test_roc_auc", "max")).reset_index()
    chk["abs_diff"] = chk["auc_max"] - chk["auc_min"]
    ideal = ideal.sort_values("arch_script").drop_duplicates(keys, keep="first")
    # paired ideal rows: identical rows in both scripts too
    pk = keys
    pair = other[other["representation"].isin(["q_ideal_paired_z", "q_ideal_shots_z"])]
    pair = pair.sort_values("arch_script").drop_duplicates(pk, keep="first")
    noisy = other[~other["representation"].isin(["q_ideal_paired_z", "q_ideal_shots_z"])]
    return pd.concat([ideal, pair, noisy], ignore_index=True), chk


# ============================================================================ tables
def table_datasets(cache_dir: Path, out: Path, datasets):
    rows = []
    for d in datasets:
        metas = sorted((cache_dir / "prepared" / d).glob("seed0_*/meta.json"), key=os.path.getmtime)
        if not metas:
            continue
        m = read_json(metas[-1])
        pf = m["partition_full"]
        rows.append({"Dataset": DS_NAME.get(d, d), "Type": m.get("data_type", ""), "Rows (raw)": m["rows_raw"],
                     "Duplicates removed": m["duplicate_rows_dropped"], "Rows used": m["rows"],
                     "Frauds": m["positives"], "Prevalence (%)": round(100 * m["prevalence"], 3),
                     "Candidate features": m["n_candidate_features"],
                     "Train / val / test frauds": f"{pf['train']['positives']} / {pf['val']['positives']} / {pf['test']['positives']}",
                     "Test rows": pf["test"]["rows"], "Split": m["split_mode"],
                     "Schema matches expected": m["schema_check"]["matches"]})
    if rows:
        t = pd.DataFrame(rows)
        write_csv_df(out / "table_datasets.csv", t)
        to_latex(t, out / "table_datasets.tex", "Datasets after cleaning (seed-0 split). Test partitions keep the original fraud prevalence.", "tab:datasets")


def table_main(df, out):
    """Classical vs quantum-derived representations, all classifiers, mean +- std over seeds."""
    reps = ["c_all", "c_selected", "q_ideal_z", "q_ideal_zzz", "q_ideal_mi_z"]
    sub = df[df["representation"].isin(reps)]
    long = (sub.groupby(["dataset", "representation", "feature_map", "classifier"])
            .agg(**{f"{m}_{s}": (f"test_{m}", s) for m in ("roc_auc", "avg_precision", "f1", "recall", "precision",
                                                         "fpr", "mcc", "balanced_accuracy") for s in ("mean", "std")},
                 val_roc_auc_mean=("val_roc_auc", "mean"), n_seeds=("seed", "nunique"),
                 repr_dim=("repr_dim", "median"), n_selected=("n_selected", "median")).reset_index())
    write_csv_df(out / "table_main_long.csv", long)
    for d in [x for x in DS_ORDER if x in set(sub["dataset"])]:
        rows = []
        s = sub[sub["dataset"] == d]
        for rep in reps:
            fms = ["classical"] if rep == "c_all" else FM_ORDER
            for fm in fms:
                r = s[(s["representation"] == rep) & (s["feature_map"] == fm)]
                if r.empty:
                    continue
                row = {"Representation": REP_NAME[rep] + ("" if rep == "c_all" else f" [{fm}]")}
                for c in CLF_ORDER:
                    rc = r[r["classifier"] == c]
                    row[f"{CLF_NAME[c]} AUC"] = ms(rc["test_roc_auc"], 3)
                    row[f"{CLF_NAME[c]} AP"] = ms(rc["test_avg_precision"], 3)
                rows.append(row)
        t = pd.DataFrame(rows)
        write_csv_df(out / f"table_main_{d}.csv", t)
        to_latex(t, out / f"table_main_{d}.tex",
                 f"{DS_NAME[d]}: test ROC-AUC and average precision (mean $\\pm$ s.d. over seeds) at the original fraud prevalence. "
                 "[Z]/[ZZ]/[Pauli] = feature map used for SFS and encoding; classical-selected uses the SFS subset of that map.",
                 f"tab:main_{d}")


def table_noise(df, out):
    reps = ["q_ideal_paired_z", "q_ideal_shots_z", "q_noisy_z", "q_noisy_shots_z"]
    sub = df[df["representation"].isin(reps)].copy()
    if sub.empty:
        return
    sub["condition"] = sub.apply(lambda r: REP_NAME[r.representation] + ("" if "ideal" in r.representation
                                                                         else f" ({r.architecture})"), axis=1)
    long = (sub.groupby(["dataset", "feature_map", "classifier", "condition"])
            .agg(auc_mean=("test_roc_auc", "mean"), auc_std=("test_roc_auc", "std"),
                 ap_mean=("test_avg_precision", "mean"), ap_std=("test_avg_precision", "std"),
                 n=("seed", "nunique"), feature_mae=("noisy_feature_mae", "mean"),
                 tvd=("noisy_tvd_mean", "mean")).reset_index())
    write_csv_df(out / "table_noise_long.csv", long)
    conds = ["Ideal (exact)", "Ideal + shots", "Noisy (exact) (superconducting)", "Noisy + shots (superconducting)",
             "Noisy (exact) (trapped_ion)", "Noisy + shots (trapped_ion)"]
    rows = []
    for d in [x for x in DS_ORDER if x in set(sub["dataset"])]:
        for fm in FM_ORDER:
            for c in CLF_ORDER:
                r = sub[(sub["dataset"] == d) & (sub["feature_map"] == fm) & (sub["classifier"] == c)]
                if r.empty:
                    continue
                row = {"Dataset": d, "Map": fm, "Classifier": CLF_NAME[c]}
                for cond in conds:
                    row[cond.replace("superconducting", "SC").replace("trapped_ion", "TI")] = ms(r[r["condition"] == cond]["test_roc_auc"], 3)
                rows.append(row)
    t = pd.DataFrame(rows)
    write_csv_df(out / "table_noise.csv", t)
    to_latex(t, out / "table_noise.tex", "Ideal vs architecture-specific noisy simulation (test ROC-AUC, paired rows: "
             "identical training/validation/test samples in every column). Shots = finite-shot estimation of $\\langle Z_i\\rangle$.",
             "tab:noise")


def table_circuits(df, out):
    cols = ["feature_map", "n_selected", "arch_script", "logical_depth", "logical_2q", "compiled_depth",
            "compiled_2q", "swap_overhead_2q"]
    if not set(cols).issubset(df.columns):
        return
    t = (df.dropna(subset=["compiled_depth"])[cols].drop_duplicates()
         .sort_values(["feature_map", "n_selected", "arch_script"]))
    write_csv_df(out / "table_circuits.csv", t)
    to_latex(t.rename(columns={"arch_script": "architecture", "n_selected": "qubits"}), out / "table_circuits.tex",
             "Feature-map circuits before (logical, \\{u,cx\\}) and after compilation to each architecture configuration.",
             "tab:circuits")


def table_sfs(run_dir, out):
    rows = []
    for f in run_dir.glob("*_*/tasks/*/sfs_summary.json"):
        s = read_json(f)
        d, fm, seed = f.parent.name.split("__")
        rows.append({"dataset": d, "feature_map": fm, "seed": int(seed.replace("seed", "")),
                     "selected": ";".join(s["best_subset"]), "k": s["best_k"], "val_roc_auc": s["best_val_roc_auc"],
                     "val_avg_precision": s["best_val_avg_precision"], "evaluations": s["n_evaluations"],
                     "max_evaluations": s["max_possible_evaluations"], "stopping": s["stopping_reason"],
                     "runtime_s": s["runtime_seconds"], "evaluator": s["evaluator"],
                     "path": " > ".join(st["added"] for st in s["stages"])})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    keys = ["dataset", "feature_map", "seed"]
    consistent = t.groupby(keys)["selected"].nunique().max() == 1
    t = t.sort_values(keys).drop_duplicates(keys)
    write_csv_df(out / "table_sfs.csv", t)
    to_latex(t[["dataset", "feature_map", "seed", "selected", "k", "val_roc_auc", "evaluations", "stopping", "runtime_s"]]
             .round(4), out / "table_sfs.tex", "Sequential forward selection results (validation data only).", "tab:sfs")
    freq = (t.assign(f=t["selected"].str.split(";")).explode("f").groupby(["dataset", "f"]).size()
            .rename("times_selected").reset_index().sort_values(["dataset", "times_selected"], ascending=[True, False]))
    write_csv_df(out / "table_sfs_feature_frequency.csv", freq)
    return consistent


def table_runtime(df, out):
    agg = {}
    for c in ("sfs_runtime_seconds", "time_ideal_features", "time_noisy_features", "train_seconds",
              "quantum_extract_seconds", "peak_memory_mb", "task_seconds"):
        if c in df:
            agg[c] = (c, "mean")
    if not agg:
        return
    t = df.groupby(["dataset", "feature_map", "classifier", "arch_script"]).agg(**agg).reset_index()
    write_csv_df(out / "table_runtime.csv", t.round(2))


def table_paired_differences(df, out):
    """Quantum minus classical (same SFS subset, same rows): per seed differences, sign counts."""
    a = df[df["representation"] == "q_ideal_z"]
    b = df[df["representation"] == "c_selected"]
    k = ["dataset", "feature_map", "classifier", "seed"]
    m = a.merge(b, on=k, suffixes=("_q", "_c"))
    if m.empty:
        return
    for met in ("roc_auc", "avg_precision"):
        m[f"diff_{met}"] = m[f"test_{met}_q"] - m[f"test_{met}_c"]
    g = m.groupby(["dataset", "feature_map", "classifier"]).agg(
        auc_diff_mean=("diff_roc_auc", "mean"), auc_diff_std=("diff_roc_auc", "std"),
        ap_diff_mean=("diff_avg_precision", "mean"), ap_diff_std=("diff_avg_precision", "std"),
        n=("seed", "nunique"), quantum_better_auc=("diff_roc_auc", lambda x: int((x > 0).sum()))).reset_index()
    write_csv_df(out / "table_quantum_minus_classical.csv", g.round(5))


def best_configs(df, out):
    """Predefined criterion: highest mean VALIDATION ROC-AUC (never test) per dataset."""
    reps = ["c_all", "c_selected", "q_ideal_z", "q_ideal_zzz", "q_ideal_mi_z"]
    s = df[df["representation"].isin(reps)]
    g = s.groupby(["dataset", "representation", "feature_map", "classifier"]).agg(
        val_auc=("val_roc_auc", "mean"), test_auc=("test_roc_auc", "mean"), test_auc_sd=("test_roc_auc", "std"),
        test_ap=("test_avg_precision", "mean"), test_f1=("test_f1", "mean"), test_recall=("test_recall", "mean"),
        test_fpr=("test_fpr", "mean"), test_mcc=("test_mcc", "mean")).reset_index()
    best = g.loc[g.groupby("dataset")["val_auc"].idxmax()]
    bestq = g[g["representation"].str.startswith("q_")]
    bestq = bestq.loc[bestq.groupby("dataset")["val_auc"].idxmax()]
    write_csv_df(out / "best_configuration_by_validation.csv",
                 pd.concat([best.assign(scope="overall"), bestq.assign(scope="quantum only")]).round(5))


# ============================================================================ figures
def fig_main_bars(df, out):
    reps = [("c_all", "classical"), ("c_selected", None), ("q_ideal_z", None)]
    for metric, lab in (("test_roc_auc", "Test ROC-AUC"), ("test_avg_precision", "Test average precision")):
        dss = [d for d in DS_ORDER if d in set(df["dataset"])]
        fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 2.8), squeeze=False)
        for ax, d in zip(axes[0], dss):
            s = df[df["dataset"] == d]
            groups = [("Classical (all)", s[s["representation"] == "c_all"])]
            for fm in FM_ORDER:
                groups.append((f"Classical sel. [{fm}]", s[(s["representation"] == "c_selected") & (s["feature_map"] == fm)]))
            for fm in FM_ORDER:
                groups.append((f"Quantum <Z> [{fm}]", s[(s["representation"] == "q_ideal_z") & (s["feature_map"] == fm)]))
            x = np.arange(len(CLF_ORDER))
            w = 0.8 / len(groups)
            for i, (name, g) in enumerate(groups):
                mu = [g[g["classifier"] == c][metric].mean() for c in CLF_ORDER]
                sd = [g[g["classifier"] == c][metric].std() for c in CLF_ORDER]
                hatch = "//" if name.startswith("Quantum") else None
                ax.bar(x - 0.4 + w * (i + 0.5), mu, w, yerr=np.nan_to_num(sd), label=name, hatch=hatch,
                       color=f"C{i}", edgecolor="black", linewidth=0.3, capsize=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels([CLF_NAME[c] for c in CLF_ORDER])
            vals = s[s["representation"].isin(["c_all", "c_selected", "q_ideal_z"])][metric].dropna()
            if not vals.empty:
                ax.set_ylim(max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.02))
            ax.set_title(DS_NAME[d])
            ax.set_ylabel(lab)
        axes[0, -1].legend(fontsize=6, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        save(fig, out, f"fig_quantum_vs_classical_{metric.replace('test_', '')}")


def fig_fmap_heatmap(df, out):
    s = df[df["representation"] == "q_ideal_z"]
    dss = [d for d in DS_ORDER if d in set(s["dataset"])]
    if not dss:
        return
    fig, axes = plt.subplots(1, len(dss), figsize=(3.0 * len(dss), 2.4), squeeze=False)
    for ax, d in zip(axes[0], dss):
        M = np.array([[s[(s["dataset"] == d) & (s["feature_map"] == fm) & (s["classifier"] == c)]["test_roc_auc"].mean()
                       for c in CLF_ORDER] for fm in FM_ORDER])
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, "--" if np.isnan(M[i, j]) else f"{M[i, j]:.3f}", ha="center", va="center",
                        color="white", fontsize=7)
        ax.set_xticks(range(3))
        ax.set_xticklabels([CLF_NAME[c] for c in CLF_ORDER], fontsize=7)
        ax.set_yticks(range(3))
        ax.set_yticklabels(FM_ORDER)
        ax.grid(False)
        ax.set_title(DS_NAME[d], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Test ROC-AUC of <Z> features (ideal), mean over seeds", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig_featuremap_classifier_heatmap")


def fig_noise(df, out):
    reps = ["q_ideal_paired_z", "q_ideal_shots_z", "q_noisy_z", "q_noisy_shots_z"]
    s = df[df["representation"].isin(reps)]
    if s.empty:
        return
    conds = [("q_ideal_paired_z", "ideal", "Ideal"), ("q_ideal_shots_z", "ideal", "Ideal+shots"),
             ("q_noisy_z", "superconducting", "SC noisy"), ("q_noisy_shots_z", "superconducting", "SC noisy+shots"),
             ("q_noisy_z", "trapped_ion", "TI noisy"), ("q_noisy_shots_z", "trapped_ion", "TI noisy+shots")]
    dss = [d for d in DS_ORDER if d in set(s["dataset"])]
    fig, axes = plt.subplots(len(dss), 3, figsize=(10, 2.5 * len(dss)), squeeze=False)
    for i, d in enumerate(dss):
        for j, c in enumerate(CLF_ORDER):
            ax = axes[i, j]
            x = np.arange(len(FM_ORDER))
            w = 0.8 / len(conds)
            for k, (rep, arch, lab) in enumerate(conds):
                g = s[(s["dataset"] == d) & (s["classifier"] == c) & (s["representation"] == rep)]
                if arch != "ideal":
                    g = g[g["architecture"] == arch]
                mu = [g[g["feature_map"] == fm]["test_roc_auc"].mean() for fm in FM_ORDER]
                sd = [g[g["feature_map"] == fm]["test_roc_auc"].std() for fm in FM_ORDER]
                ax.bar(x - 0.4 + w * (k + 0.5), mu, w, yerr=np.nan_to_num(sd), label=lab, color=f"C{k}",
                       edgecolor="black", linewidth=0.3, capsize=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(FM_ORDER)
            v = s[(s["dataset"] == d)]["test_roc_auc"].dropna()
            if not v.empty:
                ax.set_ylim(max(0, v.min() - 0.05), min(1, v.max() + 0.02))
            ax.set_title(f"{DS_NAME[d]} - {CLF_NAME[c]}", fontsize=7)
            ax.set_ylabel("Test ROC-AUC")
    axes[0, -1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig_ideal_vs_noisy")


def fig_curves(run_dir, df, out, clf="xgb", seed=0):
    dss = [d for d in DS_ORDER if d in set(df["dataset"])]
    if not dss:
        return
    fig, axes = plt.subplots(2, len(dss), figsize=(3.3 * len(dss), 6), squeeze=False)
    sdir = run_dir / f"{clf}_superconducting" / "tasks"
    if not sdir.exists():
        sdir = run_dir / f"{clf}_trapped_ion" / "tasks"
    for j, d in enumerate(dss):
        items = [(f"{d}__classical__seed{seed}", "c_all", "Classical (all)")]
        items += [(f"{d}__{fm}__seed{seed}", "q_ideal_z", f"Quantum <Z> [{fm}]") for fm in FM_ORDER]
        items += [(f"{d}__ZZ__seed{seed}", "c_selected", "Classical sel. [ZZ]")]
        for task, rep, lab in items:
            f = sdir / task / rep / "curves.npz"
            if not f.exists():
                continue
            c = load_npz(f)
            mt = read_json(sdir / task / rep / "metrics.json")
            axes[0, j].plot(c["test_roc_fpr"], c["test_roc_tpr"], label=f"{lab} ({mt['test_roc_auc']:.3f})", lw=1)
            axes[1, j].plot(c["test_pr_recall"], c["test_pr_precision"], label=f"{lab} ({mt['test_avg_precision']:.3f})", lw=1)
        axes[0, j].plot([0, 1], [0, 1], ":", color="grey")
        axes[0, j].set(title=DS_NAME[d], xlabel="False-positive rate", ylabel="True-positive rate")
        axes[1, j].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
        axes[0, j].legend(fontsize=5.5, loc="lower right")
        axes[1, j].legend(fontsize=5.5, loc="upper right")
    fig.suptitle(f"Test ROC and precision-recall curves ({CLF_NAME[clf]}, seed {seed}, original prevalence)", fontsize=8)
    fig.tight_layout()
    save(fig, out, f"fig_roc_pr_{clf}")


def fig_sfs(run_dir, out, seed=0):
    files = list(run_dir.glob(f"*_superconducting/tasks/*__seed{seed}/sfs_summary.json")) or \
        list(run_dir.glob(f"*_*/tasks/*__seed{seed}/sfs_summary.json"))
    data = {}
    for f in files:
        d, fm, _ = f.parent.name.split("__")
        data.setdefault(d, {})[fm] = read_json(f)
    if not data:
        return
    dss = [d for d in DS_ORDER if d in data]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.3 * len(dss), 2.6), squeeze=False)
    for ax, d in zip(axes[0], dss):
        for fm in FM_ORDER:
            if fm in data[d]:
                st = data[d][fm]["stages"]
                ax.plot([s["stage"] for s in st], [s["val_roc_auc"] for s in st], "o-", label=fm, ms=3)
        ax.set(title=DS_NAME[d], xlabel="Selected features (qubits)", ylabel="Validation ROC-AUC")
        ax.legend(fontsize=6)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    save(fig, out, "fig_sfs_curves")


def fig_runtime(df, out):
    s = df[df["representation"] == "q_ideal_z"]
    if s.empty:
        return
    comp = [("sfs_runtime_seconds", "SFS (ideal)"), ("time_ideal_features", "Ideal extraction"),
            ("time_noisy_features", "Noisy extraction"), ("train_seconds", "Classifier tuning")]
    comp = [c for c in comp if c[0] in s]
    dss = [d for d in DS_ORDER if d in set(s["dataset"])]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.3 * len(dss), 2.6), squeeze=False)
    for ax, d in zip(axes[0], dss):
        g = s[s["dataset"] == d].groupby("feature_map")[[c for c, _ in comp]].mean().reindex(FM_ORDER)
        x = np.arange(len(FM_ORDER))
        w = 0.8 / len(comp)
        for i, (c, lab) in enumerate(comp):
            ax.bar(x - 0.4 + w * (i + 0.5), g[c].values, w, label=lab, color=f"C{i}")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(FM_ORDER)
        ax.set(title=DS_NAME[d], ylabel="Seconds (log scale)")
    axes[0, -1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig_runtime")


def fig_confusions(df, out):
    best = out / "best_configuration_by_validation.csv"
    if not best.exists():
        return
    b = pd.read_csv(best)
    b = b[b["scope"] == "overall"]
    fig, axes = plt.subplots(1, len(b), figsize=(2.6 * max(1, len(b)), 2.4), squeeze=False)
    for ax, r in zip(axes[0], b.itertuples()):
        g = df[(df["dataset"] == r.dataset) & (df["representation"] == r.representation) &
               (df["feature_map"] == r.feature_map) & (df["classifier"] == r.classifier)]
        cm = np.array([[g["test_tn"].mean(), g["test_fp"].mean()], [g["test_fn"].mean(), g["test_tp"].mean()]])
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,.0f}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["genuine", "fraud"], yticklabels=["genuine", "fraud"],
               xlabel="Predicted", ylabel="True")
        ax.set_title(f"{r.dataset}\n{REP_NAME.get(r.representation, r.representation)} [{r.feature_map}] {CLF_NAME[r.classifier]}",
                     fontsize=6)
    fig.suptitle("Test confusion matrices of the validation-selected best configuration (mean over seeds)", fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_confusion_best")


def extras(run_dir, out):
    """Copy-in summaries of the optional experiments if they were run."""
    notes = {}
    for name in ("qsvm_reference", "sfs_vs_exhaustive", "noise_sweep"):
        f = run_dir / name / "results.csv"
        notes[name] = str(f) if f.exists() else "not run"
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--cache-dir", default=str(ROOT / "cache"))
    ap.add_argument("--run-tag", default="main")
    args = ap.parse_args()
    run_dir = Path(args.results_dir) / args.run_tag
    out = run_dir / "_paper"
    out.mkdir(parents=True, exist_ok=True)
    df, failed = load_all(run_dir)
    df, chk = dedupe(df)
    write_csv_df(out / "all_results_deduplicated.csv", df)
    write_csv_df(out / "consistency_ideal_rows_across_architectures.csv", chk)
    if not failed.empty:
        write_csv_df(out / "FAILED_runs.csv", failed)
    table_datasets(Path(args.cache_dir), out, sorted(set(df["dataset"])))
    table_main(df, out)
    table_noise(df, out)
    table_circuits(df, out)
    sfs_consistent = table_sfs(run_dir, out)
    table_runtime(df, out)
    table_paired_differences(df, out)
    best_configs(df, out)
    for f in (fig_main_bars, fig_fmap_heatmap, fig_noise, fig_runtime, fig_confusions):
        try:
            f(df, out)
        except Exception as e:
            print(f"[warn] {f.__name__} failed: {e}")
    for f in (fig_curves, fig_sfs):
        try:
            f(run_dir, df, out) if f is fig_curves else f(run_dir, out)
        except Exception as e:
            print(f"[warn] {f.__name__} failed: {e}")
    expected = {(c, a) for c in CLF_ORDER for a in ("superconducting", "trapped_ion")}
    present = set(zip(df["classifier"], df["arch_script"]))
    report = {
        "rows_completed": int(len(df)), "rows_failed": int(len(failed)),
        "scripts_present": sorted(f"{c}_{a}" for c, a in present),
        "scripts_missing": sorted(f"{c}_{a}" for c, a in expected - present),
        "datasets": sorted(set(df["dataset"])), "seeds": sorted(map(int, set(df["seed"]))),
        "pilot_rows_present": bool(df["pilot"].astype(bool).any()) if "pilot" in df else False,
        "max_abs_auc_diff_between_architecture_scripts_for_ideal_rows": float(chk["abs_diff"].max()) if len(chk) else None,
        "sfs_subsets_identical_across_scripts": None if sfs_consistent is None else bool(sfs_consistent),
        "optional_experiments": extras(run_dir, out),
    }
    write_json(out / "REPORT.json", report)
    print(json.dumps(report, indent=2))
    print(f"\nPaper tables/figures written to {out}")


if __name__ == "__main__":
    main()
