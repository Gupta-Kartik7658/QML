"""Per-run diagnostic plots (paper-level figures are produced by analyze_results.py)."""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def diagnostics_figure(path, title, y_te, s_te, w_te, y_va, s_va, w_va, thr, test_metrics, curves_te, curves_va):
    fig, ax = plt.subplots(2, 2, figsize=(8.5, 7.2))
    a = ax[0, 0]
    a.plot(curves_te["roc_fpr"], curves_te["roc_tpr"], label=f"test AUC={test_metrics['roc_auc']:.4f}")
    a.plot(curves_va["roc_fpr"], curves_va["roc_tpr"], "--", alpha=0.7, label="validation")
    a.plot([0, 1], [0, 1], ":", color="grey")
    a.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", title="ROC")
    a.legend(loc="lower right")
    a = ax[0, 1]
    a.plot(curves_te["pr_recall"], curves_te["pr_precision"], label=f"test AP={test_metrics['avg_precision']:.4f}")
    a.plot(curves_va["pr_recall"], curves_va["pr_precision"], "--", alpha=0.7, label="validation")
    a.axhline(test_metrics["prevalence_est"], ls=":", color="grey", label="prevalence")
    a.set(xlabel="Recall", ylabel="Precision", title="Precision-recall", ylim=(0, 1.02))
    a.legend(loc="upper right")
    a = ax[1, 0]
    cm = np.array([[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]])
    a.imshow(cm, cmap="Blues")
    a.grid(False)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["genuine", "fraud"], yticklabels=["genuine", "fraud"],
          xlabel="Predicted", ylabel="True", title=f"Test confusion (thr from val = {thr:.3g})")
    a = ax[1, 1]
    bins = np.linspace(np.percentile(s_te, 0.1), np.percentile(s_te, 99.9) + 1e-12, 50)
    a.hist(s_te[y_te == 0], bins=bins, density=True, alpha=0.6, label="genuine")
    a.hist(s_te[y_te == 1], bins=bins, density=True, alpha=0.6, label="fraud")
    a.axvline(thr, color="k", ls="--", lw=1, label="threshold")
    a.set(xlabel="Classifier score", ylabel="Density", title="Test score distribution")
    a.legend()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sfs_figure(path, state: dict, title: str):
    hist = state["history"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    st = [h["stage"] for h in hist]
    ax[0].scatter(st, [h["val_roc_auc"] for h in hist], s=10, alpha=0.4, label="candidates")
    stages = state["stages"]
    ax[0].plot([s["stage"] for s in stages], [s["val_roc_auc"] for s in stages], "o-", color="C3", label="selected")
    for s in stages:
        ax[0].annotate(s["added"], (s["stage"], s["val_roc_auc"]), textcoords="offset points", xytext=(3, 4), fontsize=7)
    ax[0].axvline(state["best_k"], color="k", ls=":", lw=1, label=f"best k={state['best_k']}")
    ax[0].set(xlabel="Subset size k (qubits)", ylabel="Validation ROC-AUC", title="SFS: ROC-AUC")
    ax[0].legend(fontsize=7)
    ax[1].scatter(st, [h["val_avg_precision"] for h in hist], s=10, alpha=0.4)
    ax[1].plot([s["stage"] for s in stages], [s["val_avg_precision"] for s in stages], "o-", color="C3")
    ax[1].set(xlabel="Subset size k (qubits)", ylabel="Validation average precision", title="SFS: AP (secondary)")
    from matplotlib.ticker import MaxNLocator
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def script_summary_figure(path, df, title):
    """Bar chart of test ROC-AUC / AP (mean +- std over seeds) for one classifier x architecture script."""
    import pandas as pd
    if df.empty:
        return
    g = df.groupby(["dataset", "feature_map", "representation"]).agg(
        auc=("test_roc_auc", "mean"), auc_sd=("test_roc_auc", "std"),
        ap=("test_avg_precision", "mean"), ap_sd=("test_avg_precision", "std")).reset_index()
    datasets = list(dict.fromkeys(g["dataset"]))
    fig, ax = plt.subplots(2, len(datasets), figsize=(4.8 * len(datasets), 6.5), squeeze=False)
    for j, dname in enumerate(datasets):
        sub = g[g["dataset"] == dname]
        labels = [f"{r.feature_map}\n{r.representation}" for r in sub.itertuples()]
        x = np.arange(len(sub))
        for i, (m, sd, lab) in enumerate((("auc", "auc_sd", "ROC-AUC"), ("ap", "ap_sd", "Average precision"))):
            ax[i, j].bar(x, sub[m], yerr=sub[sd].fillna(0), color=[f"C{k % 10}" for k in range(len(sub))])
            ax[i, j].set_xticks(x)
            ax[i, j].set_xticklabels(labels, rotation=75, fontsize=6)
            ax[i, j].set(title=f"{dname}: test {lab}", ylim=(max(0, float(pd.Series(sub[m]).min()) - 0.1), 1.0))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
