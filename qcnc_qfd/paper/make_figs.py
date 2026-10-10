"""Paper figures (IEEE double-column widths). Inputs: paper/data/*.csv (see build_figure_data.py) and
results/main/{shot_budget,ibm_hardware}. Usage:  python paper/make_figs.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = str(ROOT / "results" / "main")
DATA = ROOT / "paper" / "data"
OUT = str(ROOT / "paper" / "figs")
C = {"hw": "#2a78d6", "sc": "#eb6834", "ti": "#1baf7a", "ideal": "#52514e", "fez": "#6250d6"}
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                     "legend.fontsize": 6.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
P = ("train", "val", "test")


# ---------------------------------------------------------------- Fig. 2: hardware affine law + beta vs gates
def fig_affine():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    sc = pd.read_csv(DATA / "affine_scatter.csv")
    for fm, col, mk, lab in (("Z", C["hw"], "o", "Z map (product)"), ("ZZ", C["sc"], "^", "ZZ map ($k$=7, entangling)")):
        s = sc[sc["fmap"] == fm]
        ax.scatter(s["z_ideal"], s["z_hardware"], s=5, alpha=0.75, color=col, marker=mk, linewidths=0, label=lab)
        c = np.polyfit(s["z_ideal"], s["z_hardware"], 1)
        xs = np.linspace(-1, 1, 20)
        ax.plot(xs, np.polyval(c, xs), color=col, lw=1)
    ax.plot([-1, 1], [-1, 1], ":", color=C["ideal"], lw=0.8, label="noise-free ($y=x$)")
    ax.set_xlabel(r"ideal $\langle Z\rangle$ (exact simulation)")
    ax.set_ylabel(r"measured $\langle Z\rangle$ on ibm_marrakesh")
    ax.set_title("(a) median-$\\beta$ qubit, 1,689 credit-card rows", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    ax = axes[1]
    d = pd.read_csv(DATA / "beta_vs_gates.csv")
    series = [("hardware ibm_marrakesh", "IBM Heron (ibm_marrakesh)", C["hw"], "o"),
              ("hardware ibm_fez", "IBM Heron (ibm_fez)", C["fez"], "D"),
              ("sim trapped_ion", "trapped-ion model (sim.)", C["ti"], "s"),
              ("sim superconducting", "superconducting model (sim.)", C["sc"], "^")]
    gmax = 60
    for src, lab, col, mk in series:
        s = d[d["source"] == src]
        if s.empty:
            continue
        g = s["gates_2q_per_qubit"].values
        b = s["beta_median"].values
        lo, hi = s["beta_q25"].values, s["beta_q75"].values
        ax.errorbar(g, b, yerr=[b - lo, hi - b], fmt=mk, color=col, ms=4, capsize=1.5, lw=0.8, label=lab, zorder=3)
        if len(np.unique(g)) > 1:  # fit beta = beta0 * eta^g on log scale
            b0 = b[g == 0].mean()
            gg, bb = g[g > 0], b[g > 0]
            slope = np.sum(gg * np.log(bb / b0)) / np.sum(gg ** 2)
            eta = np.exp(slope)
            xs = np.linspace(0, gmax, 50)
            ax.plot(xs, b0 * eta ** xs, "-", color=col, lw=0.9, alpha=0.8)
            ax.text(xs[-1] + 0.8, b0 * eta ** xs[-1], rf"$\eta$={eta:.3f}", color="#0b0b0b", fontsize=6, va="center")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.5", "0.7", "1"])
    ax.set_xlim(-2, gmax + 12)
    ax.set_xlabel("two-qubit gates acting on each qubit (compiled)")
    ax.set_ylabel(r"signal retained $\beta$ (median over qubits)")
    ax.set_title(r"(b) $\beta \approx \beta_0\,\eta^{g}$: Z map ($g$=0) vs ZZ, $k$=3,5,7", loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{OUT}/fig_affine_beta.pdf")
    fig.savefig(f"{OUT}/fig_affine_beta.png", dpi=250)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 3: shot budget, simulation + hardware
def fig_shots():
    S = pd.read_csv(F + "/shot_budget/results.csv")
    S["shots"] = S["shots"].replace([np.inf, "inf"], np.inf).astype(float)
    hw = {("creditcard", "Z"): F + "/ibm_hardware/hardware_shot_budget.csv",
          ("creditcard", "ZZ"): F + "/ibm_hardware/hardware_shot_budget.csv",
          ("paysim", "Z"): F + "/ibm_hardware/paysim_Z/hardware_shot_budget.csv"}
    panels = [("creditcard", "Z"), ("creditcard", "ZZ"), ("paysim", "Z"), ("paysim", "ZZ")]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.85), sharey=False)
    for ax, (ds, fm) in zip(axes, panels):
        for sim, lab, col, mk in (("ideal", "ideal (sim.)", C["ideal"], "."), ("trapped_ion", "trapped-ion (sim.)", C["ti"], "s"),
                                  ("superconducting", "supercond. (sim.)", C["sc"], "^")):
            g = S[(S["dataset"] == ds) & (S["fmap"] == fm) & (S["sim"] == sim)]
            fin = g[np.isfinite(g["shots"])].groupby("shots")["test_roc_auc"].agg(["mean", "std"])
            ax.plot(fin.index, fin["mean"], marker=mk, ms=3, lw=1, color=col, label=lab)
            ax.fill_between(fin.index, fin["mean"] - fin["std"], fin["mean"] + fin["std"], color=col, alpha=0.12, lw=0)
        if (ds, fm) in hw:
            h = pd.read_csv(hw[(ds, fm)])
            hi = h[(h["fmap"] == fm) & (h["source"] == "ideal sim")]
            h = h[(h["fmap"] == fm) & (h["source"] == "hardware")]
            ax.plot(h["shots"], h["test_roc_auc"], "o-", ms=3.5, lw=1, color=C["hw"], label="IBM Heron (hardware rows)")
            ax.plot(hi["shots"], hi["test_roc_auc"], "o--", ms=2.5, lw=0.9, color=C["hw"], mfc="white",
                    label="ideal, same hardware rows")
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024, 8192])
        ax.set_xticklabels(["16", "64", "256", "1k", "8k"])
        ax.set_title(f"{'Credit card' if ds == 'creditcard' else 'PaySim'}, {fm} map", loc="left")
        ax.set_xlabel("shots per sample")
    axes[0].set_ylabel("test ROC-AUC (XGBoost)")
    hs, ls = [], []
    for a in axes:
        for h_, l_ in zip(*a.get_legend_handles_labels()):
            if l_ not in ls:
                hs.append(h_)
                ls.append(l_)
    fig.legend(hs, ls, loc="upper center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.9))
    fig.savefig(f"{OUT}/fig_shots.pdf")
    fig.savefig(f"{OUT}/fig_shots.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    fig_affine()
    fig_shots()
    print("ok")
