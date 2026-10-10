"""Rebuild paper/data/*.csv (inputs of make_figs.py) from the experiment results.

Needs the IBM data bundles (results/main/ibm_hardware/**/bundle/bundle.npz), which are not stored in the
repository because of their size; recreate them with scripts/4_ibm_prepare.* (deterministic).

    python paper/build_figure_data.py [--results results/main]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qfd.config import load_config  # noqa: E402
from qfd.featuremaps import fmap_params, observable_matrices  # noqa: E402
from qfd.noisy import compile_for_arch  # noqa: E402
from qfd.phase2 import affine_fit  # noqa: E402

P = ("train", "val", "test")


def beta_vs_gates(H: Path, cfg) -> pd.DataFrame:
    rows = []
    for sub, k in (("", 7), ("creditcard_ZZ_k3", 3), ("creditcard_ZZ_k5", 5)):  # simulated architectures, same rows
        B = dict(np.load(H / sub / "bundle" / "bundle.npz"))
        M, _ = observable_matrices(k, False)
        zi = np.concatenate([B[f"ZZ_ideal_probs_{p}"] @ M for p in P])
        for arch in ("superconducting", "trapped_ion"):
            zn = np.concatenate([B[f"ZZ_sim_{arch}_probs_{p}"] @ M for p in P])
            b = [f["beta"] for f in affine_fit(zi, zn)]
            _, _, st = compile_for_arch(fmap_params(cfg, "ZZ"), k, arch, cfg["architectures"][arch], cfg["noise"])
            n2 = st["compiled"]["two_qubit_gates"]
            rows.append({"source": f"sim {arch}", "fmap": "ZZ", "k": k, "gates_2q_total": n2, "gates_2q_per_qubit": 2 * n2 / k,
                         "beta_median": float(np.median(b)), "beta_q25": float(np.percentile(b, 25)),
                         "beta_q75": float(np.percentile(b, 75))})
    for arch, b in (("superconducting", 0.9165104873134731), ("trapped_ion", 0.9911703707930435)):  # Z map: exact law
        rows.append({"source": f"sim {arch}", "fmap": "Z", "k": 7, "gates_2q_total": 0, "gates_2q_per_qubit": 0,
                     "beta_median": b, "beta_q25": b, "beta_q75": b})
    for sub in ("", "paysim_Z", "banksim_Z", "creditcard_Z_fez", "creditcard_ZZ_k3", "creditcard_ZZ_k5"):  # hardware
        Q = pd.read_csv(H / sub / "hardware_affine_per_qubit.csv")
        for (be, fm), g in Q.groupby(["backend", "fmap"]):
            pl = next(json.load(open(j))["plan"][fm] for j in glob.glob(str(H / sub / "job_*.json"))
                      if fm in json.load(open(j))["plan"])
            k = int(g["feature"].max() + 1)
            rows.append({"source": f"hardware {be}", "experiment": sub or "creditcard", "fmap": fm, "k": k,
                         "gates_2q_total": pl["isa_2q_gates"] / pl["copies"],
                         "gates_2q_per_qubit": 2 * pl["isa_2q_gates"] / (pl["copies"] * k), "n_qubits": len(g),
                         "beta_median": g["beta"].median(), "beta_q25": g["beta"].quantile(.25),
                         "beta_q75": g["beta"].quantile(.75), "r2_median": g["r2"].median(),
                         "resid_over_shot_median": g["residual_sd_over_shot_sd"].median()})
    return pd.DataFrame(rows)


def affine_scatter(H: Path) -> pd.DataFrame:
    """Ideal vs measured <Z> of the median-beta physical qubit, Z and ZZ (k=7) maps, credit card."""
    B = dict(np.load(H / "bundle" / "bundle.npz"))
    out = []
    for fm in ("Z", "ZZ"):
        f = next(j for j in glob.glob(str(H / "job_*.json")) if fm in json.load(open(j))["plan"])
        rec = json.load(open(f))
        Hd = dict(np.load(H / f"hw_{rec['backend']}_{rec['job_id']}.npz", allow_pickle=True))
        pl = rec["plan"][fm]
        k = pl["k"]
        M, _ = observable_matrices(k, False)
        zi = np.concatenate([B[f"{fm}_ideal_probs_{p}"] @ M for p in P])
        zh = np.concatenate([Hd[f"{fm}_z_{p}"] for p in P])
        cp = np.arange(len(zi)) % pl["copies"]
        cand = sorted((np.polyfit(zi[cp == c, j], zh[cp == c, j], 1)[0], c, j)
                      for c in range(pl["copies"]) for j in range(k))
        _, c0, j0 = cand[len(cand) // 2]
        m = cp == c0
        out.append(pd.DataFrame({"fmap": fm, "physical_qubit": pl["physical_qubits"][c0 * k + j0],
                                 "z_ideal": zi[m, j0], "z_hardware": zh[m, j0]}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "main"))
    a = ap.parse_args()
    H = Path(a.results) / "ibm_hardware"
    out = ROOT / "paper" / "data"
    out.mkdir(parents=True, exist_ok=True)
    beta_vs_gates(H, load_config(str(ROOT / "configs" / "experiment.yaml"))).to_csv(out / "beta_vs_gates.csv", index=False)
    affine_scatter(H).to_csv(out / "affine_scatter.csv", index=False)
    print(f"written to {out}")
