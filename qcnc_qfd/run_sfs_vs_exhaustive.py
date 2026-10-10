"""SFS vs exhaustive subset search on a deliberately restricted reference problem.

Candidate pool = the top-`--pool` features by training-set mutual information (default 10);
subset sizes 1..`--kmax` (default 4) -> sum_k C(10,k) = 385 exhaustive evaluations vs <= 34 for SFS.
Both use the identical fixed evaluator, identical split/rows and identical validation metric.
Reported per size k: SFS subset vs exhaustive optimum, scores, rank of the SFS subset among all
C(p,k) subsets, evaluation counts and runtime. SFS is a heuristic; this measures its gap.

    python run_sfs_vs_exhaustive.py --datasets creditcard --pilot
    python run_sfs_vs_exhaustive.py --datasets creditcard banksim
"""
import qfd.threads  # noqa: F401

import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from qfd.classifiers import sfs_evaluator
from qfd.config import apply_cli, build_parser, load_config
from qfd.featuremaps import fmap_params, ideal_features
from qfd.metrics import avg_precision, roc_auc
from qfd.preprocess import load_prepared, prepare_all
from qfd.sfs import run_sfs
from qfd.utils import get_logger, limit_threadpools, write_csv_df, write_json

_W = {}


def _init(prep_dir, cfg):
    limit_threadpools(1)
    A, meta = load_prepared(prep_dir)
    sp = A["sfs_pos"]
    _W.update(Xtr=A["Xq_train"][sp], ytr=A["y_train"][sp], Xva=A["Xq_val"], yva=A["y_val"], wva=A["w_val"], cfg=cfg)


def _eval(args):
    cols, fm, seed = args
    t0 = time.perf_counter()
    Ztr, _ = ideal_features(_W["Xtr"][:, cols], fm)
    Zva, _ = ideal_features(_W["Xva"][:, cols], fm)
    m = sfs_evaluator(_W["cfg"]["sfs"].get("evaluator", "logreg"), seed, 1)
    if np.ptp(Ztr, axis=0).max() < 1e-9:
        return cols, 0.5, float("nan"), time.perf_counter() - t0
    m.fit(Ztr, _W["ytr"])
    s = m.predict_proba(Zva)[:, 1]
    return cols, roc_auc(_W["yva"], s, _W["wva"]), avg_precision(_W["yva"], s, _W["wva"]), time.perf_counter() - t0


def main():
    p = build_parser(__doc__)
    p.add_argument("--pool", type=int, default=10)
    p.add_argument("--kmax", type=int, default=4)
    args = p.parse_args()
    cfg = apply_cli(load_config(args.config), args)
    if args.datasets is None:
        cfg["run"]["datasets"] = ["creditcard"]
    pool_n, kmax = (6, 3) if cfg.get("_pilot") else (args.pool, args.kmax)
    out = Path(cfg["paths"]["results_dir"]) / cfg["run"]["run_tag"] / "sfs_vs_exhaustive"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("sfs_vs_exh", out / "run.log")
    prepared = prepare_all(cfg, cfg["run"]["datasets"], cfg["run"]["seeds"], log)
    rows, all_rows = [], []
    for (d, seed), prep in prepared.items():
        A, meta = load_prepared(prep)
        pool = meta["mi_ranking"][:pool_n]
        pidx = [meta["feature_names"].index(f) for f in pool]
        for fmn in cfg["run"]["feature_maps"]:
            fm = fmap_params(cfg, fmn)
            # --- exhaustive
            subsets = [list(c) for k in range(1, kmax + 1) for c in itertools.combinations(pidx, k)]
            t0 = time.perf_counter()
            with ProcessPoolExecutor(int(cfg["run"]["workers"]), mp_context=get_context("spawn"),
                                     initializer=_init, initargs=(prep, cfg)) as ex:
                res = list(ex.map(_eval, [(c, fm, seed) for c in subsets], chunksize=4))
            t_exh = time.perf_counter() - t0
            ex_df = pd.DataFrame([{"subset": ";".join(meta["feature_names"][i] for i in c), "k": len(c),
                                   "val_roc_auc": a, "val_avg_precision": b} for c, a, b, _ in res])
            ex_df.insert(0, "feature_map", fmn)
            ex_df.insert(0, "dataset", d)
            ex_df.insert(2, "seed", seed)
            all_rows.append(ex_df)
            # --- SFS on the same pool (no early stop: full path to kmax, for a per-k comparison)
            sfs_cfg = dict(cfg["sfs"], max_features=kmax, stop_on_no_improvement=False)
            sub_names = [meta["feature_names"][i] for i in pidx]
            sp = A["sfs_pos"]
            tmp = out / f"_sfs_state_{d}_{fmn}_{seed}.json"
            if tmp.exists():
                tmp.unlink()
            t0 = time.perf_counter()
            st = run_sfs(A["Xq_train"][sp][:, pidx], A["y_train"][sp], A["Xq_val"][:, pidx], A["y_val"], A["w_val"],
                         sub_names, fm, sfs_cfg, seed, 1, tmp)
            t_sfs = time.perf_counter() - t0
            tmp.unlink()
            for stg in st["stages"]:
                k = stg["stage"]
                ek = ex_df[ex_df["k"] == k].sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
                sfs_set = set(stg["subset"])
                match = ek["subset"].map(lambda s: set(s.split(";")) == sfs_set)
                rank = int(np.flatnonzero(match.values)[0]) + 1 if match.any() else None
                rows.append({"dataset": d, "feature_map": fmn, "seed": seed, "k": k, "pool": ";".join(pool),
                             "sfs_subset": ";".join(stg["subset"]), "sfs_val_auc": stg["val_roc_auc"],
                             "exhaustive_best_subset": ek.loc[0, "subset"], "exhaustive_best_val_auc": ek.loc[0, "val_roc_auc"],
                             "gap_auc": ek.loc[0, "val_roc_auc"] - stg["val_roc_auc"], "sfs_rank_among_size_k": rank,
                             "n_subsets_size_k": len(ek), "sfs_matches_optimum": bool(rank == 1),
                             "sfs_evaluations_total": st["n_evaluations"], "exhaustive_evaluations_total": len(subsets),
                             "sfs_seconds": t_sfs, "exhaustive_seconds_wall": t_exh})
            log.info(f"{d} {fmn} seed {seed}: SFS {st['n_evaluations']} evals {t_sfs:.0f}s | exhaustive {len(subsets)} "
                     f"evals {t_exh:.0f}s | matches optimum at k=" +
                     ",".join(str(r["k"]) for r in rows[-len(st['stages']):] if r["sfs_matches_optimum"]))
            write_csv_df(out / "results.csv", pd.DataFrame(rows))
            write_csv_df(out / "exhaustive_all_subsets.csv", pd.concat(all_rows))
    df = pd.DataFrame(rows)
    write_csv_df(out / "results.csv", df)
    write_json(out / "config.json", {"pool": pool_n, "kmax": kmax, "evaluator": cfg["sfs"]["evaluator"]})
    print(df[["dataset", "feature_map", "seed", "k", "sfs_val_auc", "exhaustive_best_val_auc", "gap_auc",
              "sfs_rank_among_size_k", "n_subsets_size_k"]].round(4).to_string())


if __name__ == "__main__":
    main()
