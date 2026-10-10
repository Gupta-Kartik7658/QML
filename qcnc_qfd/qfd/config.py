"""Configuration loading + command-line interface shared by all runner scripts."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"

FEATURE_MAPS = ("Z", "ZZ", "Pauli")
DATASETS = ("creditcard", "paysim", "banksim")
ARCHITECTURES = ("superconducting", "trapped_ion")
CLASSIFIERS = ("mlp", "xgb", "svm")


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_file"] = str(path)
    return cfg


def resolve_paths(cfg: dict) -> dict:
    for k in ("data_dir", "cache_dir", "results_dir"):
        p = Path(cfg["paths"][k])
        cfg["paths"][k] = str(p if p.is_absolute() else (ROOT / p))
    return cfg


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
    p.add_argument("--workers", type=int, default=None, help="worker processes (default: config, 6)")
    p.add_argument("--threads-per-worker", type=int, default=None, help="BLAS/XGB threads per worker")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    p.add_argument("--feature-maps", nargs="+", choices=FEATURE_MAPS, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--max-features", type=int, default=None, help="SFS maximum subset size")
    p.add_argument("--reps", type=int, default=None, help="feature-map repetitions")
    p.add_argument("--no-noise", action="store_true", help="skip noisy (architecture) simulation")
    p.add_argument("--shots", type=int, default=None, help="finite shots for noisy features (default exact)")
    p.add_argument("--split", choices=["stratified", "temporal"], default=None)
    p.add_argument("--pilot", action="store_true",
                   help="small fast run: 1 seed, small caps, max 3 features (checks everything works)")
    p.add_argument("--sample-limit", type=int, default=None,
                   help="pilot only: cap rows read from each dataset (stratified)")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--run-tag", default="main", help="results/<run-tag>/... (use different tags for pilots)")
    p.add_argument("--baselines", choices=["auto", "on", "off"], default=None,
                   help="classical-all baseline (auto = only superconducting scripts)")
    p.add_argument("--retry-failed", action="store_true", help="re-run tasks that previously failed")
    p.add_argument("--no-resume", action="store_true", help="ignore completed results and recompute")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--arch-only", action="store_true",
                   help="train only the architecture-dependent representations (q_noisy_z, q_noisy_shots_z); "
                        "the ideal/classical rows are architecture-independent and come from the other script")
    p.add_argument("--log-level", default=None)
    return p


def apply_cli(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(cfg)
    r = cfg["run"]
    if args.workers is not None:
        r["workers"] = args.workers
    if args.threads_per_worker is not None:
        r["threads_per_worker"] = args.threads_per_worker
    if args.datasets:
        r["datasets"] = list(args.datasets)
    if args.feature_maps:
        r["feature_maps"] = list(args.feature_maps)
    if args.seeds is not None:
        r["seeds"] = list(args.seeds)
    if args.max_features is not None:
        cfg["sfs"]["max_features"] = args.max_features
    if args.reps is not None:
        cfg["feature_maps"]["reps"] = args.reps
    if args.no_noise:
        cfg["noise"]["enabled"] = False
    if args.shots is not None:
        cfg["noise"]["shots"] = args.shots
    if args.split:
        cfg["split"]["mode"] = args.split
    if args.data_dir:
        cfg["paths"]["data_dir"] = args.data_dir
    if args.results_dir:
        cfg["paths"]["results_dir"] = args.results_dir
    if args.cache_dir:
        cfg["paths"]["cache_dir"] = args.cache_dir
    if args.baselines:
        cfg["baselines"]["classical_all"] = args.baselines
    if args.no_plots:
        r["task_plots"] = False
    if getattr(args, "arch_only", False):
        r["arch_only"] = True
    if args.log_level:
        r["log_level"] = args.log_level
    r["sample_limit"] = args.sample_limit
    r["run_tag"] = args.run_tag
    r["retry_failed"] = args.retry_failed
    r["resume"] = not args.no_resume
    if args.pilot:
        apply_pilot(cfg)
        if args.sample_limit is None:
            r["sample_limit"] = 60000
        if args.run_tag == "main":
            r["run_tag"] = "pilot"
    return resolve_paths(cfg)


def apply_pilot(cfg: dict) -> None:
    """Small configuration for smoke tests / pilots (never used for paper numbers)."""
    cfg["run"]["seeds"] = cfg["run"]["seeds"][:1]
    s = cfg["sampling"]
    s.update(train_cap=4000, val_neg_cap=6000, test_neg_cap=12000, sfs_train_cap=2000,
             noisy_train_cap=400, noisy_train_max_pos=200, noisy_val_neg=300, noisy_test_neg=500)
    cfg["sfs"]["max_features"] = min(3, cfg["sfs"]["max_features"])
    cfg["evaluation"]["n_bootstrap"] = 20
    cfg["classifiers"]["mlp"]["grid"] = {"hidden_layer_sizes": [[32]], "alpha": [0.0001]}
    cfg["classifiers"]["xgb"]["grid"] = {"max_depth": [3], "learning_rate": [0.1], "scale_pos_weight": [1.0]}
    cfg["classifiers"]["svm"]["grid"] = {"C": [1.0], "class_weight": ["balanced"]}
    cfg["_pilot"] = True
