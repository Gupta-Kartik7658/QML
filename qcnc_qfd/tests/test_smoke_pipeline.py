"""End-to-end smoke test on small MOCK data (+ real creditcard.csv if present in data/).

    python -m pytest tests/test_smoke_pipeline.py -q        (~1-2 min)

Exercises: loading, preprocessing, SFS, ideal + noisy quantum features, MLP/XGB/SVM, evaluation,
result saving, resume (second run computes nothing) and train/val/test separation.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    data = tmp / "data"
    subprocess.check_call([sys.executable, str(ROOT / "tests" / "make_mock_data.py"), str(data), "8000"])
    real_cc = ROOT / "data" / "creditcard.csv"
    datasets = ["paysim", "banksim"] + (["creditcard"] if real_cc.exists() else [])
    if real_cc.exists():
        shutil.copy(real_cc, data / "creditcard.csv")
    return tmp, data, datasets


def _run(script, tmp, data, datasets, extra=()):
    cmd = [sys.executable, str(ROOT / script), "--pilot", "--workers", "2", "--data-dir", str(data),
           "--cache-dir", str(tmp / "cache"), "--results-dir", str(tmp / "results"), "--run-tag", "smoke",
           "--datasets", *datasets, "--feature-maps", "Z", "ZZ", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=1800)


@pytest.mark.parametrize("clf", ["xgb", "mlp", "svm"])
def test_end_to_end(env, clf):
    tmp, data, datasets = env
    r = _run(f"run_{clf}_superconducting.py", tmp, data, datasets)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp / "results" / "smoke" / f"{clf}_superconducting"
    df = pd.read_csv(out / "results.csv")
    assert (df["status"] == "completed").all()
    reps = set(df["representation"])
    assert {"q_ideal_z", "c_selected", "c_all", "q_noisy_z", "q_ideal_paired_z"} <= reps
    # every representation within a task used the same selected features
    q = df[df["representation"].isin(["q_ideal_z", "c_selected", "q_noisy_z"])]
    assert q.groupby("task_id")["selected_features"].nunique().max() == 1
    for t in out.glob("tasks/*__seed0"):
        assert (t / "task_status.json").exists()
    f = next(out.glob("tasks/*__ZZ__seed0/q_ideal_z"))
    for name in ("metrics.json", "hp_search.csv", "iterations.json", "predictions.npz", "curves.npz", "model.joblib"):
        assert (f / name).exists(), name
    # resume: second invocation must not recompute anything
    r2 = _run(f"run_{clf}_superconducting.py", tmp, data, datasets)
    assert "0 to run" in r2.stdout


def test_same_representation_across_classifiers(env):
    tmp, data, datasets = env
    dfs = [pd.read_csv(tmp / "results" / "smoke" / f"{c}_superconducting" / "results.csv") for c in ("xgb", "mlp", "svm")]
    k = ["dataset", "feature_map", "seed", "representation"]
    cols = k + ["selected_features", "repr_dim", "n_train", "n_val", "n_test", "n_train_pos", "n_test_pos"]
    a, b, c = (d[cols].sort_values(k).reset_index(drop=True) for d in dfs)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


def test_prepared_partitions_disjoint(env):
    tmp, _, _ = env
    import numpy as np
    for d in (tmp / "cache" / "prepared").glob("*/*"):
        z = np.load(d / "arrays.npz")
        tr, va, te = set(z["rows_train"]), set(z["rows_val"]), set(z["rows_test"])
        assert not (tr & va) and not (tr & te) and not (va & te)
        meta = json.loads((d / "meta.json").read_text())
        assert all(v == 0 for k, v in meta["leakage_checks"].items() if k.endswith("overlap"))
