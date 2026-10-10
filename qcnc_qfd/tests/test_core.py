"""Unit tests:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qfd.featuremaps import (build_circuit, fmap_params, ideal_features, no_wrap_range,  # noqa: E402
                             observable_matrices, simulate_probabilities, statevectors)
from qfd.metrics import bootstrap_ci, choose_threshold, full_metrics, roc_auc  # noqa: E402
from qfd.sfs import max_evaluations, run_sfs  # noqa: E402

RNG = np.random.default_rng(0)
ZZ = {"name": "ZZ", "reps": 2, "entanglement": "full"}
PAULI = {"name": "Pauli", "reps": 2, "entanglement": "full", "paulis": ["Z", "Y", "ZZ"], "alpha": 2.0}


# ----------------------------------------------------------------------------- quantum features
@pytest.mark.parametrize("prm", [{"name": "Z", "reps": 2}, ZZ, PAULI, {"name": "ZZ", "reps": 3, "entanglement": "linear"}])
@pytest.mark.parametrize("k", [1, 2, 4, 6])
def test_fast_simulator_matches_qiskit(prm, k):
    from qiskit.quantum_info import Statevector
    X = RNG.uniform(-1, 4, (4, k))
    qc = build_circuit(prm, k)
    ref = np.array([Statevector(qc.assign_parameters(x)).probabilities() for x in X])
    assert np.abs(simulate_probabilities(X, prm) - ref).max() < 1e-10


@pytest.mark.parametrize("prm", [{"name": "Z", "reps": 1}, {"name": "ZZ", "reps": 1, "entanglement": "full"},
                                 {"name": "Pauli", "reps": 1, "paulis": ["X", "ZZ"], "entanglement": "full", "alpha": 2.0}])
def test_reps1_z_expectations_are_identically_zero(prm):
    """Documented finding: with reps=1 the <Z_i> representation carries no information."""
    F, _ = ideal_features(RNG.uniform(0, 1, (50, 4)), prm)
    assert np.abs(F).max() < 1e-12


def test_pauli_x_zz_equals_zz_map_in_z_basis():
    """Documented finding: Pauli[X,ZZ] has identical computational-basis statistics to ZZFeatureMap."""
    X = RNG.uniform(0, 1, (30, 4))
    a, _ = ideal_features(X, ZZ, extended=True)
    b, _ = ideal_features(X, {"name": "Pauli", "reps": 2, "entanglement": "full", "paulis": ["X", "ZZ"], "alpha": 2.0},
                          extended=True)
    assert np.abs(a - b).max() < 1e-6
    c, _ = ideal_features(X, PAULI, extended=True)
    assert np.abs(a - c).max() > 0.1  # the [Z,Y,ZZ] map used in the study is genuinely different


def test_feature_shapes_finiteness_and_range():
    X = RNG.uniform(0, 1, (100, 5))
    F, names = ideal_features(X, ZZ, extended=True)
    assert F.shape == (100, 5 + 10) and len(names) == 15
    assert np.isfinite(F).all() and np.abs(F).max() <= 1 + 1e-6


def test_statevector_kernel_is_valid():
    P = statevectors(RNG.uniform(0, 1, (20, 3)), ZZ)
    K = np.abs(P @ P.conj().T) ** 2
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T) and np.linalg.eigvalsh(K).min() > -1e-9


def test_no_wrap_range_property():
    lo, hi = no_wrap_range(2.0)
    span_pair = 2.0 * ((np.pi - lo) ** 2 - (np.pi - hi) ** 2)
    assert span_pair <= 2 * np.pi + 1e-9 and 2.0 * (hi - lo) <= 2 * np.pi


def test_observable_matrix():
    M, names = observable_matrices(2, True)
    # basis |00>,|01>,|10>,|11> (qiskit little-endian: index bit0 = qubit 0)
    assert names == ["Z0", "Z1", "Z0Z1"]
    assert M.tolist() == [[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]]


# ----------------------------------------------------------------------------- noise
def test_noisy_zero_noise_equals_ideal_and_readout():
    import yaml
    from qfd.noisy import apply_readout, noisy_probabilities
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "configs" / "experiment.yaml"))
    X = RNG.uniform(0, 3, (3, 3))
    for arch, z in {"superconducting": dict(T1_us=1e12, T2_us=1e12, p_dep_1q=0, p_dep_2q=0, p_readout=0),
                    "trapped_ion": dict(p_dep_1q=0, p_dep_2q=0, p_amp_damp=0, p_phase_damp=0, p_readout=0)}.items():
        P, _ = noisy_probabilities(X, ZZ, arch, cfg["architectures"][arch], cfg["noise"], z)
        assert np.abs(P - simulate_probabilities(X, ZZ)).max() < 1e-8
        Pn, _ = noisy_probabilities(X, ZZ, arch, cfg["architectures"][arch], cfg["noise"])
        assert np.abs(Pn - P).max() > 1e-4  # noise has an effect
    assert np.allclose(apply_readout(np.array([[1.0, 0, 0, 0]]), 2, 0.1), [[0.81, 0.09, 0.09, 0.01]])


# ----------------------------------------------------------------------------- metrics
def test_weighted_metrics_and_fast_bootstrap():
    from sklearn.metrics import roc_auc_score
    from qfd.metrics import _auc_ap_sorted
    y = (RNG.random(3000) < 0.05).astype(int)
    s = RNG.normal(size=3000) + 1.5 * y
    s = np.round(s, 1)  # ties
    assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-12
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss) != 0) + 1]
    auc, ap = _auc_ap_sorted(ys, np.ones(len(ys)), starts)
    from sklearn.metrics import average_precision_score
    assert abs(auc - roc_auc_score(y, s)) < 1e-9 and abs(ap - average_precision_score(y, s)) < 1e-9
    ci = bootstrap_ci(y, s, None, 50, 0)
    assert ci["roc_auc_ci_low"] < roc_auc(y, s) < ci["roc_auc_ci_high"]


def test_metric_definitions():
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    m = full_metrics(y, s, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 3)
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["specificity"] == 0.75 and m["fpr"] == 0.25
    assert choose_threshold(y, s) in s


# ----------------------------------------------------------------------------- SFS
def _toy(n=600, p=6, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, p))
    y = ((X[:, 1] > 0.7) | (X[:, 3] < 0.15)).astype(int)
    return X, y


def test_sfs_evaluation_count_and_reproducibility(tmp_path):
    X, y = _toy()
    names = [f"f{i}" for i in range(X.shape[1])]
    cfg = {"max_features": 3, "evaluator": "logreg", "metric": "roc_auc", "tol": 0.0, "stop_on_no_improvement": False}
    a = run_sfs(X[:300], y[:300], X[300:], y[300:], np.ones(300), names, {"name": "Z", "reps": 2}, cfg, 0, 1, tmp_path / "a.json")
    b = run_sfs(X[:300], y[:300], X[300:], y[300:], np.ones(300), names, {"name": "Z", "reps": 2}, cfg, 0, 1, tmp_path / "b.json")
    assert a["n_evaluations"] == max_evaluations(6, 3) == 6 + 5 + 4
    assert a["best_subset"] == b["best_subset"] and [h["val_roc_auc"] for h in a["history"]] == [h["val_roc_auc"] for h in b["history"]]
    assert {"f1", "f3"} & set(a["best_subset"])
    assert max_evaluations(30, 7) == 189 and max_evaluations(30, 6) == 165


def test_sfs_resume(tmp_path):
    X, y = _toy()
    names = [f"f{i}" for i in range(6)]
    cfg = {"max_features": 3, "evaluator": "logreg", "metric": "roc_auc", "tol": 0.0, "stop_on_no_improvement": False}
    full = run_sfs(X[:300], y[:300], X[300:], y[300:], np.ones(300), names, {"name": "Z", "reps": 2}, cfg, 0, 1, tmp_path / "f.json")
    # simulate a crash after stage 1: keep only the first stage in the checkpoint
    import json
    st = json.loads((tmp_path / "f.json").read_text())
    st["stages"], st["selected"], st["done"] = st["stages"][:1], st["selected"][:1], False
    st["history"] = [h for h in st["history"] if h["stage"] == 1]
    st.pop("stopping_reason", None)
    (tmp_path / "g.json").write_text(json.dumps(st))
    resumed = run_sfs(X[:300], y[:300], X[300:], y[300:], np.ones(300), names, {"name": "Z", "reps": 2}, cfg, 0, 1, tmp_path / "g.json")
    assert resumed["best_subset"] == full["best_subset"] and resumed["n_evaluations"] == full["n_evaluations"]


# ----------------------------------------------------------------------------- preprocessing / leakage
def test_split_no_overlap_and_train_only_fit():
    from qfd.preprocess import Preprocessor, make_split, subsample_plan
    n = 5000
    y = (RNG.random(n) < 0.03).astype(np.int8)
    tr, va, te = make_split(y, None, "stratified", [0.6, 0.2, 0.2], 0)
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == n
    X = pd.DataFrame({"a": RNG.normal(size=n), "b": RNG.lognormal(size=n), "c": np.ones(n)})
    X.loc[te, "a"] += 100.0  # a shift only in the test rows must not influence the fitted transforms
    pp = Preprocessor(["b"], [], "quantile", seed=0).fit(X.iloc[tr].reset_index(drop=True), y[tr], {})
    assert "c" in pp.constant_cols
    assert np.allclose(pp.std_scaler.mean_[0], X.loc[tr, "a"].mean())
    Xc, U = pp.transform(X.iloc[te].reset_index(drop=True), {})
    assert U.min() >= 0 and U.max() <= 1
    s = {"train_cap": 1000, "train_max_pos_frac": 0.5, "val_neg_cap": 300, "test_neg_cap": None, "sfs_train_cap": 500,
         "noisy_train_cap": 200, "noisy_train_max_pos": 50, "noisy_val_neg": 100, "noisy_test_neg": 100}
    plan = subsample_plan(y, tr, va, te, s, 0)
    assert set(plan["tr_sub"]) <= set(tr) and set(plan["va_sub"]) <= set(va) and set(plan["te_sub"]) == set(te)
    # inverse-probability weights reproduce the full validation negative count
    yv = y[plan["va_sub"]]
    assert np.isclose(plan["w_val"][yv == 0].sum(), (y[va] == 0).sum())


def test_target_encoding_is_out_of_fold():
    from qfd.preprocess import Preprocessor
    n = 2000
    cats = RNG.integers(0, 40, n).astype(str)
    y = (RNG.random(n) < 0.1).astype(int)
    X = pd.DataFrame({"m": np.zeros(n), "x": RNG.normal(size=n)})
    pp = Preprocessor([], ["m"], seed=0).fit(X, y, {"m": cats})
    full_map, prior = pp.te_maps["m"]
    in_sample = pd.Series(cats).map(full_map).to_numpy()
    assert not np.allclose(pp.te_oof["m"], in_sample)  # training rows never see their own label


def test_fmap_params_resolves_no_wrap():
    cfg = {"feature_maps": {"reps": 2, "ZZ": {"entanglement": "full", "input_range": "no_wrap"}, "Z": {"input_range": [-1, 1]}}}
    assert fmap_params(cfg, "ZZ")["input_range"] == no_wrap_range(2.0)
    assert fmap_params(cfg, "Z")["input_range"] == [-1.0, 1.0]


# ----------------------------------------------------------------------------- cache
def test_cache_keys_and_invalid_entries(tmp_path):
    from qfd.cache import FeatureCache
    c = FeatureCache(tmp_path)
    k1, k2 = c.key(a=1, fmap=ZZ), c.key(a=1, fmap=PAULI)
    assert k1 != k2 and k1 == c.key(a=1, fmap=ZZ)
    calls = []
    f = lambda: calls.append(1) or {"x": np.ones((3, 2), np.float32)}  # noqa: E731
    c.get_or_compute(k1, f, {"x": (3, 2)})
    c.get_or_compute(k1, f, {"x": (3, 2)})
    assert len(calls) == 1 and c.hits == 1
    c.get_or_compute(k1, f, {"x": (4, 2)})  # shape mismatch -> recompute
    assert len(calls) == 2
