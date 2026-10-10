"""Phase-2 unit tests: affine noise law, paired bootstrap, hybrid control features."""
import numpy as np
import pytest

from qfd.config import load_config
from qfd.featuremaps import fmap_params, ideal_probabilities, observable_matrices
from qfd.metrics import paired_bootstrap_diff, roc_auc
from qfd.phase2 import affine_fit

CFG = load_config("configs/experiment.yaml")


@pytest.mark.parametrize("arch", ["superconducting", "trapped_ion"])
def test_affine_noise_law_z_map(arch):
    from qfd.noisy import noisy_probabilities
    from run_theory_check import closed_form_beta_sc, scaled_arch
    fm = fmap_params(CFG, "Z")
    U = np.random.default_rng(0).random((40, 3))
    M, _ = observable_matrices(3, False)
    zi = ideal_probabilities(U, fm) @ M
    for factor in (1.0, 10.0):
        a = scaled_arch(CFG["architectures"][arch], factor)
        zn = noisy_probabilities(U, fm, arch, a, CFG["noise"])[0] @ M
        fits = affine_fit(zi, zn)
        assert max(f["max_abs_residual"] for f in fits) < 1e-10
        if arch == "superconducting":
            assert abs(fits[0]["beta"] - closed_form_beta_sc(a)) < 1e-10


def test_affine_law_fails_for_entangling_map():
    from qfd.noisy import noisy_probabilities
    fm = fmap_params(CFG, "ZZ")
    U = np.random.default_rng(1).random((40, 3))
    M, _ = observable_matrices(3, False)
    zi = ideal_probabilities(U, fm) @ M
    zn = noisy_probabilities(U, fm, "superconducting", CFG["architectures"]["superconducting"], CFG["noise"])[0] @ M
    assert max(f["max_abs_residual"] for f in affine_fit(zi, zn)) > 1e-3


def test_ideal_z_map_is_cosine():
    fm = fmap_params(CFG, "Z")
    U = np.random.default_rng(2).random((30, 2))
    M, _ = observable_matrices(2, False)
    lo, hi = fm["input_range"]
    assert np.abs(ideal_probabilities(U, fm) @ M - np.cos(2 * (lo + (hi - lo) * U))).max() < 1e-12


def test_paired_bootstrap():
    rng = np.random.default_rng(3)
    y = (rng.random(3000) < 0.05).astype(int)
    s1 = y + rng.normal(0, 1.0, len(y))
    s2 = y + rng.normal(0, 0.5, len(y))
    w = np.where(y == 1, 1.0, 3.0)
    r0 = paired_bootstrap_diff(y, s1, s1, w, n_boot=50)
    assert r0["d_auc"] == 0 and r0["d_auc_ci_low"] == 0 and r0["d_auc_ci_high"] == 0
    r = paired_bootstrap_diff(y, s1, s2, w, n_boot=200)
    assert abs(r["auc_a"] - roc_auc(y, s1, w)) < 1e-9 and abs(r["auc_b"] - roc_auc(y, s2, w)) < 1e-9
    assert r["d_auc"] > 0 and r["d_auc_ci_low"] > 0


def test_rff_control_shape_and_determinism():
    from run_hybrid import rff
    U = {p: np.random.default_rng(4).random((10, 3)) for p in ("train", "val", "test")}
    a, b = rff(U, 3, seed=7), rff(U, 3, seed=7)
    assert a["train"].shape == (10, 3) and np.array_equal(a["test"], b["test"])
    assert np.abs(a["val"]).max() <= 1
