"""Environment + correctness smoke test. Run this FIRST on each computer (takes < 1 minute).

    python check_env.py

Checks: package versions vs requirements.txt pins, Qiskit circuit execution, the fast simulator
against qiskit.quantum_info.Statevector, the noisy Aer path (zero noise == ideal), XGBoost,
multiprocessing with 'spawn', and the data files.
"""
import qfd.threads  # noqa: F401

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

PINS = {"qiskit": "1.3.1", "qiskit_aer": "0.15.1", "qiskit_machine_learning": "0.8.0", "numpy": "2.3.5",
        "scipy": "1.16.3", "sklearn": "1.7.2", "xgboost": "3.1.1", "pandas": "2.3.3", "matplotlib": "3.10.7"}


def _square(x):
    import numpy as np
    return float(np.square(x))


def main():
    ok = True
    print(f"Python {sys.version.split()[0]} (requirements.txt was frozen on 3.12; qiskit-aer 0.15.1 has no 3.13 wheels)")
    if sys.version_info[:2] != (3, 12):
        print("  [WARN] use Python 3.12 for qiskit-aer==0.15.1")
    for mod, want in PINS.items():
        try:
            have = __import__(mod).__version__
            flag = "OK " if have == want else "WARN"
            print(f"  [{flag}] {mod:24s} {have:10s} (pinned {want})")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {mod}: {e}")
    import numpy as np
    from qiskit.quantum_info import Statevector
    from qfd.featuremaps import build_circuit, simulate_probabilities
    rng = np.random.default_rng(0)
    worst = 0.0
    for prm in ({"name": "Z", "reps": 2}, {"name": "ZZ", "reps": 2, "entanglement": "full"},
                {"name": "Pauli", "reps": 2, "entanglement": "full", "paulis": ["Z", "Y", "ZZ"], "alpha": 2.0}):
        X = rng.uniform(0, 3, (5, 5))
        qc = build_circuit(prm, 5)
        R = np.array([Statevector(qc.assign_parameters(x)).probabilities() for x in X])
        worst = max(worst, float(np.abs(simulate_probabilities(X, prm) - R).max()))
    print(f"  [{'OK ' if worst < 1e-9 else 'FAIL'}] fast simulator vs Qiskit Statevector: max |dp| = {worst:.1e}")
    ok &= worst < 1e-9
    import yaml
    from qfd.noisy import noisy_probabilities
    cfg = yaml.safe_load(open(Path(__file__).parent / "configs" / "experiment.yaml"))
    zero = {"superconducting": dict(T1_us=1e12, T2_us=1e12, p_dep_1q=0, p_dep_2q=0, p_readout=0),
            "trapped_ion": dict(p_dep_1q=0, p_dep_2q=0, p_amp_damp=0, p_phase_damp=0, p_readout=0)}
    for arch in zero:
        X = rng.uniform(0, 3, (4, 4))
        prm = {"name": "ZZ", "reps": 2, "entanglement": "full"}
        t0 = time.perf_counter()
        P, st = noisy_probabilities(X, prm, arch, cfg["architectures"][arch], cfg["noise"], zero[arch])
        e = float(np.abs(P - simulate_probabilities(X, prm)).max())
        print(f"  [{'OK ' if e < 1e-8 else 'FAIL'}] Aer {arch:15s} zero-noise == ideal (max |dp| {e:.1e}); "
              f"compiled depth {st['compiled']['depth']}, 2q gates {st['compiled']['two_qubit_gates']}")
        ok &= e < 1e-8
    from xgboost import XGBClassifier
    XGBClassifier(n_estimators=5, n_jobs=1).fit(rng.normal(size=(50, 3)), np.r_[np.zeros(25), np.ones(25)])
    print("  [OK ] xgboost fit")
    with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as ex:
        assert list(ex.map(_square, [2, 3])) == [4.0, 9.0]
    print("  [OK ] multiprocessing (spawn)")
    cfg_data = Path(__file__).parent / cfg["paths"]["data_dir"]
    for name, d in cfg["datasets"].items():
        f = cfg_data / d["file"]
        print(f"  [{'OK ' if f.exists() else 'MISSING'}] {name:10s} {f}")
    print("\nALL CORE CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
