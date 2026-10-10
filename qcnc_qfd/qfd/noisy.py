r"""Architecture-specific noisy simulation (Qiskit Aer density-matrix, CPU only).

Two *simulation configurations* inspired by hardware families (NOT runs on real devices):

superconducting  : 7-qubit IBM Falcon-style H coupling map (0-1,1-2,1-3,3-5,4-5,5-6), basis
                   {rz, sx, x, cx}; SWAP routing by the Qiskit transpiler; thermal relaxation
                   (T1, T2, gate durations) composed with depolarising error on sx/x/cx;
                   rz is virtual (error-free, zero duration); symmetric readout error.
trapped_ion      : all-to-all connectivity, basis {rx, ry, rz, rxx} (Molmer-Sorensen-type XX);
                   depolarising + amplitude damping + phase damping on rx/ry and on each qubit
                   of rxx; rz virtual; symmetric readout error.

Because the ideal unitary is identical on both, ideal features are architecture-independent.
Architecture differences only enter through (i) the compiled circuit (routing, native gates,
depth) *combined with* (ii) the noise model. Idle/crosstalk/leakage errors are not modelled.

Readout error is applied *exactly* to the simulated probability vector with the per-qubit
confusion matrix [[1-p, p], [p, 1-p]] (in the earlier version on the main branch readout error was attached to a
saved density matrix, where it has no effect).
"""
from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, amplitude_damping_error, depolarizing_error,
                              phase_damping_error, thermal_relaxation_error)

from .featuremaps import build_circuit, circuit_stats, observable_matrices, to_angles

FALCON7_EDGES = [(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)]
FALCON7_ORDER = [0, 1, 2, 3, 5, 4, 6]  # every prefix is a connected sub-graph


def coupling_for(arch_cfg: dict, k: int):
    if arch_cfg["coupling"] == "all_to_all":
        return None
    if arch_cfg["coupling"] == "falcon7_h":
        if k > 7:
            raise ValueError("falcon7_h supports at most 7 qubits")
        phys = FALCON7_ORDER[:k]
        relabel = {p: i for i, p in enumerate(phys)}
        edges = [(relabel[a], relabel[b]) for a, b in FALCON7_EDGES if a in relabel and b in relabel]
        if k == 1:
            return None
        return CouplingMap([list(e) for e in edges] + [[b, a] for a, b in edges])
    if arch_cfg["coupling"] == "linear":
        return CouplingMap.from_line(k) if k > 1 else None
    raise ValueError(arch_cfg["coupling"])


def noise_model_for(arch: str, a: dict) -> NoiseModel:
    nm = NoiseModel(basis_gates=list(a["basis_gates"]))
    if arch == "superconducting":
        t1, t2 = a["T1_us"] * 1e3, min(a["T2_us"], 2 * a["T1_us"]) * 1e3  # ns; enforce T2 <= 2 T1
        e1 = thermal_relaxation_error(t1, t2, a["gate_time_1q_ns"]).compose(depolarizing_error(a["p_dep_1q"], 1))
        th2 = thermal_relaxation_error(t1, t2, a["gate_time_2q_ns"])
        e2 = th2.expand(th2).compose(depolarizing_error(a["p_dep_2q"], 2))
        nm.add_all_qubit_quantum_error(e1, ["sx", "x"])
        nm.add_all_qubit_quantum_error(e2, ["cx"])
    elif arch == "trapped_ion":
        damp = amplitude_damping_error(a["p_amp_damp"]).compose(phase_damping_error(a["p_phase_damp"]))
        e1 = depolarizing_error(a["p_dep_1q"], 1).compose(damp)
        e2 = depolarizing_error(a["p_dep_2q"], 2).compose(damp.expand(damp))
        nm.add_all_qubit_quantum_error(e1, ["rx", "ry"])
        nm.add_all_qubit_quantum_error(e2, ["rxx"])
    else:
        raise ValueError(arch)
    return nm


def compile_for_arch(fmap: dict, k: int, arch: str, arch_cfg: dict, noise_cfg: dict):
    """Transpile the logical feature map to the architecture. Returns (circuit, layout, stats)."""
    logical = build_circuit(fmap, k)
    cm = coupling_for(arch_cfg, k)
    tc = transpile(logical, basis_gates=list(arch_cfg["basis_gates"]), coupling_map=cm,
                   initial_layout=list(range(k)) if cm is not None else None,
                   optimization_level=int(noise_cfg.get("optimization_level", 1)),
                   seed_transpiler=int(noise_cfg.get("transpile_seed", 1234)))
    if tc.layout is not None:
        final = list(tc.layout.final_index_layout())
    else:
        final = list(range(k))
    log_stats = circuit_stats(transpile(logical, basis_gates=["u", "cx"], optimization_level=0))
    stats = {"logical_u_cx": log_stats, "compiled": circuit_stats(tc), "final_layout": final,
             "swap_overhead_2q": circuit_stats(tc)["two_qubit_gates"] - log_stats["two_qubit_gates"]}
    return tc, final, stats


def apply_readout(P: np.ndarray, k: int, p: float) -> np.ndarray:
    """Exact symmetric readout error on probability vectors (B, 2^k)."""
    if not p:
        return P
    M = np.array([[1 - p, p], [p, 1 - p]])  # M[measured, true]
    T = P.reshape((P.shape[0],) + (2,) * k)
    for q in range(k):
        axis = k - q  # qubit q is axis (k - q) (little-endian, batch axis first)
        T = np.moveaxis(np.tensordot(T, M, axes=([axis], [1])), -1, axis)
    return T.reshape(P.shape[0], -1)


def noisy_probabilities(U: np.ndarray, fmap: dict, arch: str, arch_cfg: dict, noise_cfg: dict,
                        noise_overrides: dict | None = None, logger=None):
    """Noisy probabilities of the logical qubits for every row of U ([0,1] inputs). Returns (P, stats)."""
    X = to_angles(U, fmap)
    B, k = X.shape
    a = dict(arch_cfg)
    if noise_overrides:
        a.update(noise_overrides)
    tc, final, stats = compile_for_arch(fmap, k, arch, a, noise_cfg)
    circ = tc.copy()
    circ.save_probabilities(qubits=final, label="p")  # bit j of the result <-> logical qubit j
    sim = AerSimulator(method="density_matrix", noise_model=noise_model_for(arch, a),
                       max_parallel_threads=1, max_parallel_experiments=1, max_parallel_shots=1)
    params = sorted(circ.parameters, key=lambda p: int(p.index))
    if len(params) != k:
        raise RuntimeError(f"compiled circuit has {len(params)} parameters, expected {k}")
    bs = int(noise_cfg.get("batch_size", 256))
    P = np.empty((B, 2 ** k))
    for s in range(0, B, bs):
        Xb = X[s:s + bs]
        binds = [{p: Xb[:, int(p.index)].tolist() for p in params}]
        res = sim.run(circ, parameter_binds=binds, shots=None).result()
        for i in range(len(Xb)):
            P[s + i] = np.asarray(res.data(i)["p"], dtype=float)
    P = np.clip(P, 0.0, None)
    P /= P.sum(1, keepdims=True)
    P = apply_readout(P, k, float(a.get("p_readout", 0.0)))
    return P, stats


def sample_shots(P: np.ndarray, shots: int, seed: int) -> np.ndarray:
    """Finite-shot estimate of the probability vectors (multinomial sampling, seeded)."""
    rng = np.random.default_rng(seed)
    P = np.clip(np.asarray(P, dtype=float), 0.0, None)
    P = P / P.sum(1, keepdims=True)
    return rng.multinomial(int(shots), P) / float(shots)


def noisy_features(U, fmap, arch, arch_cfg, noise_cfg, extended=False, seed=0, noise_overrides=None,
                   shots=None):
    P, stats = noisy_probabilities(U, fmap, arch, arch_cfg, noise_cfg, noise_overrides)
    if shots:
        P = sample_shots(P, int(shots), seed)
    k = np.asarray(U).shape[1]
    M, names = observable_matrices(k, extended)
    return (P @ M).astype(np.float32), names, stats
