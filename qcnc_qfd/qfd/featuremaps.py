r"""Quantum feature maps and an exact, vectorised ideal simulator.

Quantum feature definition (primary representation)
----------------------------------------------------
For an input x in R^k (k = number of SFS-selected features, one qubit per feature):

    |psi(x)> = U_phi(x) |0...0>,     z_i(x) = <psi(x)| Z_i |psi(x)>,     z(x) = [z_1, ..., z_k]

Extended (ablation) representation adds all two-body correlators z_ij(x) = <Z_i Z_j>.
Both are computed from the computational-basis probabilities p_b(x) = |<b|psi(x)>|^2:
    z_i = sum_b p_b (1 - 2 b_i),     z_ij = sum_b p_b (1 - 2 b_i)(1 - 2 b_j).

The circuits are the *Qiskit library* circuits (ZFeatureMap / ZZFeatureMap / PauliFeatureMap,
qiskit==1.3.1). For speed they are decomposed once into {u, cx} and executed on a batch of
samples with NumPy (each sample has its own gate angles). This is mathematically identical
to qiskit.quantum_info.Statevector — tests/test_featuremaps.py checks agreement to 1e-10 —
and ~1000x faster, which is what makes SFS over ~190 candidate subsets feasible on a CPU.

NOTE: with reps=1 all three maps give z_i(x) == 0 for every x (H layer followed only by
gates that are diagonal, or X-rotations of |+> which are global phases). reps>=2 is required.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap


def no_wrap_range(alpha: float = 2.0) -> list:
    """Largest interval ending at pi on which alpha*(pi-x_i)(pi-x_j) spans <= 2*pi."""
    w = float(np.sqrt(2 * np.pi / alpha))
    return [float(np.pi - w), float(np.pi)]


def fmap_params(cfg: dict, name: str) -> dict:
    """Canonical, hashable description of a feature map configuration (input range resolved)."""
    fm = cfg["feature_maps"]
    p = {"name": name, "reps": int(fm["reps"])}
    p.update({k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in (fm.get(name) or {}).items()})
    r = p.get("input_range", [-1.0, 1.0])
    if r == "no_wrap":
        r = no_wrap_range(float(p.get("alpha", 2.0)))
    p["input_range"] = [float(r[0]), float(r[1])]
    return p


def to_angles(U: np.ndarray, params: dict) -> np.ndarray:
    """[0,1]-scaled features -> circuit parameters (identity if params has no input_range)."""
    U = np.asarray(U, dtype=float)
    if "input_range" not in params:
        return U
    lo, hi = params["input_range"]
    return lo + (hi - lo) * U


def build_circuit(params: dict, n_qubits: int) -> QuantumCircuit:
    name, reps = params["name"], params["reps"]
    if name == "Z":
        return ZFeatureMap(n_qubits, reps=reps)
    if name == "ZZ":
        if n_qubits == 1:  # no pairs: identical to ZFeatureMap
            return ZFeatureMap(1, reps=reps)
        return ZZFeatureMap(n_qubits, reps=reps, entanglement=params.get("entanglement", "full"))
    if name == "Pauli":
        paulis = list(params.get("paulis", ["X", "ZZ"]))
        if n_qubits == 1:
            paulis = [p for p in paulis if len(p) == 1] or ["Z"]
        return PauliFeatureMap(n_qubits, reps=reps, paulis=paulis,
                               entanglement=params.get("entanglement", "full"),
                               alpha=float(params.get("alpha", 2.0)))
    raise ValueError(f"unknown feature map {name}")


def circuit_stats(qc: QuantumCircuit) -> dict:
    ops = dict(qc.count_ops())
    two_q = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "rxx", "rzz", "swap", "ms"))
    return {"n_qubits": qc.num_qubits, "depth": qc.depth(), "size": qc.size(),
            "two_qubit_gates": int(two_q), "ops": {k: int(v) for k, v in ops.items()}}


# ----------------------------------------------------------------------------- compiled program
class _Program:
    """Circuit compiled to a list of ('u', q, f_theta, f_phi, f_lam) / ('cx', c, t) ops."""

    def __init__(self, qc: QuantumCircuit):
        self.n = qc.num_qubits
        params = list(qc.parameters)  # ParameterVector elements x[0..n-1], sorted
        self.order = [int(p.index) for p in params]
        syms = [sympy.Symbol(f"_p{i}") for i in range(len(params))]
        tq = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
        self.ops = []
        for ins in tq.data:
            name = ins.operation.name
            qs = [tq.find_bit(q).index for q in ins.qubits]
            if name == "u":
                fs = [self._compile(v, params, syms) for v in ins.operation.params]
                self.ops.append(("u", qs[0], *fs))
            elif name == "cx":
                self.ops.append(("cx", qs[0], qs[1]))
            elif name == "barrier":
                continue
            else:
                raise RuntimeError(f"unexpected gate {name}")
        self.n_params = len(params)

    @staticmethod
    def _compile(v, params, syms):
        if isinstance(v, ParameterExpression) and v.parameters:
            expr = v.sympify()
            rep = {sympy.Symbol(p.name): s for p, s in zip(params, syms)}
            expr = expr.xreplace(rep)
            f = sympy.lambdify(syms, expr, modules="numpy")
            return ("f", f)
        return ("c", float(v))


@lru_cache(maxsize=64)
def _program(key: tuple, n_qubits: int) -> _Program:
    params = dict(key)
    if "paulis" in params:
        params["paulis"] = list(params["paulis"])
    return _Program(build_circuit(params, n_qubits))


def _key(params: dict) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in params.items()
                        if k != "input_range"))


def _eval(spec, cols, B):
    kind, v = spec
    if kind == "c":
        return np.full(B, v)
    out = v(*cols)
    return np.broadcast_to(np.asarray(out, dtype=float), (B,))


def simulate_probabilities(X: np.ndarray, params: dict, chunk: int = 8192) -> np.ndarray:
    """Exact |<b|psi(x)>|^2 for every row of X (shape (B, k)) -> (B, 2**k), qiskit bit order."""
    X = np.asarray(X, dtype=float)
    B, k = X.shape
    prog = _program(_key(params), k)
    out = np.empty((B, 2 ** k), dtype=np.float64)
    for s in range(0, B, chunk):
        Xb = X[s:s + chunk]
        out[s:s + chunk] = _run_program(prog, Xb)
    return out


def _run_program(prog: _Program, Xb: np.ndarray) -> np.ndarray:
    return np.abs(_run_program_sv(prog, Xb)) ** 2


# ----------------------------------------------------------------------------- observables
def observable_matrices(k: int, extended: bool):
    idx = np.arange(2 ** k)
    bits = (idx[:, None] >> np.arange(k)) & 1
    zs = (1 - 2 * bits).astype(np.float64)  # (2^k, k): eigenvalue of Z_i on basis state b
    cols, names = [zs], [f"Z{i}" for i in range(k)]
    if extended and k > 1:
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        cols.append(np.stack([zs[:, i] * zs[:, j] for i, j in pairs], axis=1))
        names += [f"Z{i}Z{j}" for i, j in pairs]
    return np.concatenate(cols, axis=1), names


def expectations_from_probs(P: np.ndarray, k: int, extended: bool = False):
    M, names = observable_matrices(k, extended)
    return (P @ M).astype(np.float32), names


def ideal_features(U: np.ndarray, params: dict, extended: bool = False, chunk: int = 8192):
    """U: [0,1]-scaled inputs (mapped by params['input_range']). Returns (features float32, names)."""
    X = to_angles(U, params)
    k = X.shape[1]
    M, names = observable_matrices(k, extended)
    out = np.empty((len(X), M.shape[1]), dtype=np.float32)
    for s in range(0, len(X), chunk):
        out[s:s + chunk] = simulate_probabilities(X[s:s + chunk], params, chunk) @ M
    return out, names


def ideal_probabilities(U: np.ndarray, params: dict) -> np.ndarray:
    return simulate_probabilities(to_angles(U, params), params)


def statevectors(U: np.ndarray, params: dict, chunk: int = 8192) -> np.ndarray:
    """Full statevectors (for the fidelity-kernel QSVM reference). U in [0,1]."""
    X = to_angles(U, params)
    B, k = X.shape
    prog = _program(_key(params), k)
    out = np.empty((B, 2 ** k), dtype=np.complex128)
    for s in range(0, B, chunk):
        Xb = X[s:s + chunk]
        out[s:s + chunk] = _run_program_sv(prog, Xb)
    return out


def _run_program_sv(prog, Xb):
    """Batched statevector evolution using strided tensor views (no per-gate copies)."""
    B, n = Xb.shape[0], prog.n
    cols = [Xb[:, i] for i in prog.order]
    psi = np.zeros((B,) + (2,) * n, dtype=np.complex128)
    psi[(slice(None),) + (0,) * n] = 1.0

    def ax(q):  # qiskit little-endian: qubit 0 is the last tensor axis
        return n - q

    full = [slice(None)] * (n + 1)
    for op in prog.ops:
        if op[0] == "u":
            q = op[1]
            ft, fp, fl = op[2:]
            s0 = list(full); s0[ax(q)] = 0; s0 = tuple(s0)
            s1 = list(full); s1[ax(q)] = 1; s1 = tuple(s1)
            shape = (B,) + (1,) * (n - 1)
            if ft == ("c", 0.0) and fp == ("c", 0.0):          # phase gate diag(1, e^{i lam})
                lam = _eval(fl, cols, B)
                psi[s1] *= np.exp(1j * lam).reshape(shape)
                continue
            th, ph, la = (_eval(f, cols, B) for f in (ft, fp, fl))
            c, s = np.cos(th / 2), np.sin(th / 2)
            a = c.astype(np.complex128).reshape(shape)
            b = (-np.exp(1j * la) * s).reshape(shape)
            cc = (np.exp(1j * ph) * s).reshape(shape)
            d = (np.exp(1j * (ph + la)) * c).reshape(shape)
            v0 = psi[s0].copy()
            v1 = psi[s1]
            psi[s0] = a * v0 + b * v1
            psi[s1] = cc * v0 + d * v1
        else:  # cx(control, target)
            c_, t_ = op[1], op[2]
            s10 = list(full); s10[ax(c_)] = 1; s10[ax(t_)] = 0; s10 = tuple(s10)
            s11 = list(full); s11[ax(c_)] = 1; s11[ax(t_)] = 1; s11 = tuple(s11)
            tmp = psi[s10].copy()
            psi[s10] = psi[s11]
            psi[s11] = tmp
    return psi.reshape(B, 2 ** n)
