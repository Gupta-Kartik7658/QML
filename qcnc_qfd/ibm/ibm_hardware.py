r"""IBM Quantum hardware runs (run this in the SEPARATE IBM venv: Qiskit 2.x + qiskit-ibm-runtime).

Never install qiskit-ibm-runtime into the experiment .venv (it would upgrade the pinned Qiskit 1.3.1).
The API key is read from the account saved earlier with QiskitRuntimeService.save_account(); it is never
printed, logged or written by this script.

Steps (run from this project folder; the IBM venv is described in ibm/README.md):
  0) experiment venv:  python run_ibm_hardware.py prepare            (writes results\main\ibm_hardware\bundle)
  1) OPTIONAL, IBM venv, free local rehearsal on a fake Heron (needs qiskit-aer in the IBM venv):
        python ibm\ibm_hardware.py submit --fake --copies-z 4 --copies-zz 3 --max-rows 60
  2) IBM venv, estimate the real cost (connects to IBM, submits nothing):
        python ibm\ibm_hardware.py submit --backend least_busy
  3) IBM venv, real submission (needs --yes; uses part of the 10 free minutes):
        python ibm\ibm_hardware.py submit --backend least_busy --yes
  4) IBM venv, collect the results (any time later; jobs may queue for hours):
        python ibm\ibm_hardware.py fetch --all
  5) experiment venv:  python run_ibm_hardware.py analyze

Design: several copies of the k-qubit feature-map circuit run side by side on disjoint qubits of the
156-qubit Heron chip, so each shot measures several data rows at once. No error mitigation, no twirling,
no dynamical decoupling: we want the raw device behaviour (the affine-law test needs it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager

PARTS = ("train", "val", "test")
ROOT = Path(__file__).resolve().parents[1]


def load_bundle(bdir: Path):
    B = dict(np.load(bdir / "bundle.npz"))
    meta = json.loads((bdir / "meta.json").read_text())
    circs = {}
    for fmap in meta["feature_maps"]:
        with open(bdir / f"circuit_{fmap}.qpy", "rb") as f:
            circs[fmap] = qpy.load(f)[0]
    return B, meta, circs


def verify_circuit(qc: QuantumCircuit, angles: np.ndarray, ideal_probs: np.ndarray, n_check: int = 5) -> float:
    """The loaded circuit must reproduce the exported ideal probabilities (guards against version drift)."""
    params = sorted(qc.parameters, key=lambda p: p.index)
    err = 0.0
    for r in range(min(n_check, len(angles))):
        sv = Statevector(qc.assign_parameters(dict(zip(params, angles[r]))))
        err = max(err, float(np.abs(sv.probabilities() - ideal_probs[r]).max()))
    return err


def wide_circuit(base: QuantumCircuit, copies: int):
    k = base.num_qubits
    params = sorted(base.parameters, key=lambda p: p.index)
    qr, cr = QuantumRegister(copies * k, "q"), ClassicalRegister(copies * k, "meas")
    qc = QuantumCircuit(qr, cr)
    for c in range(copies):
        pv = ParameterVector(f"c{c:03d}", k)
        qc.compose(base.assign_parameters(dict(zip(params, pv))), qubits=list(range(c * k, (c + 1) * k)), inplace=True)
    qc.barrier()
    qc.measure(qr, cr)
    return qc


def parameter_values(isa: QuantumCircuit, rows: np.ndarray, copies: int, k: int) -> np.ndarray:
    """(n_exec, n_params) values in the order of isa.parameters; row r -> exec r // copies, copy r % copies."""
    n_exec = int(np.ceil(len(rows) / copies))
    padded = np.concatenate([rows, np.repeat(rows[:1], n_exec * copies - len(rows), axis=0)])
    vals = np.empty((n_exec, isa.num_parameters))
    for col, prm in enumerate(isa.parameters):
        c = int(prm.vector.name[1:])
        vals[:, col] = padded[c::copies][:n_exec, prm.index]
    return vals


def flatten_parameters(isa: QuantumCircuit):
    """Replace every gate-angle ParameterExpression by a fresh plain Parameter (angles are then computed
    client-side). Works around IBM error 1502 (payload deserialization) seen for the ZZ map, whose angles
    are products of two inputs, 2*(pi - x_i)*(pi - x_j). The global phase is dropped (no effect on counts)."""
    from qiskit.circuit import Parameter, ParameterExpression
    flat = isa.copy()
    flat.global_phase = 0.0
    exprs, cache = [], {}
    for i, ci in enumerate(flat.data):
        ps = ci.operation.params
        if not any(isinstance(q, ParameterExpression) and q.parameters for q in ps):
            continue
        new = []
        for q in ps:
            if isinstance(q, ParameterExpression) and q.parameters:
                key = str(q)
                if key not in cache:
                    cache[key] = Parameter(f"t{len(exprs):06d}")
                    exprs.append((cache[key], q))
                new.append(cache[key])
            else:
                new.append(q)
        op = ci.operation.to_mutable()
        op.params = new
        flat.data[i] = ci.replace(operation=op)
    return flat, exprs


def flat_values(flat: QuantumCircuit, exprs, vals_orig: np.ndarray, orig_params) -> np.ndarray:
    """Evaluate every flattened angle for every execution (vectorised with sympy, exact fallback via bind)."""
    try:
        import sympy
    except ImportError:  # Qiskit 2.x does not depend on sympy -> exact per-execution binding
        sympy = None
    col = {str(p): i for i, p in enumerate(orig_params)}
    out = np.empty((len(vals_orig), flat.num_parameters))
    pos = {p: i for i, p in enumerate(flat.parameters)}
    for th, ex in exprs:
        prs = list(ex.parameters)
        try:
            if sympy is None:
                raise ImportError
            sym = ex.sympify()
            args = [sympy.Symbol(str(q)) for q in prs]
            f = sympy.lambdify(args, sym, modules="numpy")
            v = np.broadcast_to(np.asarray(f(*[vals_orig[:, col[str(q)]] for q in prs]), dtype=float), (len(vals_orig),))
        except Exception:
            v = np.array([float(ex.bind({q: vals_orig[e, col[str(q)]] for q in prs}).numeric()) for e in range(len(vals_orig))])
        out[:, pos[th]] = v
    return out


def z_from_bitarray(ba, n_rows: int, copies: int, k: int) -> np.ndarray:
    """<Z_j> per data row from a BitArray of shape (n_exec,) with copies*k bits."""
    bits = np.unpackbits(ba.array, axis=-1)[..., ::-1][..., :copies * k]  # (n_exec, shots, nbits), bit i at index i
    z = (1.0 - 2.0 * bits.astype(np.float64)).mean(axis=1)                # (n_exec, nbits)
    z = z.reshape(z.shape[0], copies, k).reshape(-1, k)                  # row = exec * copies + copy
    return z[:n_rows]


def get_backend(args):
    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        return FakeFez(), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=args.instance) if args.instance else QiskitRuntimeService()
    if args.backend == "least_busy":
        b = service.least_busy(operational=True, simulator=False, min_num_qubits=100)
    else:
        b = service.backend(args.backend)
    return b, service


def cmd_submit(args):
    out = Path(args.results_dir) / args.run_tag / "ibm_hardware" / args.subdir
    B, meta, circs = load_bundle(out / "bundle")
    backend, _ = get_backend(args)
    name = backend.name
    print(f"backend: {name} ({backend.num_qubits} qubits)")
    pm = generate_preset_pass_manager(optimization_level=args.opt_level, backend=backend, seed_transpiler=1234)
    rep_delay = getattr(backend, "default_rep_delay", None)
    if rep_delay is None:
        try:
            rep_delay = backend.configuration().default_rep_delay
        except Exception:
            rep_delay = None
    rep_delay = float(rep_delay) if rep_delay else 250e-6
    if rep_delay > 0.1:  # some versions report microseconds
        rep_delay *= 1e-6
    pubs, plan = [], {}
    total_exec = 0
    for fmap, copies in (("Z", args.copies_z), ("ZZ", args.copies_zz), ("Pauli", args.copies_zz)):
        if fmap not in circs or fmap not in args.fmaps:
            continue
        base = circs[fmap]
        k = base.num_qubits
        full = {p: len(B[f"y_{p}"]) for p in PARTS}
        # rehearsal (--max-rows N): the first N/3 rows of EACH partition; real run: all rows
        counts = {p: (min(full[p], max(2, args.max_rows // 3)) if args.max_rows else full[p]) for p in PARTS}
        rows = np.concatenate([B[f"{fmap}_angles_{p}"][:counts[p]] for p in PARTS])
        ideal = np.concatenate([B[f"{fmap}_ideal_probs_{p}"][:counts[p]] for p in PARTS])
        err = verify_circuit(base, rows, ideal)
        print(f"{fmap}: circuit check vs exported ideal probabilities: max |dp| = {err:.1e}")
        if err > 1e-8:
            sys.exit("circuit mismatch -> stop (version drift?)")
        qc = wide_circuit(base, copies)
        isa = pm.run(qc)
        ops = dict(isa.count_ops())
        n2q = sum(v for g, v in ops.items() if g in ("cz", "ecr", "cx", "rzz"))
        vals = parameter_values(isa, rows, copies, k)
        if args.flatten:  # plain parameters only (see flatten_parameters)
            flat, exprs = flatten_parameters(isa)
            fvals = flat_values(flat, exprs, vals, list(isa.parameters))
            # self-check on 2 executions: flattened circuit == original circuit after binding
            for e in range(min(2, len(vals))):
                a = isa.assign_parameters(vals[e]).remove_final_measurements(inplace=False)
                b = flat.assign_parameters(fvals[e]).remove_final_measurements(inplace=False)
                ga = [float(x) for ci in a.data for x in ci.operation.params if not hasattr(x, "parameters")]
                gb = [float(x) for ci in b.data for x in ci.operation.params if not hasattr(x, "parameters")]
                if len(ga) != len(gb) or np.abs(np.array(ga) - np.array(gb)).max(initial=0) > 1e-9:
                    sys.exit("flattening self-check failed -> not submitting")
            print(f"{fmap}: flattened {len(exprs)} parameter expressions into plain parameters (self-check OK)")
            isa, vals = flat, fvals
        total_exec += len(vals)
        layout = isa.layout.final_index_layout() if isa.layout is not None else list(range(qc.num_qubits))
        plan[fmap] = {"k": k, "copies": copies, "n_rows": int(len(rows)), "n_exec": int(len(vals)),
                      "rows_per_part": {p: int(c) for p, c in counts.items()}, "full_rows_per_part": full,
                      "isa_depth": int(isa.depth()), "isa_2q_gates": int(n2q), "isa_ops": {a: int(b) for a, b in ops.items()},
                      "physical_qubits": [int(q) for q in layout[:copies * k]], "verify_max_abs_dp": err,
                      "pub_index": len(pubs)}
        pubs.append((isa, vals))
        print(f"{fmap}: k={k}, {copies} copies -> {copies * k} qubits, depth {isa.depth()}, 2q gates {n2q}, "
              f"{len(rows)} rows in {len(vals)} executions")
    est = total_exec * args.shots * (rep_delay + 10e-6) + 5.0 * len(pubs)
    print(f"estimated QPU time: ~{est:.0f} s (rep_delay {rep_delay * 1e6:.0f} us, {args.shots} shots; "
          f"Open plan budget = 600 s per 28 days)")
    if not args.fake and not args.yes:
        print("dry run only. Add --yes to submit.")
        return
    from qiskit_ibm_runtime import SamplerV2
    # one job per feature map (one PUB per job): simplest to track, and a failure only costs that map
    for fmap, pl in plan.items():
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = args.shots
        job = sampler.run([pubs[pl["pub_index"]]])
        jid = job.job_id()
        rec = {"job_id": jid, "backend": name, "shots": args.shots, "plan": {fmap: dict(pl, pub_index=0)},
               "fake": bool(args.fake), "submitted_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
               "opt_level": args.opt_level, "estimated_qpu_seconds_all_maps": est,
               "mitigation": "none (no twirling, no DD, no readout mitigation)"}
        (out / f"job_{name}_{jid}.json").write_text(json.dumps(rec, indent=1))
        print(f"{fmap}: submitted job {jid}")
        if args.fake:
            save_results(job, rec, out)
    if not args.fake:
        print("collect later with:  python ibm\\ibm_hardware.py fetch --all")


def save_results(job, rec, out):
    res = job.result()
    arrays = {}
    for fmap, pl in rec["plan"].items():
        pub = res[pl["pub_index"]]
        z = z_from_bitarray(pub.data.meas, pl["n_rows"], pl["copies"], pl["k"])
        n, full = pl["rows_per_part"], pl["full_rows_per_part"]
        bounds = np.cumsum([0, n["train"], n["val"], n["test"]])
        for i, p in enumerate(PARTS):
            zp = z[bounds[i]:bounds[i + 1]]
            if len(zp) < full[p]:  # --max-rows rehearsal: rows not measured are NaN (analyze drops them)
                zp = np.vstack([zp, np.full((full[p] - len(zp), pl["k"]), np.nan)])
            arrays[f"{fmap}_z_{p}"] = zp
        arrays[f"{fmap}_raw_bits"] = pub.data.meas.array
    usage = None
    try:
        usage = job.usage()
    except Exception:
        try:
            usage = job.metrics().get("usage", {}).get("quantum_seconds")
        except Exception:
            pass
    rec = dict(rec, usage_quantum_seconds=usage, fetched_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    path = out / f"hw_{rec['backend']}_{rec['job_id']}.npz"
    np.savez_compressed(path, meta_json=np.array(json.dumps(rec)), **arrays)
    print(f"results -> {path} | QPU usage: {usage}")


def cmd_fetch(args):
    base = Path(args.results_dir) / args.run_tag / "ibm_hardware"
    recs = []
    for f in sorted(base.rglob("job_*.json")):  # every experiment folder
        r = json.loads(f.read_text())
        if not r.get("fake"):
            r["_dir"] = str(f.parent)
            recs.append(r)
    if args.job:
        recs = [r for r in recs if r["job_id"] in args.job]
    else:  # all recorded jobs that have no result file yet
        recs = [r for r in recs if not (Path(r["_dir"]) / f"hw_{r['backend']}_{r['job_id']}.npz").exists()]
    if not recs:
        sys.exit(f"nothing to fetch under {base}")
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=args.instance) if args.instance else QiskitRuntimeService()
    for rec in recs:
        job = service.job(rec["job_id"])
        st = str(job.status()).upper()
        print(f"job {rec['job_id']} ({list(rec['plan'])[0]} map): {st}")
        if any(w in st for w in ("QUEUED", "RUNNING", "INITIALIZING", "VALIDATING")):
            print("   not finished yet -> run fetch again later")
            continue
        if any(w in st for w in ("ERROR", "CANCELLED", "FAILED")):
            try:
                msg = job.error_message()
            except Exception:
                msg = ""
            print(f"   job did not complete: {st} {msg or ''}")
            continue
        save_results(job, {k: v for k, v in rec.items() if k != "_dir"}, Path(rec["_dir"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--backend", default="least_busy", help="ibm_fez | ibm_kingston | ibm_marrakesh | least_busy")
    s.add_argument("--fake", action="store_true", help="local rehearsal on FakeFez (no account, no budget)")
    s.add_argument("--fmaps", nargs="+", default=["Z", "ZZ"])
    s.add_argument("--copies-z", type=int, default=14)
    s.add_argument("--copies-zz", type=int, default=8)
    s.add_argument("--shots", type=int, default=1024)
    s.add_argument("--opt-level", type=int, default=1)
    s.add_argument("--max-rows", type=int, default=0, help="only the first N rows (rehearsal)")
    s.add_argument("--yes", action="store_true", help="really submit to IBM hardware")
    s.add_argument("--flatten", action="store_true",
                   help="replace parameter expressions by plain parameters (fix for IBM error 1502 on the ZZ map)")
    s.add_argument("--instance", default=None)
    s.add_argument("--run-tag", default="main")
    s.add_argument("--results-dir", default=str(ROOT / "results"))
    s.add_argument("--subdir", default="", help="experiment folder under ibm_hardware/ (must match prepare --subdir)")
    f = sub.add_parser("fetch")
    f.add_argument("--job", nargs="+", default=None, help="job id(s); default: all recorded jobs without results")
    f.add_argument("--all", action="store_true", help="(default behaviour) fetch every pending recorded job")
    f.add_argument("--instance", default=None)
    f.add_argument("--run-tag", default="main")
    f.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    cmd_submit(args) if args.cmd == "submit" else cmd_fetch(args)


if __name__ == "__main__":
    main()
