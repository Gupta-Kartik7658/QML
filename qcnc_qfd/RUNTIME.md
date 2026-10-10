# Runtime and memory

Measured for the paper's runs on one Windows laptop (Intel i5-12450H, 8 cores / 12 threads, 16 GB RAM), CPU only,
8 worker processes, full configuration (3 datasets × 3 feature maps × 3 seeds):

| Step | Wall-clock |
|---|---|
| `run_xgb_superconducting.py` (computes SFS and all superconducting noisy features; cached afterwards) | 1.4 h |
| `run_mlp_superconducting.py` (reuses the cache) | 3.4 h |
| `run_svm_superconducting.py` (reuses the cache) | 0.8 h |
| `run_sfs_vs_exhaustive.py --datasets creditcard banksim` | 0.4 h |
| `run_noise_sweep.py --arch superconducting` | 0.5 h |
| Trapped-ion step (`scripts/2_main_trapped_ion.*`, `--arch-only`, QSVM reference, noise sweep) | ≈ 3–4 h |
| `run_theory_check.py` | 0.4 h |
| `run_shot_budget.py` | 0.1 h |
| `run_hybrid.py` (189 tuned models + 243 paired bootstrap tests) | 1.5 h |

Single-task measurements (one core): SFS for 30 candidates up to 7 qubits, 189 evaluations: 4.4 min; noisy
density-matrix features ≈ 96 ms/row (superconducting) and ≈ 130 ms/row (trapped-ion) at 7 qubits, ≈ 8 ms/row for
the Z map; the vectorised ideal simulator ≈ 110 µs/row (ZZ, 7 qubits).

Memory: < 1 GB per worker. PaySim loading peaks at 2–3 GB in the main process. Disk: `cache/` ≈ 2 GB and
`results/` ≈ 0.8 GB for a full run (scores and models of every run are kept; only summaries are in git).
IBM hardware: 247 s of QPU time for the seven jobs (IBM Quantum Open Plan).
