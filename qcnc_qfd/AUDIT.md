# Audit of the previous code and paper (verified by running the code / inspecting saved outputs)

| # | Finding | Evidence | Consequence / fix |
|---|---|---|---|
| 1 | **Features were selected on the test set.** `greed_run.py` ranks subsets by AUC on the same 200-sample test set that Table III reports. | `train_qsvm_with_kernel(K_train, K_test, …, y_test)` inside the selection loop | Table III AUCs are optimistically biased. Fix: SFS on validation only. |
| 2 | **QGFS ≠ Algorithm 1.** Code = exhaustive C(30,2)=435 pairs + greedy forward to 7 (~585 evals), always running to 7. Paper = exhaustive at every size with early break. | `hybrid_quantum_feature_selection` | Paper text was wrong. Fix: SFS, as described. |
| 3 | **"Shot-based, 1024 shots" is false.** `FidelityQuantumKernel` defaults to `ComputeUncompute(Sampler())` (exact). The `AerSimulator` object is created but never used. | qiskit-machine-learning 0.8.0 source | Fix: exact vs shots are now explicit, separate conditions. |
| 4 | **Ideal "architectures" = entanglement only.** Superconducting = `entanglement='linear'`, trapped-ion = `'full'`. No coupling map, native gates or noise. | `ZZbest.txt` vs `Full_ZZFeatureMap_*.txt` configs | The Table III architecture comparison was a topology comparison. Fix: real compilation + noise models. |
| 5 | **Readout error had no effect.** It was attached to a saved density matrix. | Every readout sweep in `noise_sensitivity_*.json` has 1 unique AUC | The P_READOUT panels in Figs. 6–7 are artefacts. Fix: exact readout on probabilities. |
| 6 | `p` gates of the ZZ/Pauli maps received no 1-qubit noise (the noise list was u1/u2/u3/rz/sx/x/id). | transpiled gate counts | Under-estimated noise. |
| 7 | The Z-map noise sweeps used the Pauli map's feature subset. | `noise_sensitivity_Z.json` features | Mismatch with Table III. |
| 8 | Pauli['X','ZZ'] at reps=1: the X-rotation of \|+⟩ is a global phase. In the Z basis Pauli['X','ZZ'] ≡ ZZFeatureMap at any reps. | `tests/test_core.py` | Pauli vs ZZ differences were small by construction. Fix: ['Z','Y','ZZ']. |
| 9 | ⟨Zᵢ⟩ ≡ 0 for all three maps at reps=1. | `tests/test_core.py` | The new representation needs reps ≥ 2. |
| 10 | Evaluation: balanced 400/200 subsample, single seed, "FAR" = FP/TP, accuracy as a headline metric. | `greed_run.py` | Fix: original prevalence, 3 seeds, standard metrics with CIs. |
| 11 | Environment: `requirements.txt` was frozen on Windows / Python 3.12. `pywinpty` fails on Linux; `qiskit-aer==0.15.1` cannot install on Python 3.13. | Fresh-install attempts | Use Python 3.12 + `requirements-experiments.txt` (same pins). |
| 12 | Dataset: 1 081 exact duplicate rows (19 frauds) in `creditcard.csv`. | loader | Removed before splitting: 283 726 rows, 473 frauds. |

**Reused:** the dataset, the noise parameter values, the Qiskit feature-map classes, and the logging/plot ideas.
**Replaced:** the selection algorithm, evaluation protocol, noise simulation, and project structure (≈30 standalone
scripts → `qfd/` package + 6 runner scripts + tests).
