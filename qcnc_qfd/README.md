# Noise-invariant quantum features for financial fraud detection

Code, results and paper source for *"Entanglement Does Not Pay (Yet): Noise-Invariant Quantum Features for Fraud
Detection, Proven and Measured on IBM Heron Hardware"* (QCNC 2027 submission).

The pipeline encodes transactions with Qiskit feature maps (Z, ZZ, Pauli) and uses the measured expectation values
⟨Zᵢ⟩ as features for classical classifiers (XGBoost, MLP, RBF-SVM). Datasets are the European credit-card dataset
(real) and PaySim / BankSim (synthetic). Evaluation is leakage-free and at the original fraud prevalence. Noise is
studied in simulation (superconducting- and trapped-ion-like models) and on IBM Heron processors.

**Main findings**

* **Product (Z) map, exact affine noise law.** For the Z map, any noise made of phase-covariant single-qubit
  channels plus readout error maps every feature to α + β·⟨Zᵢ⟩. Standardised classifiers are therefore exactly
  noise-invariant, and gate noise only multiplies the shot budget by 1/β² (`theory/NOISE_INVARIANCE.md`).
* **The law holds in simulation and on hardware.** Residual ≤ 2×10⁻¹⁵ in simulation. On 360 physical qubits of
  `ibm_marrakesh` and `ibm_fez`, the per-qubit affine fit has median R² = 0.997 and a residual ≈ 1.1× shot noise.
* **Entangling ZZ map, geometric signal decay.** The retained signal decays as β ≈ β₀·ηᵍ in the number g of
  two-qubit gates per qubit, with η ≈ 0.990 on Heron hardware (k = 3, 5, 7 qubits).
* **No quantum advantage.** Quantum features never beat all classical features. Hybrid gains are reproduced by
  random Fourier features of the same inputs.

## Repository layout

```
qfd/                      library: data loading, leakage-free preprocessing, SFS, feature maps and fast
                          simulator, noise models, classifiers, metrics, caching, experiment orchestration
configs/experiment.yaml   every setting of the paper (seeds, sampling caps, maps, noise parameters, grids)
run_<clf>_<arch>.py       main experiments: clf in {xgb, mlp, svm}, arch in {superconducting, trapped_ion}
run_sfs_vs_exhaustive.py  SFS vs exhaustive subset search
run_noise_sweep.py        one-at-a-time noise-parameter sweeps
run_qsvm_reference.py     fidelity-kernel QSVM reference
run_theory_check.py       numerical check of the affine noise law and of pipeline invariance
run_shot_budget.py        accuracy vs number of shots (ideal / both noise models)
run_hybrid.py             classical + quantum features vs classical, with a random-Fourier-feature control
run_ibm_hardware.py       IBM hardware: prepare data bundles / analyse results (experiment venv)
ibm/                      IBM hardware submission and retrieval (separate Qiskit 2.x venv, see ibm/README.md)
analyze_results.py        aggregates all runs into tables and figures (results/<tag>/_paper/)
check_env.py              environment and simulator correctness check
scripts/                  numbered scripts that reproduce the paper (Windows .bat and Linux/macOS .sh)
tests/                    unit and end-to-end tests (mock data)
theory/                   proposition, proof and numerical evidence
paper/                    LaTeX source, figure scripts and figure data of the paper
results/main/             result summaries behind every number in the paper (tables, figures, hardware data)
METHODOLOGY.md            design and definitions;  AUDIT.md: audit of the earlier code on the main branch
MODEL_SPECS.md            every trained model;  RUNTIME.md: measured runtimes
```

## Setup

Python **3.12** is required (`qiskit-aer==0.15.1` has no wheels for Python 3.13).

```bash
python3.12 -m venv .venv
source .venv/bin/activate           # Windows (PowerShell): .venv\Scripts\Activate.ps1
python -m pip install -r requirements-experiments.txt
python check_env.py                 # must end with: ALL CORE CHECKS PASSED
python -m pytest tests -q           # unit + end-to-end tests on mock data
```

`requirements-experiments.txt` uses the pins of the repository's `../requirements.txt`, without the Jupyter stack,
`pywinpty` and `cupy`. Put the three datasets in `data/` (see `data/README.txt`).

## Reproducing the paper

Run from this folder. Every script is resumable: after an interruption, run the same command again. Set
`WORKERS` (default 6) to the number of parallel processes.

| Step | Command (Windows / Linux-macOS) | Output |
|---|---|---|
| 1. Superconducting configuration, classical baselines, SFS check, noise sweep | `scripts\1_main_superconducting.bat` / `bash scripts/1_main_superconducting.sh` | `results/main/*_superconducting`, `sfs_vs_exhaustive`, `noise_sweep` |
| 2. Trapped-ion configuration, QSVM reference, aggregation | `scripts\2_main_trapped_ion.bat` / `.sh` | `results/main/*_trapped_ion`, `qsvm_reference`, `_paper` |
| 3. Theory check, shot budget, hybrid test | `scripts\3_phase2.bat` / `.sh` | `results/main/theory`, `shot_budget`, `hybrid` |
| 4. IBM data bundles | `scripts\4_ibm_prepare.bat` / `.sh` | `results/main/ibm_hardware/**/bundle` |
| 5. IBM submission (IBM venv; dry run first, then `--yes`) and `python ibm/ibm_hardware.py fetch --all` | `scripts\5_ibm_submit.bat` / `.sh` | `hw_*.npz` |
| 6. Hardware analysis | `scripts\6_ibm_analyze.bat` / `.sh` | `results/main/ibm_hardware/**` |
| 7. Paper figures and PDF | `python paper/build_figure_data.py && python paper/make_figs.py`, then compile `paper/main.tex` | `paper/figs`, `paper/main.pdf` |

Measured runtimes are in `RUNTIME.md` (about 12–14 h on an 8-core laptop for steps 1–3). For a quick functional test,
add `--pilot` to any script (results go to `results/pilot/`; never use pilot numbers in the paper).

## What is in `results/main/`

Only summaries are versioned: per-run metrics of all model fits (`_paper/all_results_deduplicated.csv`, ~100
columns each), all tables (`.csv` and `.tex`), figures, theory and shot-budget results, hybrid paired tests, and,
for every IBM job, the job record and the measured bitstrings and ⟨Zᵢ⟩ (`hw_*.npz`). Trained models, per-row
scores, caches and the IBM data bundles (`bundle.npz`) are excluded for size; all of them are regenerated
deterministically by the scripts above.

## Notes on the evaluation

* Splits are stratified 60/20/20 with three seeds; every transformation is fitted on training rows only.
* Feature selection, hyper-parameters and the decision threshold use the validation set; the test set is used once.
* Test sets keep the original fraud prevalence; subsampled genuine rows are inverse-probability weighted.
* Metrics: weighted ROC-AUC and average precision with Poisson-bootstrap 95% CIs, plus threshold metrics.
