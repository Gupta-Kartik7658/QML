# Quantum Feature Extraction for Financial Fraud Detection — Methodology
*(design document of the QCNC 2027 revision; results are in `results/main/` and `paper/`)*

## 1. One-line summary
We encode transactions into quantum feature-map circuits, read out a small set of **quantum expectation
values ⟨Zᵢ⟩ as new features**, and feed them to **classical classifiers (MLP, XGBoost, SVM)**. We then compare
them fairly against classical features on **one real and two synthetic** fraud datasets, under **ideal and
hardware-noise simulation** of two quantum architectures.

**Research question:** how do feature-map design, quantum architecture (noise + connectivity) and the downstream
classifier affect fraud detection, compared with classical features of the same inputs?
The aim is a **controlled, honest comparison**, not a claim of quantum advantage.

## 2. Changes relative to the earlier version (main branch)
| Change | What we did |
|---|---|
| 1 real + 2 synthetic datasets | European credit card (real) + **PaySim** + **BankSim** (both synthetic) |
| Quantum features → MLP / XGBoost / SVM, instead of comparing QSVMs | ⟨Zᵢ⟩ feature extraction → 3 classical classifiers. The fidelity-kernel QSVM is kept as a *reference* experiment |
| QGFS too expensive | **Sequential Forward Selection (SFS)**: ≤ 189 evaluations (30 features, up to 7 qubits), versus ~2 million for exhaustive search |

## 3. Pipeline
```
 raw data ──► clean + split (train 60 / val 20 / test 20, stratified, 3 seeds)
          ──► preprocessing FITTED ON TRAIN ONLY (log, impute, scale to [0,1])
          ──► SFS on validation data: pick ≤7 features (= qubits)
          ──► quantum feature map U_φ(x) (Z / ZZ / Pauli, 2 repetitions)
          ──► measure z(x) = [⟨Z₁⟩ … ⟨Z_k⟩]          (ideal | superconducting noisy | trapped-ion noisy)
          ──► MLP / XGBoost / SVM (tuned on validation)
          ──► threshold chosen on validation ──► ONE evaluation on the untouched test set
```

## 4. Datasets (schema and counts verified by our loaders)
| Dataset | Type | Rows | Frauds | Fraud rate | Candidate features |
|---|---|---|---|---|---|
| European credit card (Kaggle mlg-ulb) | real | 284,807 → 283,726 after removing 1,081 exact duplicates | 492 → 473 | 0.167 % | 30 (V1–V28 PCA, Amount, Time) |
| PaySim (Kaggle ealaxi/paysim1) | synthetic, mobile money | 6,362,620 | 8,213 | 0.13 % | 12 |
| BankSim (Kaggle ealaxi/banksim1) | synthetic, card payments | 594,643 | 7,200 | 1.2 % | 24 |

**Leakage controls:**
* **Removed columns:** account and customer IDs, PaySim's `isFlaggedFraud` (the simulator's own rule output),
  and constant columns.
* **BankSim merchant:** target encoding fitted on training rows only, out-of-fold.
* **Test sets:** keep the **original fraud rate**. There is no balanced 1:1 test set anywhere.
* **Large partitions:** training rows are undersampled (training only). Large validation/test partitions use
  inverse-probability weights, so metrics are unbiased at the true prevalence.

## 5. Quantum feature definition
For the selected inputs x ∈ ℝᵏ (one qubit per feature):

  |ψ(x)⟩ = U_φ(x)|0…0⟩,  zᵢ(x) = ⟨ψ(x)|Zᵢ|ψ(x)⟩,  z(x) ∈ [−1, 1]ᵏ

* **Ablation:** we also test pairwise correlators ⟨ZᵢZⱼ⟩ (k + k(k−1)/2 features).
* **Simulator:** an exact, vectorised statevector simulator built from Qiskit's own circuits (Qiskit 1.3.1). It
  agrees with Qiskit's `Statevector` to 1e-14 and is about 1000× faster. That speed is what makes SFS feasible
  on a single workstation CPU.
* **Feature maps:** Qiskit `ZFeatureMap`, `ZZFeatureMap` (full entanglement) and `PauliFeatureMap`
  ([Z, Y, ZZ]), all with 2 repetitions.

## 6. Feature selection: SFS
1. Start empty. At each stage, try adding each remaining feature, and score the quantum features of the
   candidate subset with a **fixed** evaluator (logistic regression) on **validation** data (ROC-AUC primary,
   average precision secondary).
2. Add the best feature. Run the full path to 7 features and keep the best subset seen.
3. Cost: at most 30 + 29 + … + 24 = 189 evaluations per run. Every candidate score is logged.
4. Extra checks:
   * SFS vs **exhaustive** search on a 10-feature pool (how close the heuristic gets);
   * SFS vs a **predefined** top-k mutual-information subset.

## 7. Architecture configurations (simulation; IBM Heron hardware check in `ibm/`)
| | Superconducting-inspired | Trapped-ion-inspired |
|---|---|---|
| Connectivity | 7-qubit IBM Falcon-style layout, SWAPs inserted by the compiler | all-to-all |
| Native gates | rz, sx, x, cx | rz, rx, ry, rxx (Mølmer–Sørensen type) |
| Noise | T1 60 µs, T2 30 µs, gate times 60/550 ns, depolarising 4e-4 / 2e-2, readout 4 % | depolarising 6e-4 / 5e-3, amplitude 5e-5 and phase 3e-4 damping, readout 0.3 % |

* **Simulator:** Qiskit Aer density-matrix simulation, with readout error applied exactly.
* **2×2 comparison on identical rows:** ideal vs noisy × exact expectation values vs **1024 shots**. This
  separates hardware noise from measurement (shot) noise.
* **Noise sweeps:** one parameter at a time, per architecture.
* **Ideal features are identical on both architectures** (same mathematics). Differences come only from
  compilation plus noise, and we state this in the paper.

## 8. Classifiers and evaluation
* **Classifiers:**
  * MLP: 6 configurations.
  * XGBoost: 8 configurations.
  * RBF-SVM: 8 configurations.
  * All three get the same inputs, the same rows and similar tuning budgets.
* **Classical baselines:** all features, and the same SFS-selected features without quantum encoding.
* **Metrics:**
  * ROC-AUC and average precision, with 95 % bootstrap confidence intervals;
  * at the validation-chosen threshold: precision, recall, F1, specificity, FPR = FP/(FP+TN), MCC, balanced
    accuracy, confusion matrices;
  * runtime and memory.
* **Reporting:** mean ± s.d. over 3 seeds. The "best model" is chosen by **validation** AUC, never test.

**Experiment grid:** 3 datasets × 3 feature maps × 2 architectures × 3 classifiers × 3 seeds, plus baselines and
ablations ≈ 1,300 evaluated model configurations (≈ 10,000 model fits including hyper-parameter tuning).

## 9. Problems found in the earlier code and design findings
**Earlier code on the main branch (12 issues in `AUDIT.md`):**
* Features were selected on the **test set**, so the earlier AUCs are optimistic.
* The "1024 shots" claim was false; computation was exact.
* "Superconducting vs trapped-ion" was really linear vs full entanglement.
* The readout-error noise had **no effect**.

**Findings that shaped the design (each is publishable):**
1. With 1 repetition, **⟨Zᵢ⟩ = 0 for every input** in all three maps. 2 repetitions are needed.
2. The earlier **Pauli[X,ZZ] map gives exactly the same measured features as ZZFeatureMap**, so we use [Z,Y,ZZ].
3. **Phase wrap-around:** inputs in [−1, 1] make the ZZ angles wrap about 4× and scramble the features. Our
   "no-wrap" input range keeps every phase within 2π, and it was best on validation data.
4. **Early stopping in SFS** stops at a temporary dip: 0.948 → 0.946 at 2 qubits, but 0.979 at 7. So we run the
   full path.
5. **Unentangled (Z) map:** gate noise only stretches and shifts each feature, which standardised and tree
   classifiers ignore. **Only shot noise hurts** it. Proposition and proof: `theory/NOISE_INVARIANCE.md`.

## 10. Results (details in `paper/main.tex` and `results/main/`)
* **Main benchmark (Table 3 of the paper):** quantum ⟨Zᵢ⟩ features never beat all classical features (largest gain
  +0.002 ROC-AUC). Against the same SFS-selected inputs used classically, the mean ROC-AUC difference is −0.019
  (credit card), −0.005 (PaySim), −0.003 (BankSim).
* **Noise law (`results/main/theory`):** for the Z map, noisy = α + β·ideal exactly (residual ≤ 2×10⁻¹⁵ synthetic,
  6×10⁻¹⁴ on real rows); standardised classifiers are invariant to round-off level.
* **IBM Heron hardware (`results/main/ibm_hardware`):** seven jobs on `ibm_marrakesh` and `ibm_fez` (247 s QPU).
  Z map: per-qubit affine law with median R² 0.997 and residual ≈ 1.1× shot noise on 360 qubits; AUC equal to
  ideal features at the same shot count (±0.012). ZZ map: retained signal β = 0.86 / 0.73 / 0.54 for k = 3 / 5 / 7,
  i.e. β ≈ β₀·ηᵍ with η ≈ 0.990 per two-qubit gate per qubit.
* **Shot budget (`results/main/shot_budget`):** Z map converges within 16–64 shots on credit card; entangling maps on
  the superconducting model need 2,048–4,096 shots or more.
* **Hybrid test (`results/main/hybrid`):** adding quantum features to all classical features changes ROC-AUC by
  ≤ +0.008; the only significant gains (RBF-SVM, BankSim) are matched by random Fourier features of the same inputs.
* **SFS vs exhaustive:** exact optimum in 75 % of 72 comparisons, mean gap 0.001 AUC, 34 vs 385 evaluations.
