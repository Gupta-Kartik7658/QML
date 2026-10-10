# Model specifications: every trained model (inputs, architecture, hyper-parameters, outputs)

All values below come from `configs/experiment.yaml` and `qfd/classifiers.py` (scikit-learn 1.7.2,
XGBoost 3.1.1, Qiskit 1.3.1). Anything not listed uses the library default, named explicitly where it matters.
The hyper-parameter actually chosen for every run is saved in `results.csv` (`best_hp`) and in
`tasks/<task>/<representation>/hp_search.csv`.

---------------------------------------------------------------------------------------------------
## 1. Inputs: what each model receives

**Rows (identical for MLP, XGBoost and SVM within a task):**

| Dataset | Training rows (frauds) | Validation rows (frauds) | Test rows (frauds) |
|---|---|---|---|
| Credit card | 30,000 (284) | 50,094 (94) | 56,746 (95), full test partition |
| BankSim | 30,000 (4,320) | 51,440 (1,440) | 118,929 (1,440), full test partition |
| PaySim | 30,000 (≈4,928) | all frauds + 50,000 genuine | all frauds + 400,000 genuine |

* **Training rows:** all training frauds (up to 50 %) plus random genuine rows. This undersampling happens on the
  training split only.
* **Validation/test rows:** wherever genuine rows are subsampled, they get weight N_full/N_kept, so all metrics are
  estimates at the **original fraud rate**.
* **Noisy-simulation runs** use paired subsets:
  * training: 6,000 rows (≤ 3,000 frauds);
  * validation: all frauds + 4,000 genuine;
  * test: all frauds + 8,000 genuine.

**Feature vector (columns).** k = number of features chosen by SFS = number of qubits (1–7, saved per run as
`n_selected`):

| Representation | Input to classifier | Dimension | Value range |
|---|---|---|---|
| `q_ideal_z` (primary) | ⟨Z₁⟩…⟨Z_k⟩ of the feature-map state, ideal | k (≤ 7) | [−1, 1] |
| `q_ideal_zzz` | ⟨Zᵢ⟩ + all ⟨ZᵢZⱼ⟩ | k + k(k−1)/2 (28 for k = 7) | [−1, 1] |
| `q_ideal_mi_z` | ⟨Zᵢ⟩ for the top-k mutual-information features (no SFS) | k | [−1, 1] |
| `q_noisy_z` | ⟨Zᵢ⟩, architecture noise, exact | k | [−1, 1] |
| `q_noisy_shots_z` / `q_ideal_shots_z` | ⟨Zᵢ⟩ estimated from 1024 shots, noisy / ideal | k | [−1, 1] |
| `c_selected` | the same k SFS features, raw, standardised | k | z-scores |
| `c_all` | all candidate features, standardised | 30 / 12 / 24 (CC / PaySim / BankSim) | z-scores |

**Output of every classifier:**
* a continuous **fraud score** per transaction: P(fraud) for MLP and XGBoost, the signed margin
  (`decision_function`) for SVM;
* a binary label, fraud if score ≥ threshold. The threshold is chosen to maximise F1 on **validation** data.

---------------------------------------------------------------------------------------------------
## 2. Quantum feature extractor (applied before every quantum representation)

| Item | Z map | ZZ map | Pauli map |
|---|---|---|---|
| Qiskit class | `ZFeatureMap` | `ZZFeatureMap` | `PauliFeatureMap` |
| Qubits | k (one per selected feature) | k | k |
| Repetitions | 2 | 2 | 2 |
| Pauli terms | Z | Z, ZZ | Z, Y, ZZ |
| Entanglement | none | full (all pairs) | full (all pairs) |
| α (rotation scale) | 2 | 2 | 2 |
| Input angle range | [−1, 1] | [π−√π, π] = [1.369, 3.142] | [π−√π, π] |
| One repetition | H layer, then P(2xᵢ) on each qubit | H, P(2xᵢ), then for each pair CX · P(2(π−xᵢ)(π−xⱼ)) · CX | Z, Y and ZZ evolution terms with α = 2 |

* **Measurement:** Pauli-Z expectation value on each qubit, ⟨Zᵢ⟩ = Σ_b p_b (1 − 2bᵢ). Pairwise correlators come
  from the same probabilities.
* **Ideal simulator:** exact statevector, verified against Qiskit `Statevector` to 1e-14.
* **Noisy simulator:** Qiskit Aer 0.15.1 density matrix, with readout error applied exactly.
* **Shots:** 1024 multinomial samples per transaction (seeded).
* **Circuit size** (ZZ map, k = 7, 2 repetitions):

  | | Two-qubit gates | Depth |
  |---|---|---|
  | Logical circuit | 84 CX | — |
  | Compiled, superconducting | 174 CX (90 extra from SWAP routing) | 158 |
  | Compiled, trapped-ion | 84 RXX | 96 |

  Per-run values are in `circuit_<arch>.json`.

---------------------------------------------------------------------------------------------------
## 3. MLP (multilayer perceptron)

* **Pipeline:** `StandardScaler` (fitted on training rows) → `sklearn.neural_network.MLPClassifier`.
* **Architecture:** input layer of d neurons (d = dimension from §1) → hidden layer(s) → **1 output neuron with a
  sigmoid**, which gives P(fraud).

| Hyper-parameter | Value(s) |
|---|---|
| Hidden layers × neurons (searched) | **[32]**, **[64, 32]**, **[128, 64]** |
| L2 regularisation `alpha` (searched) | 1e-4, 1e-2 |
| Activation | ReLU (hidden), logistic (output) |
| Loss | binary cross-entropy + L2 penalty |
| Optimiser | Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8), constant learning rate 0.001 |
| Batch size | 256 |
| Max epochs | 500 |
| Stopping | training loss improves < 1e-5 for 20 consecutive epochs (early_stopping = False, see note) |
| Shuffle | yes, `random_state` = seed (0, 1, 2) |
| Class weighting | none (not supported by sklearn's MLP; imbalance is handled by training undersampling + threshold) |
| Grid size | 3 × 2 = **6 configurations** |

**Note on stopping:** sklearn's built-in early stopping monitors *accuracy*. On fraud data accuracy sits at the
majority-class rate, so training stopped almost immediately (we observed test AUC 0.10). We therefore stop on
training loss.

**Trainable parameters** (weights + biases):

| Input dimension d | [32] | [64, 32] | [128, 64] |
|---|---|---|---|
| 7 (quantum ⟨Z⟩, k = 7) | 289 | 2,625 | 9,345 |
| 28 (⟨Z⟩ + ⟨ZZ⟩, k = 7) | 961 | 3,969 | 12,033 |
| 30 (credit card, all features) | 1,025 | 4,097 | 12,289 |
| 12 (PaySim, all) | 449 | 2,945 | 9,985 |
| 24 (BankSim, all) | 833 | 3,713 | 11,521 |

Per-epoch loss curves and whether each run converged are saved in `iterations.json` (`loss_curve`, `converged`).

---------------------------------------------------------------------------------------------------
## 4. XGBoost (gradient-boosted decision trees)

* **Model:** `xgboost.XGBClassifier`, objective `binary:logistic`, which gives P(fraud).
* **Scaling:** none (trees don't need it).

| Hyper-parameter | Value(s) |
|---|---|
| `max_depth` (searched) | 3, 6 |
| `learning_rate` (searched) | 0.05, 0.1 |
| `scale_pos_weight` (searched) | 1.0 or genuine/fraud ratio of the training rows (≈ 104.6 credit card, 5.94 BankSim, 5.09 PaySim) |
| `n_estimators` | up to 1000 trees, with **early stopping after 50 rounds** without improvement in validation AUC-PR (weighted) |
| `subsample` | 0.8 |
| `colsample_bytree` | 1.0 |
| `tree_method` | hist (max_bin 256) |
| Other | min_child_weight 1, gamma 0, reg_lambda 1, reg_alpha 0 (defaults) |
| Threads | 1 per worker; `random_state` = seed |
| Grid size | 2 × 2 × 2 = **8 configurations** |

The boosting curve per round and the best iteration are saved in `iterations.json`.

---------------------------------------------------------------------------------------------------
## 5. SVM (classical support vector machine, RBF kernel)

* **Pipeline:** `StandardScaler` → `sklearn.svm.SVC`.
* **Kernel:** K(x, x′) = exp(−γ‖x − x′‖²).
* **Output:** signed distance to the margin. There's no probability calibration; ROC-AUC and AP only need a ranking.

| Hyper-parameter | Value(s) |
|---|---|
| `C` (searched) | 0.3, 1, 3, 10 |
| `class_weight` (searched) | None, balanced |
| `gamma` | `scale` = 1/(d · Var(X)) ≈ 1/d after standardisation |
| Solver | libsvm, tol 1e-3, shrinking on, cache 500 MB |
| Grid size | 4 × 2 = **8 configurations** |

**Why `class_weight` is searched:** an unweighted RBF-SVM on imbalanced data can rank fraud *backwards*, even on
its own training rows (we observed train AUC 0.37 against 0.86 with `balanced`). The number of support vectors
is saved in `iterations.json`.

**Not the same as the QSVM:** this classical SVM takes explicit feature vectors. The fidelity-kernel QSVM is the
separate reference in §7.

---------------------------------------------------------------------------------------------------
## 6. SFS evaluator (used only to choose features)

* **Model:** `StandardScaler` → `LogisticRegression` (L2, C = 1.0, lbfgs, `class_weight = balanced`,
  max_iter 2000).
* **Fixed:** it's **never tuned**.
* **Data:** trained on 10,000 training rows (stratified), scored on the validation rows.
* **Selection metric:** ROC-AUC (average precision recorded as secondary).

---------------------------------------------------------------------------------------------------
## 7. QSVM reference (`run_qsvm_reference.py`)

| Model | Kernel / input | Hyper-parameters |
|---|---|---|
| Fidelity-kernel QSVM | `SVC(kernel='precomputed')` on K_ij = \|⟨ψ(xᵢ)\|ψ(xⱼ)⟩\|² (exact) | C ∈ {0.1, 1, 10} × class_weight ∈ {None, balanced} |
| RBF-SVM on ⟨Z⟩ | RBF on the quantum features | same 6 configurations |
| RBF-SVM on raw features | RBF on the SFS-selected raw features | same 6 configurations |

All three use the same **2,000 training rows** (≤ 1,000 frauds) and the same validation/test sets. Kernel methods
scale as O(n²), which is why the training set is smaller here.

---------------------------------------------------------------------------------------------------
## 8. Training and selection protocol (same for every model)
1. Fit each grid configuration on the training rows.
2. Score the validation rows. Pick the best configuration by **weighted validation ROC-AUC**, with ties broken by
   average precision. There's no refit on validation data.
3. Choose the decision threshold that maximises F1 on validation data.
4. Score the **test** rows once.
5. Report:
   * ROC-AUC and average precision, each with a 95 % bootstrap CI (200 resamples);
   * precision, recall, F1, specificity, FPR = FP/(FP+TN), balanced accuracy, MCC;
   * confusion matrix (raw and weighted);
   * training, prediction and quantum-extraction time; peak memory.
6. Repeat for seeds 0, 1, 2. The seed changes the data split, the subsampling and the model's `random_state`.

**Model count:** 3 datasets × 3 feature maps × 3 seeds × up to 8 representations × 3 classifiers × 2
architecture scripts (ideal rows are shared between the two). That's ≈ 1,500 tuned models, ≈ 11,000 fits
including the grids.

**Saved per model:** `metrics.json`, `hp_search.csv`, `iterations.json`, `model.joblib` (the trained model
itself), `predictions.npz` (all validation/test scores), `curves.npz` (ROC/PR), `diagnostics.png`.
