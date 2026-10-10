# Noise invariance of product (Z) feature maps: proposition, proof, corollaries

*Proof for the gate-level noise model below; numerical checks by `run_theory_check.py` (Qiskit Aer, both
architecture models, `results/main/theory/`) and hardware checks by `run_ibm_hardware.py`
(`results/main/ibm_hardware/`). Section "Numerical evidence" quotes those files.*

## Setting

* **Inputs and angles.** One qubit per selected feature. φᵢ = 2·x̃ᵢ, where x̃ᵢ is the angle-scaled input
  (Qiskit `ZFeatureMap` applies P(2x̃ᵢ)).
* **Circuit.** `ZFeatureMap` with reps = 2 acts on qubit i as H → P(φᵢ) → H → P(φᵢ), with no entangling
  gates.
* **Features.** zᵢ(x) = ⟨Zᵢ⟩ after the circuit.
* **Ideal value.** zᵢ^ideal(x) = cos φᵢ. (Proof: H|0⟩ has Bloch vector (1,0,0); P(φ) rotates it to
  (cos φ, sin φ, 0); H swaps x and z; the last P(φ) commutes with Z.)

**Noise class 𝒩.** A single-qubit channel Φ is in 𝒩 if its action on the Bloch vector r is

  Φ(r) = ( λ⊥ r_x , λ⊥ r_y , λ∥ r_z + κ ),

for real λ⊥, λ∥, κ. 𝒩 is closed under composition and contains every channel in both of our noise models:

| Channel | λ⊥ | λ∥ | κ |
|---|---|---|---|
| depolarising, Qiskit parameter p (ρ ↦ (1−p)ρ + p·I/2) | 1−p | 1−p | 0 |
| amplitude damping γ | √(1−γ) | 1−γ | γ |
| phase damping λ | √(1−λ) | 1 | 0 |
| thermal relaxation (T1, T2, gate time t; ground-state equilibrium) | e^{−t/T2} | e^{−t/T1} | 1−e^{−t/T1} |

**Readout.** Independent per-qubit readout with flip probabilities p₀₁ = P(read 1 | 0) and
p₁₀ = P(read 0 | 1) maps the measured ⟨Z⟩ to ρ_ro·z + o, where ρ_ro = 1 − p₀₁ − p₁₀ and o = p₁₀ − p₀₁. For
symmetric readout, o = 0.

**Assumption A (gate-level, single-qubit noise).** Each gate g on qubit i is implemented as Φ_g ∘ U_g,
with Φ_g ∈ 𝒩. Readout is as above.

*Compilation.*
* **Superconducting configuration.** H compiles to Rz(π/2)·SX·Rz(π/2), Rz is error-free and virtual, and
  noise follows SX. Every Φ ∈ 𝒩 commutes with every Rz, because it scales the x–y plane isotropically. So
  the noisy Hadamard is Rz∘Φ∘SX∘Rz = Φ∘H, and Assumption A holds exactly.
* **Trapped-ion configuration.** H compiles into {rx, ry} with noise after each gate. Assumption A is not
  proved for this compilation; it is verified numerically below.

## Proposition (affine noise law)

Under Assumption A, for the reps = 2 `ZFeatureMap`, every qubit i and every input x:

  zᵢ^noisy(x) = αᵢ + βᵢ · zᵢ^ideal(x),

with

  βᵢ = ρ_ro · λ∥^{(P₂)} · λ∥^{(H₂)} · λ⊥^{(P₁)} · λ⊥^{(H₁)},
  αᵢ = ρ_ro · ( λ∥^{(P₂)} κ^{(H₂)} + κ^{(P₂)} ) + o.

Here H₁, P₁, H₂, P₂ label the four gates on qubit i in order. Neither αᵢ nor βᵢ depends on x. Moreover,
zᵢ^noisy depends only on xᵢ.

**Proof.** Follow the Bloch vector of qubit i.

| Step | Bloch vector |
|---|---|
| start | (0, 0, 1) |
| after H₁ | (1, 0, 0) |
| after Φ^{(H₁)} | (λ⊥¹, 0, κ¹) |
| after P₁ = Rz(φ) | (λ⊥¹ cos φ, λ⊥¹ sin φ, κ¹) |
| after Φ^{(P₁)} | z-component irrelevant below; x-component λ⊥^{P₁} λ⊥¹ cos φ |
| after H₂, which maps (x, y, z) ↦ (z, −y, x) | z-component λ⊥^{P₁} λ⊥¹ cos φ |
| after Φ^{(H₂)} | z-component λ∥^{H₂} λ⊥^{P₁} λ⊥¹ cos φ + κ^{H₂} |
| after P₂ | z-component unchanged (Rz preserves it) |
| after Φ^{(P₂)} | z-component multiplied by λ∥^{P₂}, plus κ^{P₂} |

Readout then applies z ↦ ρ_ro z + o. Substituting cos φ = zᵢ^ideal gives the stated α and β.

There are no two-qubit gates, and every channel and the readout act on one qubit. So the reduced state of
qubit i, and hence zᵢ, depends only on φᵢ. ∎

**Closed form for our superconducting configuration.**
* Rz/P gates are error-free: λ = 1, κ = 0.
* H₁ and H₂ each carry Φ = depolarising(p₁q) ∘ thermal(T1, T2, t₁q).
* Readout is symmetric with probability p_ro.

So

  β = (1 − 2p_ro) · e^{−t₁q/T1} · e^{−t₁q/T2} · (1 − p₁q)²,

which gives **β = 0.91651** for (p_ro, t₁q, T1, T2, p₁q) = (0.04, 60 ns, 60 µs, 30 µs, 4·10⁻⁴).

## Corollary 1: exact invariance of the learning pipeline (exact expectations)

If every βᵢ > 0, then for any training set:

1. **Standardisation removes the noise.** Every pipeline that standardises features on the training rows
   sees *identical* inputs:
   (βz + α − (βμ + α)) / (βσ) = (z − μ)/σ.
   This covers the logistic-regression SFS evaluator, the MLP and the RBF-SVM. So the SFS path, the chosen
   hyper-parameters, the validation threshold and every test metric are identical under noise and without
   it.
2. **Trees are unaffected.** Gradient-boosted trees depend on features only through their per-feature
   order (split search and histogram bins). A strictly increasing per-feature map leaves them unchanged.

**Consequence.** For product feature maps, hardware gate noise and readout noise of class 𝒩 have *no*
effect on fraud-detection performance in the exact-expectation limit.
* Every degradation seen for the Z map comes from finite sampling.
* For entangling maps (ZZ, Pauli), the noise is not removable this way.

## Corollary 2: gate noise is a shot-budget multiplier

With S shots, the estimate is ẑ = α + βz + ε, with E[ε] = 0 and Var ε = (1 − (α + βz)²)/S. The affine
inverse (ẑ − α)/β is unbiased for z, with variance (1 − (α + βz)²)/(β²S). An ideal device with S′ shots has
variance (1 − z²)/S′. Equating the two, the noisy S-shot feature carries the information of

  S_eff(z) = β² S · (1 − z²)/(1 − (α + βz)²) ≤ β² S  (for α ≈ 0, |β| ≤ 1)

ideal shots. In words: **gate and readout noise multiply the shot budget needed for a given accuracy by about
1/β²**.

| Configuration | β | Shot multiplier ≈ 1/β² |
|---|---|---|
| Superconducting, baseline | 0.9165 | 1.19 |
| Trapped-ion, baseline (fitted) | 0.9912 | 1.02 |
| Superconducting, all noise parameters ×10 | ≈ 0.147 | ≈ 46 |

`run_shot_budget.py` tests this directly. Plotted against β²S, the noisy and ideal AUC-versus-shots curves
should overlap.

## Remark: why entangling maps behave differently

With two-qubit gates, a channel on one qubit propagates into the reduced states of the others, so
zᵢ^noisy becomes a function of several inputs. It is then no longer an invertible per-feature transform of
zᵢ^ideal. Numerically (superconducting, baseline noise), the best per-feature affine fit for ZZ leaves
residuals of order 10⁻²; at ×10 noise the features collapse to constants (β → 0). This matches the noise
sweeps: ZZ/Pauli degrade, and collapse with shots, while Z is flat.

## Numerical evidence

| Check | Result | File |
|---|---|---|
| Affine fit of noisy on ideal ⟨Zᵢ⟩, Z map, both models, noise ×1/×3/×10 (synthetic inputs) | max residual 2.2×10⁻¹⁵ | `results/main/theory/affine_synthetic.csv` |
| Fitted β vs closed form (superconducting) | difference 2×10⁻¹⁵ (β = 0.91651) | same |
| Same fit on the real paired rows of all 18 Z-map tasks | max residual 6×10⁻¹⁴ | `affine_real.csv` |
| Retraining on noisy instead of ideal Z features: mean \|ΔAUC\| logistic / MLP / SVM | 1.0×10⁻⁷ / 1.1×10⁻⁷ / 4.8×10⁻⁷ (round-off control 1.2×10⁻⁷ / 1.4×10⁻⁷ / 3.2×10⁻⁶) | `pipeline_invariance.csv` |
| XGBoost, same test | 1.9×10⁻³ (32-bit histogram binning; control 0) | same |
| Entangling maps, logistic \|ΔAUC\| | ZZ trapped-ion 4×10⁻⁶, ZZ superconducting 2×10⁻⁴, Pauli superconducting 3×10⁻³ | same |
| IBM Heron, Z map, per-physical-qubit affine fit (360 qubits, 2 devices, 3 datasets) | median R² 0.997; residual s.d. 1.08–1.11× shot noise | `results/main/ibm_hardware/**/hardware_affine_per_qubit*.csv` |
| IBM Heron, ZZ map, k = 3 / 5 / 7 | median β 0.86 / 0.73 / 0.54 → β ≈ β₀·ηᵍ, η ≈ 0.990 per two-qubit gate per qubit | same, and `paper/data/beta_vs_gates.csv` |

## Related work

* Rescaling of kernels and expectation values under global depolarising noise:
  * Wang, Du, Luo, Tao, *Towards understanding the power of quantum kernels in the NISQ era*, Quantum 5,
    531 (2021).
  * Heyraud, Li, Denis, Le Boité, Ciuti, *Noisy quantum kernel machines*, Phys. Rev. A 106, 052421 (2022).
* Concentration of quantum features and kernels: Thanasilp et al., *Exponential concentration in quantum
  kernel methods*, Nat. Commun. (2024).
* **Our difference from these works:**
  * an *exact* per-feature law for **non-unital** noise (T1/amplitude damping) plus readout, not only
    depolarising noise;
  * an *exact* invariance of the whole pipeline, including feature selection;
  * the shot-multiplier interpretation, tested on real fraud data.
