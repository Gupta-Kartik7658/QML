#!/usr/bin/env python3
# Requires: qiskit==1.3.1, qiskit-aer, qiskit-machine-learning, numpy, pandas, scikit-learn

import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity
import math

# ----------------- CONFIG -----------------
DATA_PATH = "creditcard.csv"
LABEL = "Class"
FEATURE_LIST = ['V14','V12','V4','V10','V8','V13','V7']   # your 7 features
TRAIN_SIZE = 200
TEST_SIZE = 100
RANDOM_STATE = 42
# ------------------------------------------

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[LABEL])
    y = df[LABEL]
    # Select balanced subset same as before
    fraud_idx = y[y==1].index
    gen_idx = y[y==0].index
    total = TRAIN_SIZE + TEST_SIZE
    n_fraud = total // 2
    n_gen = total // 2
    rng = np.random.default_rng(RANDOM_STATE)
    fraud_sample = rng.choice(fraud_idx, n_fraud, replace=False)
    genuine_sample = rng.choice(gen_idx, n_gen, replace=False)
    sample_idx = np.concatenate([fraud_sample, genuine_sample])
    rng.shuffle(sample_idx)
    X_bal = X.loc[sample_idx].reset_index(drop=True)
    y_bal = y.loc[sample_idx].reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal,
        train_size=TRAIN_SIZE, test_size=TEST_SIZE,
        stratify=y_bal,
        random_state=RANDOM_STATE
    )
    scaler = MinMaxScaler(feature_range=(-1,1))
    Xtr_scaled = scaler.fit_transform(X_train)
    Xte_scaled = scaler.transform(X_test)
    feature_names = X.columns.tolist()
    return Xtr_scaled, Xte_scaled, y_train.values, y_test.values, feature_names

def build_featuremap(mapname, n_qubits):
    if mapname == "Z":
        fmap = ZFeatureMap(feature_dimension=n_qubits, reps=1)
    elif mapname == "ZZ":
        fmap = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement='linear', insert_barriers=True)
    elif mapname == "Pauli":
        fmap = PauliFeatureMap(feature_dimension=n_qubits, paulis=['X','Y','Z','ZZ'],
                               entanglement='linear', reps=1)
    else:
        raise ValueError(mapname)
    return fmap

def circuit_for_sample(feature_map, sample_vector):
    """
    Returns a QuantumCircuit with numeric parameters bound for the given sample.
    We try Statevector.from_instruction on the bound circuit.
    """
    # feature_map is a QuantumCircuit possibly with Parameters
    # Two possible ways to bind parameters: feature_map.assign_parameters(...) often works;
    # Statevector.from_instruction accepts a circuit with numerical parameters.
    try:
        bound = feature_map.assign_parameters(sample_vector)
    except Exception:
        try:
            # alternative: bind by params order
            params = feature_map.parameters
            bind_dict = {p: float(v) for p, v in zip(params, sample_vector)}
            bound = feature_map.bind_parameters(bind_dict)
        except Exception:
            # last resort: manually create a copy and substitute
            bound = feature_map.assign_parameters(sample_vector)  # hope this works
    return bound

def state_and_probs(circuit):
    """
    Compute statevector from circuit using Statevector.from_instruction
    Returns complex statevector and probabilities (abs^2)
    """
    sv = Statevector.from_instruction(circuit)  # exact statevector
    probs = np.abs(sv.data)**2
    return sv.data, probs

def pretty_print_state(amplitudes, probs, max_show=8):
    """Print amplitudes and probs for basis states (trim large vectors)"""
    dim = len(amplitudes)
    # generate basis labels for n qubits
    n = int(math.log2(dim))
    labels = [format(i, '0{}b'.format(n))[::-1] for i in range(dim)]  # qiskit uses little-endian visual order
    # Print header
    for i in range(min(dim, max_show)):
        amp = amplitudes[i]
        p = probs[i]
        print(f"  |{labels[i]}> : amp = {amp.real:+.6f}{amp.imag:+.6f}j  prob = {p:.6f}")
    if dim > max_show:
        print(f"  ... {dim - max_show} more amplitudes (total {dim})")

# ----------------- Main -----------------
Xtr, Xte, ytr, yte, feat_names = load_and_prepare()

# Get indices of the 7 selected features in the full feature list
selected_indices = [feat_names.index(f) for f in FEATURE_LIST]
# subset arrays for QSVM encoding (these are the x vectors used as rotations)
Xtr_sel = Xtr[:, selected_indices]
Xte_sel = Xte[:, selected_indices]

# Choose the two samples (first fraud and first non-fraud in training set)
fraud_idx = np.where(ytr == 1)[0][0]
nonfraud_idx = np.where(ytr == 0)[0][0]
print(f"Using training sample indices -> fraud: {fraud_idx}, non-fraud: {nonfraud_idx}\n")

feature_maps = ["Z", "ZZ", "Pauli"]

results = {}

for fmap_name in feature_maps:
    print(f"\n=== Feature Map: {fmap_name} ===")
    fmap = build_featuremap(fmap_name, len(selected_indices))
    # Build bound circuits for fraud and non-fraud
    fraud_vec = Xtr_sel[fraud_idx]
    nonfraud_vec = Xtr_sel[nonfraud_idx]

    # (Important) Some maps expect parameter angles like 2*x or other conventions.
    # Qiskit's standard feature maps typically use internal formulas; we pass the scaled values directly.
    circ_fraud = circuit_for_sample(fmap, fraud_vec)
    circ_nonfraud = circuit_for_sample(fmap, nonfraud_vec)

    # Obtain statevectors (exact)
    sv_fraud, probs_fraud = state_and_probs(circ_fraud)
    sv_nonfraud, probs_nonfraud = state_and_probs(circ_nonfraud)

    # Fidelity between the two states
    fid = state_fidelity(sv_fraud, sv_nonfraud)

    print("\nFraud sample (amplitudes & probs):")
    pretty_print_state(sv_fraud, probs_fraud, max_show=8)

    print("\nNon-Fraud sample (amplitudes & probs):")
    pretty_print_state(sv_nonfraud, probs_nonfraud, max_show=8)

    print(f"\nFidelity (|<psi_fraud|psi_nonfraud>|^2 approx): {fid:.6f}")

    results[fmap_name] = {
        'sv_fraud': sv_fraud,
        'probs_fraud': probs_fraud,
        'sv_nonfraud': sv_nonfraud,
        'probs_nonfraud': probs_nonfraud,
        'fidelity': fid
    }

# Optionally: print summary fidelities
print("\n\n=== SUMMARY FIDELITIES ===")
for k,v in results.items():
    print(f" {k:6s} fidelity = {v['fidelity']:.6f}")

