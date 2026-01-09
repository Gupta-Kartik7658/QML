#!/usr/bin/env python3
"""
COMPARATIVE QUANTUM KERNEL ANALYSIS (Sequential Version)
Models: RBF vs Z vs ZZ vs Pauli
Includes custom compute_quantum_kernel_matrix() function
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

# Qiskit Imports
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_aer import AerSimulator


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class config:
    FEATURE_MAP_REPS = 1
    ENTANGLEMENT = "linear"      # changed dynamically per model
    PAULI_STRINGS = ["X", "ZZ"]
    FEATURE_MAP_TYPE = None      # updated inside loop
    ALPHA = 1.0
    USE_GPU = False


DATA_PATH = "creditcard.csv"
OUTPUT_FOLDER = "circuit_diagrams"

TRAIN_SIZE = 400
TEST_SIZE = 200

# Feature Lists from your table
FEATS_Z          = ['V4', 'V11', 'V19', 'V13', 'V20']
FEATS_ZZ_SC      = ['V14', 'V7', 'V4', 'V19', 'V20', 'V17']
FEATS_ZZ_ION     = ['V14', 'V7', 'V2', 'V19', 'V8', 'V4']
FEATS_PAULI_SC   = ['V14', 'V12', 'V4', 'V17', 'V8']
FEATS_PAULI_ION  = ['V17', 'V21', 'V2', 'V1', 'V14']


if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


# ==============================================================================
# 2. LOAD + PREPARE DATA
# ==============================================================================
print("Loading data...")

df = pd.read_csv(DATA_PATH)
y_all = df["Class"].values

fraud_idx = np.where(y_all == 1)[0]
genuine_idx = np.where(y_all == 0)[0]

np.random.seed(42)
indices = np.concatenate([
    np.random.choice(fraud_idx, (TRAIN_SIZE + TEST_SIZE)//2, replace=False),
    np.random.choice(genuine_idx, (TRAIN_SIZE + TEST_SIZE)//2, replace=False)
])
np.random.shuffle(indices)

# Split indices into fixed train/test
train_idx_split, test_idx_split = train_test_split(
    indices,
    train_size=TRAIN_SIZE, test_size=TEST_SIZE,
    stratify=y_all[indices], random_state=42
)

print(f"Data Indices Prepared: {len(train_idx_split)} Train, {len(test_idx_split)} Test")


# ==============================================================================
# 3. CUSTOM QUANTUM KERNEL FUNCTION
# ==============================================================================
def compute_quantum_kernel_matrix(X_train, X_test, feature_indices, feature_names, verbose=False):
    """Compute quantum kernel matrix with PauliFeatureMap support."""
    
    n_features = len(feature_indices)
    
    X_train_subset = X_train[:, feature_indices]
    X_test_subset  = X_test[:, feature_indices]

    # Select feature map
    if config.FEATURE_MAP_TYPE == 'ZZ':
        feature_map = ZZFeatureMap(
            feature_dimension=n_features,
            reps=config.FEATURE_MAP_REPS,
            entanglement=config.ENTANGLEMENT
        )
    elif config.FEATURE_MAP_TYPE == 'Z':
        feature_map = ZFeatureMap(
            feature_dimension=n_features,
            reps=config.FEATURE_MAP_REPS
        )
    elif config.FEATURE_MAP_TYPE == 'Pauli':
        feature_map = PauliFeatureMap(
            feature_dimension=n_features,
            reps=config.FEATURE_MAP_REPS,
            paulis=config.PAULI_STRINGS,
            entanglement=config.ENTANGLEMENT,
            alpha=config.ALPHA
        )
    else:
        raise ValueError("Invalid feature map type.")

    # Backend selection
    if config.USE_GPU:
        sim = AerSimulator(method='statevector', device='GPU')
    else:
        sim = AerSimulator(method='statevector', device='CPU')

    kernel = FidelityQuantumKernel(feature_map=feature_map)

    start = time.time()
    try:
        K_train = kernel.evaluate(X_train_subset, X_train_subset)
        K_test = kernel.evaluate(X_test_subset,  X_train_subset)
    except Exception as e:
        print(f"[Kernel Error] {e}")
        return None, None, 0.0
    duration = time.time() - start

    return K_train, K_test, duration


# ==============================================================================
# 4. WORKER (SEQUENTIAL)
# ==============================================================================
def run_experiment(cfg):
    name     = cfg["name"]
    f_list   = cfg["features"]
    model_t  = cfg["type"]
    ent      = cfg.get("ent", None)

    print(f"\n===== Running {name} | {len(f_list)} features =====")

    # Update config
    config.FEATURE_MAP_TYPE = model_t
    if ent:   # quantum models only
        config.ENTANGLEMENT = ent

    # Extract data
    X_train = df.loc[train_idx_split, f_list].values
    y_train = df.loc[train_idx_split, "Class"].values
    X_test  = df.loc[test_idx_split,  f_list].values
    y_test  = df.loc[test_idx_split,  "Class"].values

    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Classical baseline
    if model_t == "RBF":
        svc = SVC(kernel="rbf", gamma="scale", C=1.0)
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        return name, confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred), None, f_list

    # Quantum (Z / ZZ / Pauli)
    feature_indices = list(range(len(f_list)))

    K_train, K_test, t_kernel = compute_quantum_kernel_matrix(
        X_train, X_test, feature_indices, f_list
    )

    if K_train is None:
        print("Skipping model due to kernel failure.")
        return name, None, 0.0, None, f_list

    svc = SVC(kernel="precomputed")
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)

    # Build feature map again for saving circuit image
    if model_t == "Z":
        fm = ZFeatureMap(len(f_list), reps=1)
    elif model_t == "ZZ":
        fm = ZZFeatureMap(len(f_list), reps=1, entanglement=ent)
    else:
        fm = PauliFeatureMap(len(f_list), reps=1, paulis=["X", "ZZ"], entanglement=ent)

    return name, confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred), fm, f_list


# ==============================================================================
# 5. MAIN LOOP (SEQUENTIAL)
# ==============================================================================
configs = [
    {'name': 'RBF-SVM (Classical)', 'type': 'RBF',   'features': FEATS_ZZ_SC},

    {'name': 'Z FeatureMap',        'type': 'Z',     'features': FEATS_Z},

    {'name': 'ZZ Map (Linear)',     'type': 'ZZ',    'features': FEATS_ZZ_SC,  'ent': 'linear'},
    {'name': 'ZZ Map (Full)',       'type': 'ZZ',    'features': FEATS_ZZ_ION, 'ent': 'full'},

    {'name': 'Pauli Map (Linear)',  'type': 'Pauli', 'features': FEATS_PAULI_SC, 'ent': 'linear'},
    {'name': 'Pauli Map (Full)',    'type': 'Pauli', 'features': FEATS_PAULI_ION,'ent': 'full'}
]

print(f"\nRunning {len(configs)} models sequentially...\n")

results = []
start_total = time.time()
for cfg in configs:
    results.append(run_experiment(cfg))
print(f"\nTotal Time: {time.time() - start_total:.2f} seconds\n")


# ==============================================================================
# 6. VISUALIZATION
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

print("Generating matrices & saving circuits...\n")

for i, (name, cm, acc, circuit, feats_used) in enumerate(results):

    if cm is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    ax=axes[i], cbar=False,
                    xticklabels=['Genuine', 'Fraud'],
                    yticklabels=['Genuine', 'Fraud'])
        axes[i].set_title(f"{name}\nAcc: {acc:.2%} | {len(feats_used)} Feats")
    else:
        axes[i].set_title(f"{name}\n(Kernel Failed)")

    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

    # Save circuit if available
    if circuit is not None:
        try:
            filename = f"{OUTPUT_FOLDER}/{name.replace(' ', '_')}.png"
            circuit.decompose().draw("mpl", filename=filename)
            print(f"Saved circuit: {filename}")
        except Exception as e:
            print(f"Could not save circuit for {name}: {e}")

plt.savefig("Model_Comparison_Confusion_Matrices.png", dpi=300)
plt.show()
