#!/usr/bin/env python3
"""
QUANTUM KERNEL TRAINING WITH THERMAL NOISE
Config: 400 Train / 200 Test | Selectable Feature Maps | IBM Eagle Noise Profile
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix # <--- ADDED confusion_matrix
from sklearn.svm import SVC

# Qiskit Imports
from qiskit import transpile
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError
from qiskit.quantum_info import DensityMatrix, state_fidelity

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DATA_PATH = 'creditcard.csv'

# --- Feature Selection ---
FEATURE_LIST = ["V14", "V7", "V4", "V19", "V20", "V17"] 

# --- Feature Map Selection ---
# Options: 'Z', 'ZZ', 'Pauli_X_ZZ'
MAP_TYPE = 'ZZ'  
REPS = 1

# --- Dataset Size (Balanced) ---
TRAIN_SIZE = 400
TEST_SIZE = 200

# --- Hardware Noise Profile (IBM Eagle "Lower Quartile") ---
NOISE_CONFIG = {
    'T1': 60.0e3,         # 60 µs
    'T2': 30.0e3,         # 30 µs
    'GATE_TIME_1Q': 60.0, # 60 ns
    'GATE_TIME_2Q': 550.0,# 550 ns
    'P_DEP_1Q': 0.0004,   # 0.04%
    'P_DEP_2Q': 0.02,     # 2.0%
    'P_READOUT': 0.04     # 4.0%
}

print(f"\n{'='*60}")
print(f"QSVM CONFIGURATION")
print(f"{'='*60}")
print(f"Features:    {FEATURE_LIST}")
print(f"Map Type:    {MAP_TYPE}")
print(f"Train Size:  {TRAIN_SIZE}")
print(f"Test Size:   {TEST_SIZE}")
print(f"Simulation:  Density Matrix with Thermal Noise")

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"CRITICAL: {DATA_PATH} not found.")
    exit(1)

X_all = df[FEATURE_LIST].values
y_all = df['Class'].values

# Balance the Dataset (50/50 Fraud/Genuine)
fraud_idx = np.where(y_all == 1)[0]
genuine_idx = np.where(y_all == 0)[0]

# Ensure we have enough data
required_total = TRAIN_SIZE + TEST_SIZE
if len(fraud_idx) < required_total // 2:
    print(f"Warning: Not enough fraud samples for requested size. Using max available.")
    half_size = len(fraud_idx)
else:
    half_size = required_total // 2

indices = np.concatenate([
    np.random.choice(fraud_idx, half_size, replace=False),
    np.random.choice(genuine_idx, half_size, replace=False)
])
np.random.shuffle(indices)

X_bal = X_all[indices]
y_bal = y_all[indices]

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, 
    train_size=TRAIN_SIZE, 
    test_size=TEST_SIZE, 
    stratify=y_bal, 
    random_state=42
)

# Scaling (-1 to 1 is optimal for Quantum Feature Maps)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Data Prepared.")

# ==============================================================================
# 3. NOISE MODEL GENERATOR
# ==============================================================================
def create_noise_model():
    nm = NoiseModel()
    
    # 1. Single Qubit Errors (Thermal + Depolarizing)
    t1_err = thermal_relaxation_error(NOISE_CONFIG['T1'], NOISE_CONFIG['T2'], NOISE_CONFIG['GATE_TIME_1Q'])
    dep_err = depolarizing_error(NOISE_CONFIG['P_DEP_1Q'], 1)
    combined_1q = t1_err.compose(dep_err)
    nm.add_all_qubit_quantum_error(combined_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'id'])
    
    # 2. Two Qubit Errors (Thermal + Depolarizing)
    t2_q0 = thermal_relaxation_error(NOISE_CONFIG['T1'], NOISE_CONFIG['T2'], NOISE_CONFIG['GATE_TIME_2Q'])
    t2_q1 = thermal_relaxation_error(NOISE_CONFIG['T1'], NOISE_CONFIG['T2'], NOISE_CONFIG['GATE_TIME_2Q'])
    t2_err = t2_q0.tensor(t2_q1)
    dep_2q = depolarizing_error(NOISE_CONFIG['P_DEP_2Q'], 2)
    combined_2q = t2_err.compose(dep_2q)
    nm.add_all_qubit_quantum_error(combined_2q, ['cx', 'cz', 'ecr'])
    
    # 3. Readout Errors
    p_ro = NOISE_CONFIG['P_READOUT']
    ro_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    nm.add_all_qubit_readout_error(ro_err)
    
    return nm

noise_model = create_noise_model()

# ==============================================================================
# 4. KERNEL COMPUTATION
# ==============================================================================
def get_feature_map(n_features, map_type):
    if map_type == 'Z':
        return ZFeatureMap(feature_dimension=n_features, reps=REPS)
    elif map_type == 'ZZ':
        return ZZFeatureMap(feature_dimension=n_features, reps=REPS, entanglement='full')
    elif map_type == 'Pauli_X_ZZ':
        return PauliFeatureMap(feature_dimension=n_features, reps=REPS, paulis=['X', 'ZZ'])
    else:
        raise ValueError("Invalid Map Type. Choose Z, ZZ, or Pauli_X_ZZ")

def compute_kernel_matrix(X1, X2, map_type):
    n_features = X1.shape[1]
    feature_map = get_feature_map(n_features, map_type)
    
    # Use Density Matrix simulator for accurate noise simulation
    simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
    
    print(f" > Generating Quantum States for {len(X1)} + {len(X2)} samples...")
    
    def get_density_matrices(data):
        circuits = []
        for x in data:
            qc = feature_map.assign_parameters(x)
            qc.save_density_matrix()
            circuits.append(qc)
        
        # Run in batch
        transpiled = transpile(circuits, simulator)
        # shots=None forces exact calculation of the noisy density matrix
        result = simulator.run(transpiled, shots=None).result()
        
        return [DensityMatrix(result.data(i)['density_matrix']) for i in range(len(data))]

    # 1. Compute states for all data points
    states1 = get_density_matrices(X1)
    states2 = get_density_matrices(X2)
    
    # 2. Compute Fidelity (Gram Matrix)
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    print(f" > Calculating Fidelity Matrix ({n1}x{n2})...")
    start_k = time.time()
    
    for i in range(n1):
        for j in range(n2):
            K[i, j] = state_fidelity(states1[i], states2[j])
            
    print(f" > Matrix calculated in {time.time() - start_k:.2f}s")
    return K

# ==============================================================================
# 5. EXECUTION
# ==============================================================================
print(f"\nStarting Kernel Calculation...")
t_start = time.time()

# 1. Train Kernel
print("\n--- Computing Training Kernel ---")
K_train = compute_kernel_matrix(X_train, X_train, MAP_TYPE)

# 2. Test Kernel (Train vs Test)
print("\n--- Computing Testing Kernel ---")
K_test = compute_kernel_matrix(X_test, X_train, MAP_TYPE)

# 3. SVM Training
print("\n--- Training SVM ---")
svc = SVC(kernel='precomputed')
svc.fit(K_train, y_train)

# 4. Prediction
print("--- Predicting ---")
y_pred = svc.predict(K_test)

# Metrics Calculation
try:
    y_score = svc.decision_function(K_test)
    auc = roc_auc_score(y_test, y_score)
except:
    auc = 0.0
    print("Warning: Could not calculate AUC")

acc = accuracy_score(y_test, y_pred)

# --- NEW: Hit Rate and FAR Calculation ---
# cm structure: [[TN, FP], [FN, TP]]
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Hit Rate (Recall / Sensitivity): TP / (TP + FN)
hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0

# False Alarm Rate (False Positive Rate): FP / (FP + TN)
far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

total_time = time.time() - t_start

# ==============================================================================
# 6. RESULTS
# ==============================================================================
print(f"\n{'='*60}")
print(f"FINAL RESULTS ({MAP_TYPE} Map | {len(FEATURE_LIST)} Features)")
print(f"{'='*60}")
print(f"Accuracy:       {acc:.4f}")
print(f"AUC Score:      {auc:.4f}")
print(f"Hit Rate (TPR): {hit_rate:.4f}")
print(f"False Alarm (FAR): {far:.4f}")
print(f"Total Time:     {total_time:.2f} seconds")
print(f"{'='*60}")