#!/usr/bin/env python3
"""
NOISE DEGRADATION TEST: 2 to 7 FEATURES
Simulates "Bad" Real-World Hardware (IBM Eagle Profile)
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC

# Qiskit Imports
from qiskit import transpile
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError
from qiskit.quantum_info import DensityMatrix, state_fidelity

# ==============================================================================
# CONFIGURATION: REAL-WORLD "BAD" HARDWARE
# ==============================================================================
DATA_PATH = 'creditcard.csv'

# IBM Eagle/Falcon "Lower Quartile" Profile
# This represents a "bad day" on a real quantum computer
NOISE_CONFIG = {
    'T1': 60.0e3,         # 60 µs (Short relaxation)
    'T2': 30.0e3,         # 30 µs (Severe dephasing)
    'GATE_TIME_1Q': 60.0, # 60 ns
    'GATE_TIME_2Q': 550.0,# 550 ns (Slow CNOT gates = more decay)
    'P_DEP_1Q': 0.0004,   # 0.04% 1Q error
    'P_DEP_2Q': 0.02,     # 2.0% 2Q error (High error rate)
    'P_READOUT': 0.04     # 4.0% Readout error (Significant noise floor)
}

# Dataset Config (Small sizes for reasonable simulation speed)
TRAIN_SIZE = 40
TEST_SIZE = 20 
FEATURE_LIST = ['V14', 'V7', 'V4', 'V19', 'V20', 'V17', 'V10'] # First 7 features

print(f"\n{'='*60}")
print(f"REAL-WORLD NOISE SIMULATION: 2 -> 7 FEATURES")
print(f"{'='*60}")
print(f"Noise Profile (IBM Eagle - Lower Quartile):")
print(f"  T1: {NOISE_CONFIG['T1']/1000}µs | T2: {NOISE_CONFIG['T2']/1000}µs")
print(f"  2Q Gate Time: {NOISE_CONFIG['GATE_TIME_2Q']}ns")
print(f"  2Q Gate Error: {NOISE_CONFIG['P_DEP_2Q']*100}%")
print(f"  Readout Error: {NOISE_CONFIG['P_READOUT']*100}%")

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"CRITICAL: {DATA_PATH} not found.")
    exit(1)

X_all = df[FEATURE_LIST].values
y_all = df['Class'].values

# Balance and Split
fraud_idx = np.where(y_all == 1)[0]
genuine_idx = np.where(y_all == 0)[0]
sample_size = min(len(fraud_idx), 200)
indices = np.concatenate([
    np.random.choice(fraud_idx, sample_size, replace=False),
    np.random.choice(genuine_idx, sample_size, replace=False)
])
np.random.shuffle(indices)

X_bal = X_all[indices]
y_bal = y_all[indices]

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_bal, y_bal, train_size=TRAIN_SIZE, test_size=TEST_SIZE, stratify=y_bal, random_state=42
)

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)

print(f"\nData Prepared: {TRAIN_SIZE} Train / {TEST_SIZE} Test samples.")

# ==============================================================================
# 2. NOISE MODEL GENERATOR
# ==============================================================================
def create_noise_model():
    nm = NoiseModel()
    
    # 1. Thermal + Depolarizing (Single Qubit)
    t1_err = thermal_relaxation_error(NOISE_CONFIG['T1'], NOISE_CONFIG['T2'], NOISE_CONFIG['GATE_TIME_1Q'])
    dep_err = depolarizing_error(NOISE_CONFIG['P_DEP_1Q'], 1)
    combined_1q = t1_err.compose(dep_err)
    nm.add_all_qubit_quantum_error(combined_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])
    
    # 2. Thermal + Depolarizing (Two Qubit)
    t2_q0 = thermal_relaxation_error(NOISE_CONFIG['T1'], NOISE_CONFIG['T2'], NOISE_CONFIG['GATE_TIME_2Q'])
    t2_q1 = thermal_relaxation_error(NOISE_CONFIG['T1'], NOISE_CONFIG['T2'], NOISE_CONFIG['GATE_TIME_2Q'])
    t2_err = t2_q0.tensor(t2_q1)
    dep_2q = depolarizing_error(NOISE_CONFIG['P_DEP_2Q'], 2)
    combined_2q = t2_err.compose(dep_2q)
    nm.add_all_qubit_quantum_error(combined_2q, ['cx', 'cz', 'ecr'])
    
    # 3. Readout Error
    p_ro = NOISE_CONFIG['P_READOUT']
    ro_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    nm.add_all_qubit_readout_error(ro_err)
    
    return nm

noise_model = create_noise_model()

# ==============================================================================
# 3. KERNEL COMPUTATION (THE FIXED VERSION)
# ==============================================================================
def compute_kernel(X1, X2, n_features):
    # Setup Simulator
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=1, entanglement='full')
    simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
    
    # Helper to get states
    def get_states(data):
        circuits = []
        for x in data:
            qc = feature_map.assign_parameters(x) # FIXED: assign_parameters
            qc.save_density_matrix()              # FIXED: Explicit save
            circuits.append(qc)
        
        # Run batch
        transpiled = transpile(circuits, simulator)
        job = simulator.run(transpiled, shots=None) # FIXED: Exact simulation
        result = job.result()
        
        return [DensityMatrix(result.data(i)['density_matrix']) for i in range(len(data))]

    # Get states
    states1 = get_states(X1)
    states2 = get_states(X2)
    
    # Compute Gram Matrix
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            K[i, j] = state_fidelity(states1[i], states2[j])
            
    return K

# ==============================================================================
# 4. MAIN SIMULATION LOOP (2 to 7 Features)
# ==============================================================================
results_log = []

print(f"\nStarting Simulation Loop...")
print(f"{'Features':<10} | {'Time (s)':<10} | {'AUC':<10} | {'Acc':<10}")
print("-" * 50)

for n_feat in range(2, 8):
    start_time = time.time()
    
    # Subset Data
    X_tr_sub = X_train_full[:, :n_feat]
    X_te_sub = X_test_full[:, :n_feat]
    
    # Compute Kernels
    K_train = compute_kernel(X_tr_sub, X_tr_sub, n_feat)
    K_test = compute_kernel(X_te_sub, X_tr_sub, n_feat)
    
    # Train SVM
    svc = SVC(kernel='precomputed')
    svc.fit(K_train, y_train)
    
    # Metrics
    y_pred = svc.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        y_score = svc.decision_function(K_test)
        auc = roc_auc_score(y_test, y_score)
    except:
        auc = 0.0
        
    elapsed = time.time() - start_time
    
    print(f"{n_feat:<10} | {elapsed:<10.2f} | {auc:<10.4f} | {acc:<10.4f}")
    
    results_log.append({
        'Features': n_feat,
        'AUC': auc,
        'Accuracy': acc
    })

# ==============================================================================
# 5. FINAL SUMMARY
# ==============================================================================
print(f"\n{'='*60}")
print(f"DEGRADATION SUMMARY")
print(f"{'='*60}")

if len(results_log) > 1:
    first = results_log[0]
    last = results_log[-1]
    auc_drop = first['AUC'] - last['AUC']
    print(f"2 Features AUC: {first['AUC']:.4f}")
    print(f"7 Features AUC: {last['AUC']:.4f}")
    print(f"Net Degradation: {auc_drop*100:.2f}%")
    
    if last['AUC'] < 0.6:
        print("\n[OBSERVATION] The noise has successfully destroyed the model.")
    elif last['AUC'] < first['AUC']:
        print("\n[OBSERVATION] Clear degradation visible as circuit depth/width increased.")
    else:
        print("\n[OBSERVATION] Model is still surprisingly robust. (Data separation is very strong).")