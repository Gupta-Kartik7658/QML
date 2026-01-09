#!/usr/bin/env python3
"""
QUANTUM KERNEL TRAINING WITH TRAPPED ION NOISE
Multi-Map Evaluation | Realistic Ion Trap Noise Profile
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC

# Qiskit Imports
from qiskit import transpile
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error, depolarizing_error, ReadoutError
from qiskit.quantum_info import DensityMatrix, state_fidelity

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DATA_PATH = 'creditcard.csv'

# --- Feature Configurations for Each Map ---
FEATURE_CONFIGS = {
    'Z': ["V4", "V11", "V19", "V13", "V20"],
    'ZZ': ["V14", "V7", "V2", "V19", "V8", "V4"],
    'Pauli_X_ZZ': ["V17", "V21", "V2", "V1", "V14"]
}

REPS = 1

# --- Dataset Size (Balanced) ---
TRAIN_SIZE = 400
TEST_SIZE = 200

# --- Trapped Ion Noise Profile (Based on Real IonQ/Oxford Ionics Hardware) ---
# References:
# - IonQ Aria: 1Q: 99.94%, 2Q: 99.5% (production system)
# - IonQ Forte: 1Q: 99.96%, 2Q: 99.6% (30-qubit system, 2024)
# - Oxford Ionics: 1Q: 99.9992%, 2Q: 99.97% (lab record, 2024)
# - IonQ Harmony: 1Q: 99%, 2Q: 98% (older generation)
# 
# We use IonQ Aria specs as realistic "commercial" system:

NOISE_CONFIG = {
    # Single-qubit gate fidelity: 99.94% -> error rate 0.06%
    # This includes all errors: amplitude damping, dephasing, control errors
    'SINGLE_QUBIT_ERROR': 0.0006,
    
    # Two-qubit gate (Mølmer-Sørensen) fidelity: 99.5% -> error rate 0.5%
    # Main error sources: motional mode imperfections, crosstalk, laser intensity fluctuations
    'TWO_QUBIT_ERROR': 0.005,
    
    # Amplitude damping (spontaneous emission from metastable states)
    # T1 ~ 1-50 seconds for trapped ions (excellent!)
    # Contribution to gate error is minimal
    'AMPLITUDE_DAMPING_PROB': 0.00005,  # ~0.005% - very low
    
    # Phase damping (dephasing from magnetic field fluctuations, laser phase noise)
    # T2 ~ 0.2-2 seconds typically
    # Main contribution to single-qubit error
    'PHASE_DAMPING_PROB': 0.0003,  # ~0.03%
    
    # Readout error (state discrimination error)
    # Modern trapped ion systems: 99.5-99.9% fidelity
    # SPAM (State Preparation and Measurement) error
    'READOUT_ERROR': 0.003  # 0.3% - based on IonQ Aria specs
}

print(f"\n{'='*70}")
print(f"TRAPPED ION QSVM - MULTI-MAP EVALUATION")
print(f"{'='*70}")
print(f"Train Size:  {TRAIN_SIZE}")
print(f"Test Size:   {TEST_SIZE}")
print(f"Simulation:  Density Matrix with Trapped Ion Noise")
print(f"\nNoise Parameters (IonQ Aria-like Commercial System):")
print(f"  Single-Qubit Gate Error:  {NOISE_CONFIG['SINGLE_QUBIT_ERROR']*100:.3f}% (Fidelity: 99.94%)")
print(f"  Two-Qubit Gate Error:     {NOISE_CONFIG['TWO_QUBIT_ERROR']*100:.2f}% (Fidelity: 99.5%)")
print(f"  Amplitude Damping:        {NOISE_CONFIG['AMPLITUDE_DAMPING_PROB']*100:.4f}%")
print(f"  Phase Damping:            {NOISE_CONFIG['PHASE_DAMPING_PROB']*100:.3f}%")
print(f"  Readout Error:            {NOISE_CONFIG['READOUT_ERROR']*100:.2f}%")

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"\nCRITICAL: {DATA_PATH} not found.")
    exit(1)

# Get all unique features needed
all_features = set()
for features in FEATURE_CONFIGS.values():
    all_features.update(features)
all_features = list(all_features)

X_all = df[all_features + ['Class']].copy()
y_all = X_all['Class'].values
X_all = X_all.drop('Class', axis=1)

# Balance the Dataset (50/50 Fraud/Genuine)
fraud_idx = np.where(y_all == 1)[0]
genuine_idx = np.where(y_all == 0)[0]

required_total = TRAIN_SIZE + TEST_SIZE
half_size = required_total // 2

if len(fraud_idx) < half_size or len(genuine_idx) < half_size:
    print(f"\nERROR: Not enough samples in dataset.")
    print(f"Required: {half_size} fraud + {half_size} genuine = {required_total} total")
    print(f"Available: {len(fraud_idx)} fraud, {len(genuine_idx)} genuine")
    exit(1)

indices = np.concatenate([
    np.random.choice(fraud_idx, half_size, replace=False),
    np.random.choice(genuine_idx, half_size, replace=False)
])
np.random.shuffle(indices)

X_bal = X_all.iloc[indices]
y_bal = y_all[indices]

# Stratified Split
actual_train_size = min(TRAIN_SIZE, len(y_bal) - TEST_SIZE)
actual_test_size = min(TEST_SIZE, len(y_bal) - actual_train_size)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_bal, y_bal, 
    train_size=actual_train_size,
    test_size=actual_test_size,
    stratify=y_bal, 
    random_state=42
)

print(f"\nData Prepared: {len(y_train)} train, {len(y_test)} test samples")
print(f"Train class distribution: Fraud={sum(y_train==1)}, Genuine={sum(y_train==0)}")
print(f"Test class distribution:  Fraud={sum(y_test==1)}, Genuine={sum(y_test==0)}")

# ==============================================================================
# 3. TRAPPED ION NOISE MODEL
# ==============================================================================
def create_trapped_ion_noise_model():
    """
    Creates a noise model representative of trapped ion quantum computers.
    Key differences from superconducting qubits:
    - Much longer coherence times (T1 ~ seconds vs microseconds)
    - Primary errors: gate infidelity, phase noise, crosstalk
    - High-fidelity single-qubit gates
    - Lower fidelity two-qubit gates (Mølmer-Sørensen)
    """
    nm = NoiseModel()
    
    # 1. Single-Qubit Gate Errors
    # Combination of:
    # - Amplitude damping (spontaneous emission - very rare)
    # - Phase damping (magnetic field fluctuations)
    # - Small depolarizing component (other errors)
    
    amp_damp = amplitude_damping_error(NOISE_CONFIG['AMPLITUDE_DAMPING_PROB'])
    phase_damp = phase_damping_error(NOISE_CONFIG['PHASE_DAMPING_PROB'])
    depol_1q = depolarizing_error(NOISE_CONFIG['SINGLE_QUBIT_ERROR'] * 0.3, 1)
    
    # Compose errors
    single_qubit_error = amp_damp.compose(phase_damp).compose(depol_1q)
    
    # Apply to all single-qubit gates
    nm.add_all_qubit_quantum_error(single_qubit_error, 
                                   ['u1', 'u2', 'u3', 'rz', 'ry', 'rx', 'sx', 'x', 'id', 'h'])
    
    # 2. Two-Qubit Gate Errors (Mølmer-Sørensen gate)
    # These are typically less accurate than single-qubit gates
    # Main error sources: crosstalk, imperfect laser pulses
    
    # Apply amplitude and phase damping to both qubits
    amp_damp_2q = amp_damp.tensor(amp_damp)
    phase_damp_2q = phase_damp.tensor(phase_damp)
    depol_2q = depolarizing_error(NOISE_CONFIG['TWO_QUBIT_ERROR'], 2)
    
    two_qubit_error = amp_damp_2q.compose(phase_damp_2q).compose(depol_2q)
    
    # Apply to two-qubit gates
    nm.add_all_qubit_quantum_error(two_qubit_error, 
                                   ['cx', 'cz', 'cy', 'swap', 'iswap'])
    
    # 3. Readout Errors (state discrimination)
    # Trapped ions have excellent readout but not perfect
    p_ro = NOISE_CONFIG['READOUT_ERROR']
    # Asymmetric readout errors (|1⟩ -> |0⟩ more likely than |0⟩ -> |1⟩)
    ro_err = ReadoutError([[1 - p_ro, p_ro], 
                           [p_ro * 1.5, 1 - p_ro * 1.5]])
    nm.add_all_qubit_readout_error(ro_err)
    
    return nm

noise_model = create_trapped_ion_noise_model()

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
        raise ValueError("Invalid Map Type")

def compute_kernel_matrix(X1, X2, feature_map):
    """Compute kernel matrix using trapped ion noise model"""
    simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
    
    def get_density_matrices(data):
        circuits = []
        for x in data:
            qc = feature_map.assign_parameters(x)
            qc.save_density_matrix()
            circuits.append(qc)
        
        transpiled = transpile(circuits, simulator)
        result = simulator.run(transpiled, shots=None).result()
        
        return [DensityMatrix(result.data(i)['density_matrix']) for i in range(len(data))]

    states1 = get_density_matrices(X1)
    states2 = get_density_matrices(X2)
    
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            K[i, j] = state_fidelity(states1[i], states2[j])
            
    return K

# ==============================================================================
# 5. TRAIN AND EVALUATE EACH CONFIGURATION
# ==============================================================================
results = []

print(f"\n{'='*70}")
print(f"STARTING EVALUATIONS")
print(f"{'='*70}\n")

for map_type, feature_list in FEATURE_CONFIGS.items():
    print(f"\n{'─'*70}")
    print(f"MAP: {map_type} | FEATURES: {feature_list}")
    print(f"{'─'*70}")
    
    t_start = time.time()
    
    # 1. Prepare data for this feature set
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train_full[feature_list])
    X_test = scaler.transform(X_test_full[feature_list])
    
    # 2. Create feature map
    n_features = len(feature_list)
    feature_map = get_feature_map(n_features, map_type)
    
    # 3. Compute kernels
    print(f"Computing Training Kernel ({len(X_train)}x{len(X_train)})...")
    K_train = compute_kernel_matrix(X_train, X_train, feature_map)
    
    print(f"Computing Testing Kernel ({len(X_test)}x{len(X_train)})...")
    K_test = compute_kernel_matrix(X_test, X_train, feature_map)
    
    # 4. Train SVM
    print(f"Training SVM...")
    svc = SVC(kernel='precomputed')
    svc.fit(K_train, y_train)
    
    # 5. Predict
    print(f"Predicting...")
    y_pred = svc.predict(K_test)
    
    # 6. Calculate Metrics
    try:
        y_score = svc.decision_function(K_test)
        auc = roc_auc_score(y_test, y_score)
    except:
        auc = 0.0
    
    acc = accuracy_score(y_test, y_pred)
    
    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    total_time = time.time() - t_start
    
    # Store results
    results.append({
        'Map': map_type,
        'Features': len(feature_list),
        'Accuracy': acc,
        'AUC': auc,
        'Hit_Rate': hit_rate,
        'FAR': far,
        'Time': total_time
    })
    
    # Print individual results
    print(f"\n  ✓ Accuracy:         {acc:.4f}")
    print(f"  ✓ AUC Score:        {auc:.4f}")
    print(f"  ✓ Hit Rate (TPR):   {hit_rate:.4f}")
    print(f"  ✓ False Alarm Rate: {far:.4f}")
    print(f"  ✓ Time:             {total_time:.2f}s")

# ==============================================================================
# 6. SUMMARY TABLE
# ==============================================================================
print(f"\n{'='*70}")
print(f"FINAL SUMMARY - TRAPPED ION QSVM")
print(f"{'='*70}")
print(f"{'Map':<15} {'Features':<10} {'Accuracy':<10} {'AUC':<10} {'Hit Rate':<10} {'FAR':<10} {'Time (s)':<10}")
print(f"{'─'*70}")

for r in results:
    print(f"{r['Map']:<15} {r['Features']:<10} {r['Accuracy']:<10.4f} {r['AUC']:<10.4f} "
          f"{r['Hit_Rate']:<10.4f} {r['FAR']:<10.4f} {r['Time']:<10.1f}")

print(f"{'='*70}")

# Best performing configuration
best_acc = max(results, key=lambda x: x['Accuracy'])
best_auc = max(results, key=lambda x: x['AUC'])

print(f"\n🏆 Best Accuracy: {best_acc['Map']} ({best_acc['Accuracy']:.4f})")
print(f"🏆 Best AUC:      {best_auc['Map']} ({best_auc['AUC']:.4f})")
print(f"{'='*70}\n")