#!/usr/bin/env python3
"""
QUANTUM KERNEL NOISE SENSITIVITY ANALYSIS
Systematically varies each noise parameter across 10 values for 3 feature maps
Uses multiprocessing to run all 3 feature maps in parallel
Saves results to JSON for each feature map
"""

import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from multiprocessing import Process, Queue
import os

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

# --- Dataset Size (Balanced) ---
TRAIN_SIZE = 400
TEST_SIZE = 200
REPS = 1

# --- Feature Maps Configuration ---
FEATURE_MAP_CONFIGS = {
    'Z': {
        'type': 'Z',
        'features': ['V14', 'V12', 'V4', 'V17', 'V8']
    },
    'ZZ': {
        'type': 'ZZ',
        'features': ['V14', 'V12', 'V4', 'V17', 'V8']
    },
    'Pauli_X_ZZ': {
        'type': 'Pauli_X_ZZ',
        'features': ['V14', 'V12', 'V4', 'V17', 'V8']
    }
}

# --- Baseline Noise Parameters (IBM Eagle "Lower Quartile") ---
BASELINE_NOISE = {
    'T1': 60.0e3,         # 60 µs
    'T2': 30.0e3,         # 30 µs
    'GATE_TIME_1Q': 60.0, # 60 ns
    'GATE_TIME_2Q': 550.0,# 550 ns
    'P_DEP_1Q': 0.0004,   # 0.04%
    'P_DEP_2Q': 0.02,     # 2.0%
    'P_READOUT': 0.04     # 4.0%
}

# --- Noise Parameter Sweep Ranges ---
# Note: T2 must satisfy T2 <= 2*T1 (physical constraint)
NOISE_SWEEPS = {
    'T1': np.linspace(10.0e3, 150.0e3, 10),      # 10 µs to 150 µs
    'T2': np.linspace(5.0e3, 60.0e3, 10),        # 5 µs to 60 µs (safe with baseline T1=60µs)
    'GATE_TIME_1Q': np.linspace(20.0, 200.0, 10), # 20 ns to 200 ns
    'GATE_TIME_2Q': np.linspace(200.0, 1500.0, 10), # 200 ns to 1500 ns
    'P_DEP_1Q': np.linspace(0.0001, 0.002, 10),  # 0.01% to 0.2%
    'P_DEP_2Q': np.linspace(0.005, 0.05, 10),    # 0.5% to 5%
    'P_READOUT': np.linspace(0.01, 0.1, 10)      # 1% to 10%
}

print(f"\n{'='*70}")
print(f"QUANTUM KERNEL NOISE SENSITIVITY ANALYSIS")
print(f"{'='*70}")
print(f"Feature Maps: {list(FEATURE_MAP_CONFIGS.keys())}")
print(f"Parameters to Sweep: {list(NOISE_SWEEPS.keys())}")
print(f"Points per Sweep: 10")
print(f"Train Size: {TRAIN_SIZE} | Test Size: {TEST_SIZE}")
print(f"{'='*70}\n")

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================
def prepare_data(feature_list):
    """Prepare balanced dataset for given features"""
    print(f"  [DEBUG] Loading CSV from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"  [DEBUG] CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"CRITICAL: {DATA_PATH} not found.")
        exit(1)

    print(f"  [DEBUG] Extracting features: {feature_list}")
    X_all = df[feature_list].values
    y_all = df['Class'].values
    print(f"  [DEBUG] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")

    # Balance the Dataset (50/50 Fraud/Genuine)
    fraud_idx = np.where(y_all == 1)[0]
    genuine_idx = np.where(y_all == 0)[0]
    print(f"  [DEBUG] Found {len(fraud_idx)} fraud samples, {len(genuine_idx)} genuine samples")

    required_total = TRAIN_SIZE + TEST_SIZE
    if len(fraud_idx) < required_total // 2:
        half_size = len(fraud_idx)
        print(f"  [DEBUG] WARNING: Using {half_size} per class (limited by fraud samples)")
    else:
        half_size = required_total // 2
        print(f"  [DEBUG] Using {half_size} samples per class")

    print(f"  [DEBUG] Sampling indices...")
    indices = np.concatenate([
        np.random.choice(fraud_idx, half_size, replace=False),
        np.random.choice(genuine_idx, half_size, replace=False)
    ])
    np.random.shuffle(indices)
    print(f"  [DEBUG] Selected {len(indices)} total samples")

    X_bal = X_all[indices]
    y_bal = y_all[indices]

    # Stratified Split
    print(f"  [DEBUG] Performing stratified split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, 
        train_size=TRAIN_SIZE, 
        test_size=TEST_SIZE, 
        stratify=y_bal, 
        random_state=42
    )
    print(f"  [DEBUG] Split complete - Train: {len(X_train)}, Test: {len(X_test)}")

    # Scaling (-1 to 1)
    print(f"  [DEBUG] Scaling features to [-1, 1]...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"  [DEBUG] Scaling complete")

    return X_train, X_test, y_train, y_test

# ==============================================================================
# 3. NOISE MODEL GENERATOR
# ==============================================================================
def create_noise_model(noise_params):
    """Create noise model with given parameters"""
    nm = NoiseModel()
    
    # Ensure T2 <= 2*T1 (physical constraint)
    t1 = noise_params['T1']
    t2 = min(noise_params['T2'], 2 * t1)
    
    # 1. Single Qubit Errors (Thermal + Depolarizing)
    t1_err = thermal_relaxation_error(t1, t2, noise_params['GATE_TIME_1Q'])
    dep_err = depolarizing_error(noise_params['P_DEP_1Q'], 1)
    combined_1q = t1_err.compose(dep_err)
    nm.add_all_qubit_quantum_error(combined_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'id'])
    
    # 2. Two Qubit Errors (Thermal + Depolarizing)
    t2_q0 = thermal_relaxation_error(t1, t2, noise_params['GATE_TIME_2Q'])
    t2_q1 = thermal_relaxation_error(t1, t2, noise_params['GATE_TIME_2Q'])
    t2_err = t2_q0.tensor(t2_q1)
    dep_2q = depolarizing_error(noise_params['P_DEP_2Q'], 2)
    combined_2q = t2_err.compose(dep_2q)
    nm.add_all_qubit_quantum_error(combined_2q, ['cx', 'cz', 'ecr'])
    
    # 3. Readout Errors
    p_ro = noise_params['P_READOUT']
    ro_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    nm.add_all_qubit_readout_error(ro_err)
    
    return nm

# ==============================================================================
# 4. FEATURE MAP SELECTION
# ==============================================================================
def get_feature_map(n_features, map_type):
    """Get feature map based on type"""
    if map_type == 'Z':
        return ZFeatureMap(feature_dimension=n_features, reps=REPS)
    elif map_type == 'ZZ':
        return ZZFeatureMap(feature_dimension=n_features, reps=REPS, entanglement='full')
    elif map_type == 'Pauli_X_ZZ':
        return PauliFeatureMap(feature_dimension=n_features, reps=REPS, paulis=['X', 'ZZ'])
    else:
        raise ValueError("Invalid Map Type")

# ==============================================================================
# 5. KERNEL COMPUTATION
# ==============================================================================
def compute_kernel_matrix(X1, X2, map_type, noise_model, matrix_type="", worker_id="MAIN"):
    """Compute kernel matrix with noise"""
    print(f"[WORKER-{worker_id}]     [DEBUG] Starting {matrix_type} kernel computation: {len(X1)}x{len(X2)}")
    n_features = X1.shape[1]
    feature_map = get_feature_map(n_features, map_type)
    
    # Use Density Matrix simulator
    simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
    
    def get_density_matrices(data, label=""):
        print(f"[WORKER-{worker_id}]       [DEBUG] Creating {len(data)} quantum circuits for {label}...")
        circuits = []
        for idx, x in enumerate(data):
            if idx % 50 == 0:
                print(f"[WORKER-{worker_id}]         [DEBUG] Building circuit {idx}/{len(data)}")
            qc = feature_map.assign_parameters(x)
            qc.save_density_matrix()
            circuits.append(qc)
        
        print(f"[WORKER-{worker_id}]       [DEBUG] Transpiling {len(circuits)} circuits...")
        transpiled = transpile(circuits, simulator)
        
        print(f"[WORKER-{worker_id}]       [DEBUG] Running simulation for {len(circuits)} circuits...")
        sim_start = time.time()
        result = simulator.run(transpiled, shots=None).result()
        print(f"[WORKER-{worker_id}]       [DEBUG] Simulation completed in {time.time()-sim_start:.2f}s")
        
        print(f"[WORKER-{worker_id}]       [DEBUG] Extracting density matrices...")
        return [DensityMatrix(result.data(i)['density_matrix']) for i in range(len(data))]

    # Compute states
    states1 = get_density_matrices(X1, "X1")
    states2 = get_density_matrices(X2, "X2")
    
    # Compute Fidelity Matrix
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    print(f"[WORKER-{worker_id}]       [DEBUG] Computing {n1}x{n2} fidelity matrix...")
    fid_start = time.time()
    total_calcs = n1 * n2
    for i in range(n1):
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - fid_start
            progress = (i * n2) / total_calcs
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            print(f"[WORKER-{worker_id}]         [DEBUG] Fidelity progress: {i}/{n1} rows ({progress*100:.1f}%) - ETA: {eta:.1f}s")
        for j in range(n2):
            K[i, j] = state_fidelity(states1[i], states2[j])
    
    print(f"[WORKER-{worker_id}]       [DEBUG] Fidelity matrix computed in {time.time()-fid_start:.2f}s")
    return K

# ==============================================================================
# 6. SINGLE EVALUATION
# ==============================================================================
def evaluate_qsvm(X_train, X_test, y_train, y_test, map_type, noise_params, worker_id="MAIN"):
    """Run single QSVM evaluation with given noise parameters"""
    
    print(f"[WORKER-{worker_id}]     [DEBUG] Creating noise model...")
    # Create noise model
    noise_model = create_noise_model(noise_params)
    
    print(f"[WORKER-{worker_id}]     [DEBUG] Computing TRAIN kernel matrix...")
    # Compute kernels
    K_train = compute_kernel_matrix(X_train, X_train, map_type, noise_model, "TRAIN", worker_id)
    
    print(f"[WORKER-{worker_id}]     [DEBUG] Computing TEST kernel matrix...")
    K_test = compute_kernel_matrix(X_test, X_train, map_type, noise_model, "TEST", worker_id)
    
    print(f"[WORKER-{worker_id}]     [DEBUG] Training SVM...")
    # Train SVM
    svc = SVC(kernel='precomputed')
    svc.fit(K_train, y_train)
    
    print(f"[WORKER-{worker_id}]     [DEBUG] Predicting...")
    # Predict
    y_pred = svc.predict(K_test)
    
    print(f"[WORKER-{worker_id}]     [DEBUG] Computing metrics...")
    # Metrics
    try:
        y_score = svc.decision_function(K_test)
        auc = roc_auc_score(y_test, y_score)
    except:
        auc = 0.0
    
    acc = accuracy_score(y_test, y_pred)
    
    # Confusion Matrix Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"[WORKER-{worker_id}]     [DEBUG] Evaluation complete - Acc:{acc:.4f} AUC:{auc:.4f}")
    
    return {
        'accuracy': float(acc),
        'auc': float(auc),
        'recall': float(recall),
        'far': float(far)
    }

# ==============================================================================
# 7. NOISE SENSITIVITY SWEEP
# ==============================================================================
# ==============================================================================
# 7. NOISE SENSITIVITY SWEEP (WORKER FUNCTION)
# ==============================================================================
def run_noise_sensitivity_worker(feature_map_name, config, result_queue):
    """Worker function to run complete noise sensitivity analysis for one feature map"""
    
    try:
        print(f"\n{'='*70}")
        print(f"[WORKER-{feature_map_name}] ANALYZING: {feature_map_name} Feature Map")
        print(f"[WORKER-{feature_map_name}] Features: {config['features']}")
        print(f"[WORKER-{feature_map_name}] Process ID: {os.getpid()}")
        print(f"{'='*70}\n")
        
        # Prepare data
        print(f"[WORKER-{feature_map_name}] [DEBUG] Preparing data with features: {config['features']}")
        X_train, X_test, y_train, y_test = prepare_data(config['features'])
        print(f"[WORKER-{feature_map_name}] [DEBUG] Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        
        results = {
            'feature_map': feature_map_name,
            'features': config['features'],
            'baseline_noise': BASELINE_NOISE.copy(),
            'sweeps': {}
        }
        
        # Sweep each parameter
        for param_name, param_values in NOISE_SWEEPS.items():
            print(f"\n[WORKER-{feature_map_name}] --- Sweeping {param_name} ---")
            print(f"[WORKER-{feature_map_name}] Range: {param_values[0]:.2e} to {param_values[-1]:.2e}")
            print(f"[WORKER-{feature_map_name}] [DEBUG] {len(param_values)} values to test")
            
            sweep_results = []
            
            for i, param_value in enumerate(param_values):
                # Create noise config with this parameter varied
                noise_params = BASELINE_NOISE.copy()
                noise_params[param_name] = param_value
                
                print(f"\n[WORKER-{feature_map_name}]   [{i+1}/10] {param_name} = {param_value:.2e}")
                print(f"[WORKER-{feature_map_name}]   [DEBUG] Full noise config: T1={noise_params['T1']:.2e}, T2={noise_params['T2']:.2e}, "
                      f"GT1Q={noise_params['GATE_TIME_1Q']:.2e}, GT2Q={noise_params['GATE_TIME_2Q']:.2e}")
                print(f"[WORKER-{feature_map_name}]   [DEBUG] P_DEP_1Q={noise_params['P_DEP_1Q']:.2e}, P_DEP_2Q={noise_params['P_DEP_2Q']:.2e}, "
                      f"P_RO={noise_params['P_READOUT']:.2e}")
                
                # Evaluate
                start_time = time.time()
                
                metrics = evaluate_qsvm(
                    X_train, X_test, y_train, y_test,
                    config['type'], noise_params, feature_map_name
                )
                
                elapsed = time.time() - start_time
                print(f"[WORKER-{feature_map_name}]   ✓ Completed in {elapsed:.1f}s | Acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f} "
                      f"Recall={metrics['recall']:.4f} FAR={metrics['far']:.4f}")
                
                sweep_results.append({
                    'parameter_value': float(param_value),
                    'metrics': metrics
                })
            
            results['sweeps'][param_name] = sweep_results
            print(f"\n[WORKER-{feature_map_name}] [DEBUG] Completed sweep for {param_name}")
        
        # Save to JSON
        filename = f"noise_sensitivity_{feature_map_name}.json"
        print(f"\n[WORKER-{feature_map_name}] [DEBUG] Saving results to {filename}...")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[WORKER-{feature_map_name}] ✓ Results saved to: {filename}")
        
        # Send completion signal via queue
        result_queue.put({
            'feature_map': feature_map_name,
            'status': 'success',
            'filename': filename
        })
        
    except Exception as e:
        print(f"[WORKER-{feature_map_name}] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'feature_map': feature_map_name,
            'status': 'error',
            'error': str(e)
        })

def run_noise_sensitivity(feature_map_name, config):
    """Legacy function kept for compatibility - now calls worker directly"""
    from multiprocessing import Queue
    q = Queue()
    run_noise_sensitivity_worker(feature_map_name, config, q)
    return q.get()

# ==============================================================================
# 8. MAIN EXECUTION WITH MULTIPROCESSING
# ==============================================================================
if __name__ == "__main__":
    overall_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"STARTING PARALLEL NOISE SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Number of Feature Maps: {len(FEATURE_MAP_CONFIGS)}")
    print(f"Running in parallel using multiprocessing...")
    print(f"{'='*70}\n")
    
    # Create a queue for collecting results
    result_queue = Queue()
    
    # Create and start worker processes for each feature map
    processes = []
    for fm_name, fm_config in FEATURE_MAP_CONFIGS.items():
        print(f"[MAIN] Launching worker for {fm_name}...")
        p = Process(
            target=run_noise_sensitivity_worker,
            args=(fm_name, fm_config, result_queue)
        )
        p.start()
        processes.append(p)
        print(f"[MAIN] Worker for {fm_name} started (PID: {p.pid})")
    
    print(f"\n[MAIN] All {len(processes)} workers launched. Waiting for completion...\n")
    
    # Wait for all processes to complete and collect results
    completed = 0
    all_results = {}
    
    while completed < len(FEATURE_MAP_CONFIGS):
        result = result_queue.get()  # Block until a result is available
        completed += 1
        
        fm_name = result['feature_map']
        if result['status'] == 'success':
            print(f"\n[MAIN] ✓ Worker {fm_name} completed successfully")
            print(f"[MAIN]   Results saved to: {result['filename']}")
            all_results[fm_name] = result
        else:
            print(f"\n[MAIN] ✗ Worker {fm_name} failed with error:")
            print(f"[MAIN]   {result['error']}")
        
        print(f"[MAIN] Progress: {completed}/{len(FEATURE_MAP_CONFIGS)} workers completed\n")
    
    # Join all processes
    print(f"[MAIN] Waiting for all worker processes to terminate...")
    for p in processes:
        p.join()
    
    print(f"[MAIN] All workers terminated.")
    
    # Create summary of all results
    summary = {
        'total_feature_maps': len(FEATURE_MAP_CONFIGS),
        'completed_successfully': sum(1 for r in all_results.values() if r['status'] == 'success'),
        'feature_maps': list(all_results.keys()),
        'files_generated': [r['filename'] for r in all_results.values() if r['status'] == 'success']
    }
    
    with open('noise_sensitivity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'='*70}")
    print(f"PARALLEL ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Total Time: {overall_elapsed/60:.1f} minutes ({overall_elapsed:.1f} seconds)")
    print(f"Successful Completions: {summary['completed_successfully']}/{summary['total_feature_maps']}")
    print(f"\nFiles Generated:")
    for filename in summary['files_generated']:
        print(f"  - {filename}")
    print(f"  - noise_sensitivity_summary.json")
    print(f"{'='*70}\n")