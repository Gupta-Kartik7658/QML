#!/usr/bin/env python3
"""
PARALLEL QUANTUM FEATURE SELECTION - Exhaustive Search
Optimized for 16 logical cores (8 physical cores with hyperthreading)
Compatible with: qiskit 1.3.1, qiskit-aer-gpu 0.15.1, qiskit-machine-learning 0.8.0
"""

import numpy as np
import pandas as pd
import time
import itertools
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
import warnings
import traceback
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel

print("=" * 80)
print("PARALLEL QUANTUM FEATURE SELECTION - EXHAUSTIVE SEARCH")
print("=" * 80)
print(f"\nStarting: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Available CPU cores: {cpu_count()}")
print(f"Will use parallel processing for combination evaluation\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset parameters
    DATA_PATH = 'creditcard.csv'
    FRAUD_LABEL = 'Class'
    
    # Feature pre-filter list
    FEATURE_PREFILTER_LIST = ['V10', 'V14', 'V12', 'V4', 'V17', 'V5', 'V20', 'V27', 'V22', 'V11']
    
    # Feature selection parameters
    MIN_FEATURES = 7
    MAX_FEATURES = 7
    
    # Dataset sizes - KEEP YOUR ORIGINAL VALUES
    TRAIN_SIZE = 400
    TEST_SIZE = 200
    RANDOM_STATE = 40
    
    # Quantum parameters - KEEP YOUR ORIGINAL VALUES
    FEATURE_MAP_TYPE = 'ZZ'
    FEATURE_MAP_REPS = 1
    ENTANGLEMENT = 'linear'
    
    # GPU parameters - KEEP YOUR ORIGINAL VALUES
    USE_GPU = False
    GPU_DEVICE = 'CPU'
    SHOTS = 1024
    
    # Parallel processing parameters - NEW
    N_WORKERS = 5  # Use 14 out of 16 logical cores (leave 2 for system)
    BATCH_SIZE = 5  # Number of combinations to process before updating progress
    
    # Algorithm parameters
    N_TRIALS = 1
    
    # Results output
    RESULTS_FILE = 'parallel_qfs_results.txt'
    
config = Config()

print(f"[*] Configuration:")
print(f"  Training size: {config.TRAIN_SIZE}")
print(f"  Test size: {config.TEST_SIZE}")
print(f"  Feature map depth: {config.FEATURE_MAP_REPS}")
print(f"  Entanglement: {config.ENTANGLEMENT}")
print(f"  Shots: {config.SHOTS}")
print(f"  Parallel workers: {config.N_WORKERS}")
print(f"  Results file: {config.RESULTS_FILE}")

if config.FEATURE_PREFILTER_LIST:
    print(f"\n[+] Feature Pre-filter: Active")
    print(f"     Selecting from: {config.FEATURE_PREFILTER_LIST}")
    print(f"     Range to test: {config.MIN_FEATURES} to {config.MAX_FEATURES} features")
else:
    print(f"\n[+] Feature Pre-filter: INACTIVE")
    print(f"     Selecting from: ALL features")
print()

# ============================================================================
# GPU CHECK
# ============================================================================

def check_gpu_with_timing():
    """Check GPU and measure actual kernel computation time"""
    print("=" * 80)
    print("GPU AVAILABILITY & TIMING TEST")
    print("=" * 80)
    
    try:
        simulator = AerSimulator()
        available_devices = simulator.available_devices()
        print(f"Available devices: {available_devices}")
        
        if 'GPU' not in available_devices:
            print("[-] GPU not available")
            return False
        
        print("[+] GPU is available!\n")
        
        # Test GPU speed vs CPU
        print("Testing GPU vs CPU performance...")
        qc = QuantumCircuit(5)
        for i in range(5):
            qc.h(i)
        for i in range(4):
            qc.cx(i, i+1)
        qc.measure_all()
        
        # CPU test
        cpu_sim = AerSimulator(method='statevector', device='CPU')
        cpu_start = time.time()
        cpu_job = cpu_sim.run(qc, shots=1000)
        cpu_result = cpu_job.result()
        cpu_time = time.time() - cpu_start
        
        # GPU test  
        gpu_sim = AerSimulator(method='statevector', device='GPU')
        gpu_start = time.time()
        gpu_job = gpu_sim.run(qc, shots=1000)
        gpu_result = gpu_job.result()
        gpu_time = time.time() - gpu_start
        
        print(f"\n  CPU time: {cpu_time:.4f}s")
        print(f"  GPU time: {gpu_time:.4f}s")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        
        if gpu_time < cpu_time:
            print(f"  [+] GPU is faster!")
        else:
            print(f"  [!] GPU slower (expected for small circuits)")
            print(f"      GPU has overhead, but will be faster for larger problems")
        
        return True
        
    except Exception as e:
        print(f"[-] Error: {e}")
        return False

if config.USE_GPU:
    gpu_ok = check_gpu_with_timing()
    if not gpu_ok:
        config.USE_GPU = False
        config.GPU_DEVICE = 'CPU'
else:
    print("=" * 80)
    print("GPU CHECK SKIPPED")
    print("=" * 80)
    print("  Config.USE_GPU is set to False. Forcing CPU.")
    config.GPU_DEVICE = 'CPU'

print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(logger=None):
    """Load data with optional logging"""
    msg = "=" * 80 + "\n"
    msg += "LOADING DATA\n"
    msg += "=" * 80 + "\n"
    
    if logger:
        logger.write(msg)
    else:
        print(msg, end='')
    
    try:
        df = pd.read_csv(config.DATA_PATH)
        msg = f"[+] Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n"
        
        if config.FRAUD_LABEL not in df.columns:
            msg += f"!!! ERROR: Fraud label '{config.FRAUD_LABEL}' not found in columns!\n"
            msg += f"    Available columns: {df.columns.tolist()}\n"
            if logger:
                logger.write(msg)
            else:
                print(msg, end='')
            return None, None
            
        msg += f"  Fraud: {df[config.FRAUD_LABEL].sum()}\n"
        msg += f"  Genuine: {len(df) - df[config.FRAUD_LABEL].sum()}\n"
        
        if logger:
            logger.write(msg)
        else:
            print(msg, end='')
        
        X = df.drop(columns=[config.FRAUD_LABEL])
        y = df[config.FRAUD_LABEL]
    
        return X, y
        
    except FileNotFoundError:
        msg = f"!!! ERROR: Data file not found at '{config.DATA_PATH}'\n"
        if logger:
            logger.write(msg)
        else:
            print(msg, end='')
        return None, None

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(X, y, trial=0):
    print(f"\n{'='*80}")
    print(f"PREPARING DATA - Trial {trial + 1}")
    print(f"{'='*80}")
    
    # Balance dataset
    fraud_idx = y[y == 1].index
    genuine_idx = y[y == 0].index
    
    total = config.TRAIN_SIZE + config.TEST_SIZE
    n_fraud = total // 2
    n_genuine = total // 2
    
    if len(fraud_idx) < n_fraud:
        print(f"!!! WARNING: Not enough fraud samples. Requested {n_fraud}, have {len(fraud_idx)}.")
        n_fraud = len(fraud_idx)
        n_genuine = n_fraud
        total = n_fraud + n_genuine
        config.TRAIN_SIZE = int(total * (config.TRAIN_SIZE / (config.TRAIN_SIZE + config.TEST_SIZE)))
        config.TEST_SIZE = total - config.TRAIN_SIZE
        print(f"    New total: {total}, Train: {config.TRAIN_SIZE}, Test: {config.TEST_SIZE}")

    np.random.seed(config.RANDOM_STATE + trial)
    fraud_sample = np.random.choice(fraud_idx, n_fraud, replace=False)
    genuine_sample = np.random.choice(genuine_idx, n_genuine, replace=False)
    
    sample_idx = np.concatenate([fraud_sample, genuine_sample])
    np.random.shuffle(sample_idx)
    
    X_bal = X.loc[sample_idx]
    y_bal = y.loc[sample_idx]
    
    print(f"[+] Balanced: {len(X_bal)} samples ({n_fraud} fraud, {n_genuine} genuine)")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal,
        train_size=config.TRAIN_SIZE,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE + trial,
        stratify=y_bal
    )
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[+] Split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"[+] Normalized to [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns.tolist()

# ============================================================================
# QUANTUM KERNEL COMPUTATION
# ============================================================================

def compute_quantum_kernel_matrix(X_train, X_test, feature_indices, feature_names, verbose=False):
    """Compute quantum kernel matrix"""
    n_features = len(feature_indices)
    
    if verbose:
        print(f"\n  [*] Computing Quantum Kernel Matrix")
        print(f"       Features: {[feature_names[i] for i in feature_indices]}")
        print(f"       Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Select features
    X_train_subset = X_train[:, feature_indices]
    X_test_subset = X_test[:, feature_indices]
    
    # Create feature map
    if config.FEATURE_MAP_TYPE == 'ZZ':
        feature_map = ZZFeatureMap(
            feature_dimension=n_features,
            reps=config.FEATURE_MAP_REPS,
            entanglement=config.ENTANGLEMENT
        )
    else:
        feature_map = ZFeatureMap(
            feature_dimension=n_features,
            reps=config.FEATURE_MAP_REPS
        )
    
    # Create simulator
    if config.USE_GPU:
        simulator = AerSimulator(method='statevector', device='GPU')
    else:
        simulator = AerSimulator(method='statevector', device='CPU')
    
    # Create quantum kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # Compute kernel matrices
    kernel_start = time.time()
    K_train = quantum_kernel.evaluate(X_train_subset, X_train_subset)
    K_test = quantum_kernel.evaluate(X_test_subset, X_train_subset)
    kernel_time = time.time() - kernel_start
    
    return K_train, K_test, kernel_time

# ============================================================================
# TRAIN QSVM
# ============================================================================

def train_qsvm_with_kernel(K_train, K_test, y_train, y_test, verbose=False):
    """Train classical SVM with precomputed quantum kernel"""
    
    # Train SVM
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    
    # Predict
    y_pred = svm.predict(K_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        y_score = svm.decision_function(K_test)
        auc = roc_auc_score(y_test, y_score)
    except:
        auc = 0.0
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_ratio = fp / tp if tp > 0 else np.inf
    
    if verbose:
        print(f"\n       [*] Results:")
        print(f"         Accuracy: {accuracy:.4f}")
        print(f"         AUC: {auc:.4f}")
        print(f"         Hit Rate: {hit_rate:.4f}")
        print(f"         False Alarm Ratio: {false_alarm_ratio:.4f}")
    
    return accuracy, auc, hit_rate, false_alarm_ratio

# ============================================================================
# GLOBAL DATA STORAGE FOR MULTIPROCESSING (WINDOWS COMPATIBLE)
# ============================================================================
# On Windows, workers are spawned (not forked), so we explicitly pass data
_worker_data = {}

def init_worker(X_train, X_test, y_train, y_test, feature_names):
    """
    Initialize worker process with dataset.
    Called ONCE per worker when the pool starts.
    
    WINDOWS NOTE:
    - Windows uses 'spawn' (not 'fork'), so each worker is a fresh process
    - Data is explicitly passed and copied to each worker
    - Still efficient: Initialization happens ONCE per worker, not per task
    - For 10 workers and 120 tasks, data is copied 10 times (not 120 times!)
    """
    global _worker_data
    _worker_data['X_train'] = X_train
    _worker_data['X_test'] = X_test
    _worker_data['y_train'] = y_train
    _worker_data['y_test'] = y_test
    _worker_data['feature_names'] = feature_names
    # No print to avoid spam from multiple workers

# ============================================================================
# PARALLEL WORKER FUNCTION
# ============================================================================

def evaluate_single_combination(combo_idx_pair):
    """
    Worker function to evaluate a single feature combination.
    Accesses dataset from _worker_data (initialized once per worker).
    
    Returns: (combo, idx, auc, accuracy, hit_rate, false_alarm_ratio, success, kernel_time)
    """
    combo, idx = combo_idx_pair
    
    # Access the worker's copy of the dataset
    X_train = _worker_data['X_train']
    X_test = _worker_data['X_test']
    y_train = _worker_data['y_train']
    y_test = _worker_data['y_test']
    feature_names = _worker_data['feature_names']
    
    try:
        # Compute quantum kernel
        K_train, K_test, kernel_time = compute_quantum_kernel_matrix(
            X_train, X_test, list(combo), feature_names, verbose=False
        )
        
        # Train QSVM
        acc, auc, hr, far = train_qsvm_with_kernel(
            K_train, K_test, y_train, y_test, verbose=False
        )
        
        return (combo, idx, auc, acc, hr, far, True, kernel_time)
        
    except Exception as e:
        # Return failure indicator
        return (combo, idx, 0.0, 0.0, 0.0, np.inf, False, 0.0)

# ============================================================================
# PARALLEL FEATURE SELECTION
# ============================================================================

def parallel_quantum_feature_selection(X_train, y_train, X_test, y_test, feature_names):
    """Exhaustive quantum feature selection with parallel processing"""
    
    # Calculate dataset size for reporting
    dataset_size_mb = (X_train.nbytes + X_test.nbytes + y_train.nbytes + y_test.nbytes) / (1024**2)
    
    print("\n" + "=" * 80)
    print("PARALLEL QUANTUM FEATURE SELECTION ALGORITHM")
    print("=" * 80)
    print(f"Device: {config.GPU_DEVICE}")
    print(f"Workers: {config.N_WORKERS}")
    print(f"Algorithm: EXHAUSTIVE SEARCH (all combinations)")
    print(f"Full feature list: {len(feature_names)} features")
    print(f"Selection range: {config.MIN_FEATURES}-{config.MAX_FEATURES}")
    print(f"Dataset: {dataset_size_mb:.2f} MB per worker")
    
    # Determine feature indices to test
    if config.FEATURE_PREFILTER_LIST:
        print(f"\nUsing pre-filter list: {config.FEATURE_PREFILTER_LIST}")
        try:
            feature_indices_to_test = [feature_names.index(name) for name in config.FEATURE_PREFILTER_LIST]
            print(f"Indices to test: {feature_indices_to_test}")
        except ValueError as e:
            print(f"\n!!! ERROR: Feature name in FEATURE_PREFILTER_LIST not found: {e}")
            return {}, []
    else:
        print("\nNo pre-filter list specified. Using ALL features.")
        feature_indices_to_test = list(range(len(feature_names)))
        
    print(f"Total features in selection pool: {len(feature_indices_to_test)}")
    print("=" * 80)
    
    all_results = {}
    selected_features = []
    
    best_auc = 0.0
    best_acc_at_best_auc = 0.0
    best_hr_at_best_auc = 0.0
    best_far_at_best_auc = np.inf
    
    for n_features in range(config.MIN_FEATURES, config.MAX_FEATURES + 1):
        print(f"\n{'='*80}")
        print(f"SELECTING BEST {n_features} FEATURES (PARALLEL)")
        print(f"{'='*80}")
        
        if n_features == config.MIN_FEATURES:
            # First iteration - try all combinations in parallel
            remaining = feature_indices_to_test
            combos = list(itertools.combinations(remaining, n_features))
            
            print(f"\n[*] Total combinations to test: {len(combos)}")
            print(f"   Using {config.N_WORKERS} parallel workers")
            print(f"   Expected speedup: ~{config.N_WORKERS}x")
            
            if len(combos) > 1000:
                print(f"\n[!]?  WARNING: {len(combos)} combinations is very large!")
                print(f"    Estimated time: {len(combos) * 15 / config.N_WORKERS / 60:.1f} minutes")
                response = input(f"\nContinue? (y/n): ")
                if response.lower() != 'y':
                    print("Exiting...")
                    return all_results, selected_features
            
            best_combo = None
            stage_start = time.time()
            
            # Prepare data for parallel processing
            # Add index to each combo for progress tracking
            combos_with_idx = [(combo, idx) for idx, combo in enumerate(combos)]
            
            # Run parallel processing
            print(f"\n[>>>] Starting parallel processing...")
            print(f"   Progress updates every {config.BATCH_SIZE} combinations")
            print(f"   Dataset copied to each of {config.N_WORKERS} workers (Windows spawn method)")
            
            completed = 0
            total_kernel_time = 0.0
            
            # Initialize pool with worker initialization function
            # On Windows, each worker gets its own copy of the data
            with Pool(processes=config.N_WORKERS, 
                     initializer=init_worker,
                     initargs=(X_train, X_test, y_train, y_test, feature_names)) as pool:
                # Use imap_unordered for progress tracking
                # Note: No need to pass data in partial function anymore!
                for result in pool.imap_unordered(evaluate_single_combination, combos_with_idx, chunksize=1):
                    combo, idx, auc, acc, hr, far, success, kernel_time = result
                    
                    completed += 1
                    total_kernel_time += kernel_time
                    
                    if success and auc > best_auc:
                        best_auc = auc
                        best_combo = combo
                        best_acc_at_best_auc = acc
                        best_hr_at_best_auc = hr
                        best_far_at_best_auc = far
                        print(f"\n  [*][*][*] NEW BEST! AUC: {auc:.4f} [*][*][*]")
                        print(f"      Features: {[feature_names[i] for i in combo]}")
                        print(f"      (Acc: {acc:.4f}, HR: {hr:.4f}, FAR: {far:.4f})")
                    
                    # Progress update
                    if completed % config.BATCH_SIZE == 0:
                        elapsed = time.time() - stage_start
                        avg_time = elapsed / completed
                        remaining_time = avg_time * (len(combos) - completed)
                        progress_pct = (completed / len(combos)) * 100
                        
                        print(f"\n  Progress: {completed}/{len(combos)} ({progress_pct:.1f}%)")
                        print(f"  Elapsed: {elapsed/60:.1f}m | Remaining: {remaining_time/60:.1f}m")
                        print(f"  Avg time per combo: {avg_time:.2f}s")
                        print(f"  Total kernel time: {total_kernel_time:.1f}s")
            
            total_elapsed = time.time() - stage_start
            print(f"\n  [+] Completed all {len(combos)} combinations in {total_elapsed/60:.1f} minutes")
            print(f"  [+] Average time per combo: {total_elapsed/len(combos):.2f}s")
            print(f"  [+] Effective speedup: ~{len(combos) * 15 / total_elapsed:.1f}x (vs sequential)")
            
            if best_combo:
                selected_features = list(best_combo)
            else:
                print(f"!!! No combinations succeeded for {n_features} features. Stopping.")
                return all_results, selected_features
        
        else:
            # Incremental selection - add one feature (also parallelized)
            remaining = [f for f in feature_indices_to_test if f not in selected_features]
            
            print(f"\nAdding 1 feature to {len(selected_features)} selected")
            print(f"Testing {len(remaining)} candidates in parallel...")
            
            best_new_auc = best_auc
            best_new_feat = None
            
            # Create test combinations
            test_combos = [(tuple(selected_features + [f]), idx) for idx, f in enumerate(remaining)]
            
            print(f"\n[>>>] Testing {len(remaining)} features in parallel...")
            
            stage_start = time.time()
            completed = 0
            
            # Initialize pool with worker initialization function
            with Pool(processes=config.N_WORKERS,
                     initializer=init_worker,
                     initargs=(X_train, X_test, y_train, y_test, feature_names)) as pool:
                for result in pool.imap_unordered(evaluate_single_combination, test_combos, chunksize=1):
                    combo, idx, auc, acc, hr, far, success, kernel_time = result
                    
                    completed += 1
                    new_feat = remaining[idx]
                    
                    if success and auc > best_new_auc:
                        best_new_auc = auc
                        best_new_feat = new_feat
                        best_acc_at_best_auc = acc
                        best_hr_at_best_auc = hr
                        best_far_at_best_auc = far
                        print(f"\n  [*][*][*] NEW BEST! AUC: {auc:.4f} [*][*][*]")
                        print(f"      Added feature: {feature_names[new_feat]}")
                        print(f"      (Acc: {acc:.4f}, HR: {hr:.4f}, FAR: {far:.4f})")
                    
                    print(f"  Progress: {completed}/{len(remaining)}", end='\r')
            
            total_elapsed = time.time() - stage_start
            print(f"\n  [+] Tested all candidates in {total_elapsed:.1f}s")
            
            if best_new_feat is not None:
                selected_features.append(best_new_feat)
                best_auc = best_new_auc
            else:
                print(f"!!! No new feature improved AUC. Stopping.")
                return all_results, selected_features
        
        print(f"\n{'='*80}")
        print(f"BEST {n_features} FEATURES:")
        print(f"  {[feature_names[i] for i in selected_features]}")
        print(f"  AUC: {best_auc:.4f}")
        print(f"  Accuracy: {best_acc_at_best_auc:.4f}")
        print(f"  Hit Rate: {best_hr_at_best_auc:.4f}")
        print(f"  False Alarm Ratio: {best_far_at_best_auc:.4f}")
        print(f"{'='*80}")
        
        all_results[n_features] = {
            'features': selected_features.copy(),
            'feature_names': [feature_names[i] for i in selected_features],
            'auc': best_auc,
            'accuracy': best_acc_at_best_auc,
            'hit_rate': best_hr_at_best_auc,
            'false_alarm_ratio': best_far_at_best_auc
        }
    
    return all_results, selected_features

# ============================================================================
# OUTPUT LOGGER CLASS
# ============================================================================

class OutputLogger:
    """Dual logging to both console and file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.log.flush()
    
    def close(self):
        self.log.close()

# ============================================================================
# DATA PREPARATION HELPER
# ============================================================================

def prepare_balanced_dataset(X_full, y_full):
    """Prepare and balance the dataset once for all trials"""
    trial = 0
    X_train, X_test, y_train, y_test, feature_names = prepare_data(X_full, y_full, trial=trial)
    return X_train, X_test, y_train, y_test, feature_names

# ============================================================================
# MAIN
# ============================================================================

def main():
    
    # ========================================================================
    # INITIALIZE OUTPUT LOGGING TO FILE FIRST
    # ========================================================================
    output_logger = OutputLogger(config.RESULTS_FILE)
    sys.stdout = output_logger
    
    try:
        # Print header and configuration
        print("=" * 80)
        print("PARALLEL QUANTUM FEATURE SELECTION - EXHAUSTIVE SEARCH")
        print("WINDOWS COMPATIBLE VERSION")
        print("=" * 80)
        print(f"\nStarting: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Available CPU cores: {cpu_count()}")
        print(f"Platform: Windows (using 'spawn' multiprocessing)\n")
        
        print(f"[*] Configuration:")
        print(f"  Training size: {config.TRAIN_SIZE}")
        print(f"  Test size: {config.TEST_SIZE}")
        print(f"  Feature map depth: {config.FEATURE_MAP_REPS}")
        print(f"  Entanglement: {config.ENTANGLEMENT}")
        print(f"  Shots: {config.SHOTS}")
        print(f"  Parallel workers: {config.N_WORKERS}")
        print(f"  Results file: {config.RESULTS_FILE}")
        
        if config.FEATURE_PREFILTER_LIST:
            print(f"\n[+] Feature Pre-filter: Active")
            print(f"     Selecting from: {config.FEATURE_PREFILTER_LIST}")
            print(f"     Range to test: {config.MIN_FEATURES} to {config.MAX_FEATURES} features")
        else:
            print(f"\n[+] Feature Pre-filter: INACTIVE")
            print(f"     Selecting from: ALL features")
        
        print(f"\n[*] MEMORY OPTIMIZATION:")
        print(f"   Single balanced dataset created ONCE")
        print(f"   On Windows: Each worker gets a copy (spawn method)")
        print(f"   Still efficient: Workers initialized once, not per task")
        print()
        
        # ====================================================================
        # LOAD DATA
        # ====================================================================
        X_full, y_full = load_data(output_logger)
        
        if X_full is None:
            print("\nHalting execution due to data loading error.")
            return
        
        print("\n" + "=" * 80)
        print("STARTING PARALLEL QUANTUM FEATURE SELECTION")
        print("=" * 80)
        
        overall_start = time.time()
        
        # ====================================================================
        # STEP 1: PREPARE DATASET ONCE
        # ====================================================================
        print("\n? STEP 1: Creating balanced dataset (ONE TIME ONLY)")
        X_train, X_test, y_train, y_test, feature_names = prepare_balanced_dataset(X_full, y_full)
        
        dataset_size_mb = (X_train.nbytes + X_test.nbytes + y_train.nbytes + y_test.nbytes) / (1024**2)
        print(f"\n[*] Dataset size: {dataset_size_mb:.2f} MB")
        print(f"   On Windows: Each of {config.N_WORKERS} workers gets a copy")
        print(f"   Memory per worker: {dataset_size_mb:.2f} MB")
        print(f"   Total memory: ~{dataset_size_mb * config.N_WORKERS:.2f} MB")
        print(f"   But: Workers initialized ONCE (not per task!)")
        print(f"   Efficiency: {config.N_WORKERS} copies for 120 tasks (not 120 copies!)")
        
        # ====================================================================
        # STEP 2: RUN PARALLEL FEATURE SELECTION
        # ====================================================================
        print(f"\n[*] STEP 2: Running exhaustive feature selection")
        results, best_features = parallel_quantum_feature_selection(
            X_train, y_train, X_test, y_test, feature_names
        )
        
        overall_time = time.time() - overall_start
        
        # ====================================================================
        # FINAL SUMMARY AND RESULTS
        # ====================================================================
        print(f"\n{'#'*80}")
        print(f"FINAL RESULTS")
        print(f"{'#'*80}")
        
        if not results:
            print("\nNo results to display.")
        else:
            for n_feat, res in results.items():
                print(f"\n{'='*80}")
                print(f"BEST {n_feat} FEATURES:")
                print(f"{'='*80}")
                print(f"  Features: {res['feature_names']}")
                print(f"  Feature indices: {res['features']}")
                print(f"  AUC: {res['auc']:.4f}")
                print(f"  Accuracy: {res['accuracy']:.4f}")
                print(f"  Hit Rate (Recall): {res['hit_rate']:.4f}")
                print(f"  False Alarm Ratio (FP/TP): {res['false_alarm_ratio']:.4f}")
                print(f"{'='*80}")
        
        print(f"\n{'='*80}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"  Algorithm: Exhaustive Search")
        print(f"  Dataset created: 1 time")
        print(f"  Dataset size: {dataset_size_mb:.2f} MB")
        print(f"  Workers used: {config.N_WORKERS}")
        print(f"  Platform: Windows (spawn method)")
        print(f"  Total execution time: {overall_time/60:.2f} minutes ({overall_time:.1f} seconds)")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{'='*80}")
        print(f"RESULTS SAVED TO: {config.RESULTS_FILE}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n\n{'='*80}")
        print(f"EXCEPTION OCCURRED")
        print(f"{'='*80}")
        print(traceback.format_exc())
        print(f"{'='*80}")
    
    finally:
        # Close the output logger
        sys.stdout = output_logger.terminal
        output_logger.close()
        
        print(f"\n[OK] Complete! Results saved to: {config.RESULTS_FILE}")

if __name__ == "__main__":
    main()