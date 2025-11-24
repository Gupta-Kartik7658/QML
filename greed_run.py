#!/usr/bin/env python3
"""
HYBRID QUANTUM FEATURE SELECTION
Step 1: Exhaustive search for best 2-feature pair (C(30,2) = 435 combinations)
Step 2-7: Greedy forward selection (add best feature one at a time)
Compatible with: qiskit 1.3.1, qiskit-aer-gpu 0.15.1, qiskit-machine-learning 0.8.0
"""

import numpy as np
import pandas as pd
import time
import itertools
from multiprocessing import Pool, cpu_count
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

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset parameters
    DATA_PATH = 'creditcard.csv'
    FRAUD_LABEL = 'Class'
    
    # Feature pre-filter list - 30 FEATURES
    # YOU CAN MODIFY THIS LIST
    FEATURE_PREFILTER_LIST = [
        'V14', 'V7', 'V4', 'V19', 'V20', 'V17', 'V10', 'V12', 'V5', 'V27', 'V22', 'V11',
        'V16', 'V3', 'V9', 'V26', 'V21', 'V8', 'V18', 'V1', 'V6',
        'V2', 'V28', 'V13', 'V15', 'V23', 'V24', 'V25', 'Amount', 'Time'
    ]
    
    # Feature selection parameters
    MIN_FEATURES = 2  # Start with 2-feature exhaustive search
    MAX_FEATURES = 7  # End with 7 features
    
    # Dataset sizes
    TRAIN_SIZE = 400
    TEST_SIZE = 200
    RANDOM_STATE = 42
    
    # Quantum parameters
    FEATURE_MAP_TYPE = 'ZZ'
    FEATURE_MAP_REPS = 1
    ENTANGLEMENT = 'full'
    
    # GPU parameters
    USE_GPU = False
    GPU_DEVICE = 'CPU'
    SHOTS = 1024
    
    # Parallel processing parameters
    N_WORKERS = 6  # Adjust based on your system
    BATCH_SIZE = 6  # Progress update frequency
    
    # Algorithm parameters
    N_TRIALS = 1
    
    # Results output
    RESULTS_FILE = f'Full_ZZFeatureMap_qfs_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
config = Config()

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
            msg += f"!!! ERROR: Fraud label '{config.FRAUD_LABEL}' not found!\n"
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
    
    fraud_idx = y[y == 1].index
    genuine_idx = y[y == 0].index
    
    total = config.TRAIN_SIZE + config.TEST_SIZE
    n_fraud = total // 2
    n_genuine = total // 2
    
    if len(fraud_idx) < n_fraud:
        print(f"[!] WARNING: Not enough fraud samples.")
        n_fraud = len(fraud_idx)
        n_genuine = n_fraud
        total = n_fraud + n_genuine
        config.TRAIN_SIZE = int(total * (config.TRAIN_SIZE / (config.TRAIN_SIZE + config.TEST_SIZE)))
        config.TEST_SIZE = total - config.TRAIN_SIZE

    np.random.seed(config.RANDOM_STATE + trial)
    fraud_sample = np.random.choice(fraud_idx, n_fraud, replace=False)
    genuine_sample = np.random.choice(genuine_idx, n_genuine, replace=False)
    
    sample_idx = np.concatenate([fraud_sample, genuine_sample])
    np.random.shuffle(sample_idx)
    
    X_bal = X.loc[sample_idx]
    y_bal = y.loc[sample_idx]
    
    print(f"[+] Balanced: {len(X_bal)} samples ({n_fraud} fraud, {n_genuine} genuine)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal,
        train_size=config.TRAIN_SIZE,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE + trial,
        stratify=y_bal
    )
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[+] Split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"[+] Normalized to [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns.tolist()

# ============================================================================
# QUANTUM KERNEL & QSVM
# ============================================================================

def compute_quantum_kernel_matrix(X_train, X_test, feature_indices, feature_names, verbose=False):
    """Compute quantum kernel matrix"""
    n_features = len(feature_indices)
    
    X_train_subset = X_train[:, feature_indices]
    X_test_subset = X_test[:, feature_indices]
    
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
    
    if config.USE_GPU:
        simulator = AerSimulator(method='statevector', device='GPU')
    else:
        simulator = AerSimulator(method='statevector', device='CPU')
    
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    kernel_start = time.time()
    K_train = quantum_kernel.evaluate(X_train_subset, X_train_subset)
    K_test = quantum_kernel.evaluate(X_test_subset, X_train_subset)
    kernel_time = time.time() - kernel_start
    
    return K_train, K_test, kernel_time

def train_qsvm_with_kernel(K_train, K_test, y_train, y_test, verbose=False):
    """Train classical SVM with precomputed quantum kernel"""
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    y_pred = svm.predict(K_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        y_score = svm.decision_function(K_test)
        auc = roc_auc_score(y_test, y_score)
    except:
        auc = 0.0
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_ratio = fp / tp if tp > 0 else np.inf
    
    return accuracy, auc, hit_rate, false_alarm_ratio

# ============================================================================
# WORKER FUNCTIONS
# ============================================================================

_worker_data = {}

def init_worker(X_train, X_test, y_train, y_test, feature_names):
    """Initialize worker with data (called once per worker)"""
    global _worker_data
    _worker_data['X_train'] = X_train
    _worker_data['X_test'] = X_test
    _worker_data['y_train'] = y_train
    _worker_data['y_test'] = y_test
    _worker_data['feature_names'] = feature_names

def evaluate_single_combination(combo_idx_pair):
    """Evaluate a single feature combination"""
    combo, idx = combo_idx_pair
    
    X_train = _worker_data['X_train']
    X_test = _worker_data['X_test']
    y_train = _worker_data['y_train']
    y_test = _worker_data['y_test']
    feature_names = _worker_data['feature_names']
    
    try:
        K_train, K_test, kernel_time = compute_quantum_kernel_matrix(
            X_train, X_test, list(combo), feature_names, verbose=False
        )
        
        acc, auc, hr, far = train_qsvm_with_kernel(
            K_train, K_test, y_train, y_test, verbose=False
        )
        
        return (combo, idx, auc, acc, hr, far, True, kernel_time)
        
    except Exception as e:
        return (combo, idx, 0.0, 0.0, 0.0, np.inf, False, 0.0)

# ============================================================================
# HYBRID FEATURE SELECTION ALGORITHM
# ============================================================================

def hybrid_quantum_feature_selection(X_train, y_train, X_test, y_test, feature_names):
    """
    HYBRID APPROACH:
    - Step 1: Exhaustive search for best 2-feature pair (C(30,2) = 435 combos)
    - Steps 2-7: Greedy forward selection (add best feature one at a time)
    """
    
    from math import comb
    
    dataset_size_mb = (X_train.nbytes + X_test.nbytes + y_train.nbytes + y_test.nbytes) / (1024**2)
    
    print("\n" + "=" * 80)
    print("HYBRID QUANTUM FEATURE SELECTION ALGORITHM")
    print("=" * 80)
    print(f"Device: {config.GPU_DEVICE}")
    print(f"Workers: {config.N_WORKERS}")
    print(f"Algorithm: HYBRID (Exhaustive 2-features + Greedy 3-7)")
    
    # Determine feature indices to test
    if config.FEATURE_PREFILTER_LIST:
        print(f"\n[+] Using pre-filter list ({len(config.FEATURE_PREFILTER_LIST)} features):")
        print(f"    {config.FEATURE_PREFILTER_LIST}")
        try:
            available_features = [feature_names.index(name) for name in config.FEATURE_PREFILTER_LIST]
            print(f"    Indices: {available_features}")
        except ValueError as e:
            print(f"\n!!! ERROR: Feature name not found: {e}")
            return {}, []
    else:
        print("\n[+] Using ALL features")
        available_features = list(range(len(feature_names)))
    
    n_avail = len(available_features)
    n_pairs = comb(n_avail, 2)
    
    print(f"\n[*] Algorithm Plan:")
    print(f"    Step 1: Exhaustive 2-feature pairs = C({n_avail},2) = {n_pairs} combinations")
    print(f"    Steps 2-7: Greedy selection = ~{n_avail * 5} additional evaluations")
    print(f"    Total evaluations: ~{n_pairs + n_avail * 5} (vs {comb(n_avail, 7):,} for pure exhaustive!)")
    print(f"    Dataset: {dataset_size_mb:.2f} MB per worker")
    print("=" * 80)
    
    all_results = {}
    selected_features = []
    
    # ========================================================================
    # STEP 1: EXHAUSTIVE SEARCH FOR BEST 2-FEATURE PAIR
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 1: EXHAUSTIVE 2-FEATURE PAIR SEARCH")
    print(f"{'='*80}")
    
    pair_combos = list(itertools.combinations(available_features, 2))
    print(f"\n[*] Testing all {len(pair_combos)} pairs in parallel...")
    print(f"    Using {config.N_WORKERS} workers")
    print(f"    Estimated time: ~{len(pair_combos) * 15 / config.N_WORKERS / 60:.1f} minutes")
    
    # Ask for confirmation for large searches
    if len(pair_combos) > 500:
        response = input(f"\nContinue with {len(pair_combos)} combinations? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return {}, []
    
    stage_start = time.time()
    combos_with_idx = [(combo, idx) for idx, combo in enumerate(pair_combos)]
    
    best_auc = 0.0
    best_pair = None
    best_metrics = None
    completed = 0
    total_kernel_time = 0.0
    
    print(f"\n[>>>] Starting parallel evaluation...")
    print(f"      Progress updates every {config.BATCH_SIZE} combinations\n")
    
    with Pool(processes=config.N_WORKERS,
             initializer=init_worker,
             initargs=(X_train, X_test, y_train, y_test, feature_names)) as pool:
        
        for result in pool.imap_unordered(evaluate_single_combination, combos_with_idx, chunksize=1):
            combo, idx, auc, acc, hr, far, success, kernel_time = result
            
            completed += 1
            total_kernel_time += kernel_time
            
            if success and auc > best_auc:
                best_auc = auc
                best_pair = combo
                best_metrics = (acc, hr, far)
                print(f"  [*][*][*] NEW BEST! AUC: {auc:.4f} [*][*][*]")
                print(f"      Pair: {[feature_names[i] for i in combo]}")
                print(f"      (Acc: {acc:.4f}, HR: {hr:.4f}, FAR: {far:.4f})")
            
            # Progress updates
            if completed % config.BATCH_SIZE == 0:
                elapsed = time.time() - stage_start
                avg_time = elapsed / completed
                remaining_time = avg_time * (len(pair_combos) - completed)
                progress_pct = (completed / len(pair_combos)) * 100
                
                print(f"\n  Progress: {completed}/{len(pair_combos)} ({progress_pct:.1f}%)")
                print(f"  Elapsed: {elapsed/60:.1f}m | Remaining: {remaining_time/60:.1f}m")
                print(f"  Avg: {avg_time:.2f}s/combo | Best AUC so far: {best_auc:.4f}")
    
    stage_time = time.time() - stage_start
    
    if best_pair is None:
        print("\n!!! ERROR: No valid 2-feature pair found!")
        return {}, []
    
    selected_features = list(best_pair)
    best_acc, best_hr, best_far = best_metrics
    
    print(f"\n{'─'*80}")
    print(f"[+] BEST 2-FEATURE PAIR FOUND:")
    print(f"    Features: {[feature_names[i] for i in selected_features]}")
    print(f"    AUC: {best_auc:.4f} | Acc: {best_acc:.4f} | HR: {best_hr:.4f} | FAR: {best_far:.4f}")
    print(f"    Time: {stage_time/60:.1f} minutes ({stage_time:.1f}s)")
    print(f"    Avg: {stage_time/len(pair_combos):.2f}s per combination")
    print(f"{'─'*80}")
    
    # Store 2-feature results
    all_results[2] = {
        'features': selected_features.copy(),
        'feature_names': [feature_names[i] for i in selected_features],
        'auc': best_auc,
        'accuracy': best_acc,
        'hit_rate': best_hr,
        'false_alarm_ratio': best_far
    }
    
    # ========================================================================
    # STEPS 2-7: GREEDY FORWARD SELECTION
    # ========================================================================
    for step in range(3, config.MAX_FEATURES + 1):
        print(f"\n{'='*80}")
        print(f"STEP {step-1}: ADDING FEATURE #{step} (GREEDY)")
        print(f"{'='*80}")
        
        remaining_features = [f for f in available_features if f not in selected_features]
        
        if len(remaining_features) == 0:
            print("[!] No more features to test.")
            break
        
        print(f"[*] Current selection: {[feature_names[i] for i in selected_features]}")
        print(f"    Testing {len(remaining_features)} candidates in parallel...")
        
        test_combos = [(selected_features + [f], idx) for idx, f in enumerate(remaining_features)]
        
        step_start = time.time()
        best_auc_this_step = 0.0
        best_new_feature = None
        best_new_metrics = None
        completed = 0
        
        print(f"\n[>>>] Evaluating candidates...")
        
        with Pool(processes=config.N_WORKERS,
                 initializer=init_worker,
                 initargs=(X_train, X_test, y_train, y_test, feature_names)) as pool:
            
            for result in pool.imap_unordered(evaluate_single_combination, test_combos, chunksize=1):
                combo, idx, auc, acc, hr, far, success, kernel_time = result
                
                completed += 1
                candidate_feature = remaining_features[idx]
                
                if success:
                    print(f"  [{completed:2d}/{len(remaining_features)}] "
                          f"Feature {feature_names[candidate_feature]:8s}: "
                          f"AUC={auc:.4f}, Acc={acc:.4f}, HR={hr:.4f}, FAR={far:.4f}")
                    
                    if auc > best_auc_this_step:
                        best_auc_this_step = auc
                        best_new_feature = candidate_feature
                        best_new_metrics = (acc, hr, far)
                        print(f"       ★ NEW BEST for this step! ★")
                else:
                    print(f"  [{completed:2d}/{len(remaining_features)}] "
                          f"Feature {feature_names[candidate_feature]:8s}: FAILED")
        
        step_time = time.time() - step_start
        
        if best_new_feature is not None:
            selected_features.append(best_new_feature)
            best_auc = best_auc_this_step
            best_acc, best_hr, best_far = best_new_metrics
            
            print(f"\n{'─'*80}")
            print(f"[+] BEST FEATURE FOR STEP {step-1}: {feature_names[best_new_feature]}")
            print(f"    Selected features: {[feature_names[i] for i in selected_features]}")
            print(f"    AUC: {best_auc:.4f} | Acc: {best_acc:.4f} | HR: {best_hr:.4f} | FAR: {best_far:.4f}")
            print(f"    Step time: {step_time:.1f}s")
            print(f"{'─'*80}")
            
            # Store results for this step
            all_results[step] = {
                'features': selected_features.copy(),
                'feature_names': [feature_names[i] for i in selected_features],
                'auc': best_auc,
                'accuracy': best_acc,
                'hit_rate': best_hr,
                'false_alarm_ratio': best_far
            }
        else:
            print(f"\n[!] No valid feature found in step {step-1}. Stopping.")
            break
    
    print(f"\n{'='*80}")
    print(f"HYBRID SELECTION COMPLETE")
    print(f"{'='*80}")
    
    return all_results, selected_features

# ============================================================================
# OUTPUT LOGGER
# ============================================================================

class OutputLogger:
    """Dual logging to console and file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8', buffering=1)
        
        header = f"""{'='*80}
HYBRID QUANTUM FEATURE SELECTION - SESSION LOG
{'='*80}
Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results file: {filepath}
{'='*80}

"""
        self.log.write(header)
        self.log.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        footer = f"""
{'='*80}
Session end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        self.log.write(footer)
        self.log.flush()
        self.log.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    output_logger = OutputLogger(config.RESULTS_FILE)
    sys.stdout = output_logger
    
    try:
        print("=" * 80)
        print("HYBRID QUANTUM FEATURE SELECTION")
        print("Exhaustive 2-Features + Greedy 3-7")
        print("=" * 80)
        print(f"\nStarting: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Available CPU cores: {cpu_count()}\n")
        
        print(f"[*] Configuration:")
        print(f"    Training size: {config.TRAIN_SIZE}")
        print(f"    Test size: {config.TEST_SIZE}")
        print(f"    Feature map: {config.FEATURE_MAP_TYPE}, reps={config.FEATURE_MAP_REPS}")
        print(f"    Entanglement: {config.ENTANGLEMENT}")
        print(f"    Shots: {config.SHOTS}")
        print(f"    Workers: {config.N_WORKERS}")
        print(f"    Results file: {config.RESULTS_FILE}")
        
        print(f"\n[*] Feature Pre-filter: {len(config.FEATURE_PREFILTER_LIST)} features")
        print(f"    Selection range: {config.MIN_FEATURES} to {config.MAX_FEATURES}")
        
        from math import comb
        n_features = len(config.FEATURE_PREFILTER_LIST)
        exhaustive_count = comb(n_features, 7)
        hybrid_count = comb(n_features, 2) + (5 * n_features)
        
        print(f"\n[*] Complexity Comparison:")
        print(f"    Pure exhaustive C({n_features},7): {exhaustive_count:,} evaluations")
        print(f"    Hybrid approach: ~{hybrid_count:,} evaluations")
        print(f"    Speedup: ~{exhaustive_count/hybrid_count:.0f}x faster!")
        
        # Load data
        X_full, y_full = load_data(output_logger)
         
        if X_full is None:
            print("\n[!] Data loading failed. Exiting.")
            return
        
        overall_start = time.time()
        
        # Prepare dataset
        print("\n" + "=" * 80)
        print("PREPARING DATASET")
        print("=" * 80)
        X_train, X_test, y_train, y_test, feature_names = prepare_data(X_full, y_full)
        
        dataset_size_mb = (X_train.nbytes + X_test.nbytes + y_train.nbytes + y_test.nbytes) / (1024**2)
        print(f"\n[*] Dataset: {dataset_size_mb:.2f} MB per worker")
        print(f"    Total memory: ~{dataset_size_mb * config.N_WORKERS:.2f} MB")
        
        # Run hybrid feature selection
        results, best_features = hybrid_quantum_feature_selection(
            X_train, y_train, X_test, y_test, feature_names
        )
        
        overall_time = time.time() - overall_start
        
        # Final results
        print(f"\n{'#'*80}")
        print(f"FINAL RESULTS")
        print(f"{'#'*80}")
        
        if not results:
            print("\n[!] No results to display.")
        else:
            for n_feat, res in sorted(results.items()):
                print(f"\n{'='*80}")
                print(f"BEST {n_feat} FEATURES:")
                print(f"{'='*80}")
                print(f"  Features: {res['feature_names']}")
                print(f"  Feature indices: {res['features']}")
                print(f"  AUC: {res['auc']:.4f}")
                print(f"  Accuracy: {res['accuracy']:.4f}")
                print(f"  Hit Rate (Recall): {res['hit_rate']:.4f}")
                print(f"  False Alarm Ratio (FP/TP): {res['false_alarm_ratio']:.4f}")
        
        print(f"\n{'='*80}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"  Algorithm: Hybrid (Exhaustive 2-pair + Greedy 3-7)")
        print(f"  Total evaluations: ~{hybrid_count}")
        print(f"  Total time: {overall_time/60:.2f} minutes ({overall_time:.1f}s)")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{'='*80}")
        print(f"RESULTS SAVED TO: {config.RESULTS_FILE}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"EXCEPTION OCCURRED")
        print(f"{'='*80}")
        print(traceback.format_exc())
    
    finally:
        sys.stdout = output_logger.terminal
        output_logger.close()
        print(f"\n[OK] Complete! Results saved to: {config.RESULTS_FILE}")

# def resume_step7(X_train, X_test, y_train, y_test, feature_names, selected_features_names):
#     """
#     Continue greedy selection only for the 7th feature.
#     selected_features_names = list of feature names already selected.
#     """

#     # Convert names → indices
#     selected_features = [feature_names.index(name) for name in selected_features_names]

#     # Only prefilter features are used
#     available_indices = [feature_names.index(f) for f in config.FEATURE_PREFILTER_LIST]

#     # Remaining = all prefiler features that are not in the selected list
#     remaining_features = [f for f in available_indices if f not in selected_features]

#     print("\n===============================================================")
#     print("  RESUMING GREEDY FEATURE SELECTION FOR THE 7th FEATURE")
#     print("===============================================================\n")
#     print(f"Already selected (6): {selected_features_names}")
#     print(f"Remaining candidates ({len(remaining_features)}):")
#     print([feature_names[i] for i in remaining_features])
#     print()

#     # Create evaluation combos
#     combos = [(selected_features + [f], idx) for idx, f in enumerate(remaining_features)]

#     best_auc = 0
#     best_feat = None
#     best_metrics = None

#     with Pool(processes=config.N_WORKERS,
#               initializer=init_worker,
#               initargs=(X_train, X_test, y_train, y_test, feature_names)) as pool:

#         for combo, idx, auc, acc, hr, far, success, kt in \
#             pool.imap_unordered(evaluate_single_combination, combos):

#             fname = feature_names[remaining_features[idx]]
#             print(f"Feature {fname}: AUC={auc:.4f}, Acc={acc:.4f}, HR={hr:.4f}, FAR={far:.4f}")

#             if success and auc > best_auc:
#                 best_auc = auc
#                 best_feat = remaining_features[idx]
#                 best_metrics = (acc, hr, far)

#     print("\n=================== BEST 7th FEATURE ===================")
#     if best_feat is None:
#         print("No valid feature found!")
#         return None

#     print(f"Chosen 7th feature: {feature_names[best_feat]}")
#     print(f"AUC: {best_auc:.4f}")
#     print(f"Acc: {best_metrics[0]:.4f}")
#     print(f"HR : {best_metrics[1]:.4f}")
#     print(f"FAR: {best_metrics[2]:.4f}")
#     print("========================================================\n")

#     return best_feat, best_auc, best_metrics



if __name__ == "__main__":
    main()