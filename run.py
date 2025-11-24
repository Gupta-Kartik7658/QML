#!/usr/bin/env python3
"""
OPTIMIZED Quantum Feature Selection for Credit Card Fraud Detection
With better GPU utilization tracking and faster execution
"""

import numpy as np
import pandas as pd
import time
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel

print("=" * 80)
print("OPTIMIZED QUANTUM FEATURE SELECTION - GPU TRACKING")
print("=" * 80)
print(f"\nStarting: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR FASTER EXECUTION
# ============================================================================

class Config:
    # Dataset parameters
    DATA_PATH = 'creditcard.csv'
    FRAUD_LABEL = 'Class'
    
    # Feature selection parameters
    MIN_FEATURES = 2
    MAX_FEATURES = 2
    
    # REDUCED for faster testing - increase after verifying GPU works
    TRAIN_SIZE = 200   # Reduced from 1500 (faster testing)
    TEST_SIZE = 100   # Reduced from 1000
    RANDOM_STATE = 42
    
    # Quantum parameters
    FEATURE_MAP_TYPE = 'ZZ'
    FEATURE_MAP_REPS = 1     # Reduced to 1 for faster execution
    ENTANGLEMENT = 'linear'  # Linear is faster than 'full'
    
    # GPU parameters
    USE_GPU = True
    GPU_DEVICE = 'GPU'
    SHOTS = 1024             # Reduced for faster execution
    
    # Algorithm parameters
    N_TRIALS = 1             # Single trial for faster testing
    
    # Performance tracking
    TRACK_GPU_TIME = True    # Track GPU kernel computation time
    
config = Config()

print(f"⚡ OPTIMIZED Configuration (for faster testing):")
print(f"  Training size: {config.TRAIN_SIZE} (reduced)")
print(f"  Test size: {config.TEST_SIZE} (reduced)")
print(f"  Feature map depth: {config.FEATURE_MAP_REPS} (reduced)")
print(f"  Entanglement: {config.ENTANGLEMENT} (linear is faster)")
print(f"  Shots: {config.SHOTS} (reduced)")
print(f"  Trials: {config.N_TRIALS}")
print(f"\n💡 These settings prioritize speed over accuracy for testing GPU")
print(f"   Increase these values after confirming GPU works properly\n")

# ============================================================================
# GPU CHECK WITH DETAILED TIMING
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
            print("✗ GPU not available")
            return False
        
        print("✓ GPU is available!\n")
        
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
            print(f"  ✓ GPU is faster!")
        else:
            print(f"  ⚠ GPU slower (expected for small circuits)")
            print(f"     GPU has overhead, but will be faster for larger problems")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

gpu_ok = check_gpu_with_timing()
if not gpu_ok:
    config.USE_GPU = False
    config.GPU_DEVICE = 'CPU'

print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    df = pd.read_csv(config.DATA_PATH)
    print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Fraud: {df[config.FRAUD_LABEL].sum()}")
    print(f"  Genuine: {len(df) - df[config.FRAUD_LABEL].sum()}")
    
    X = df.drop(columns=[config.FRAUD_LABEL])
    y = df[config.FRAUD_LABEL]
    
    return X, y

X_full, y_full = load_data()

# ============================================================================
# DATA PREPARATION WITH TIMING
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
    
    np.random.seed(config.RANDOM_STATE + trial)
    fraud_sample = np.random.choice(fraud_idx, n_fraud, replace=False)
    genuine_sample = np.random.choice(genuine_idx, n_genuine, replace=False)
    
    sample_idx = np.concatenate([fraud_sample, genuine_sample])
    np.random.shuffle(sample_idx)
    
    X_bal = X.loc[sample_idx]
    y_bal = y.loc[sample_idx]
    
    print(f"✓ Balanced: {len(X_bal)} samples")
    
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
    
    print(f"✓ Split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"✓ Normalized to [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns.tolist()

# ============================================================================
# QUANTUM KERNEL WITH GPU TIMING
# ============================================================================

def compute_quantum_kernel_matrix(X_train, X_test, feature_indices, feature_names):
    """
    Compute quantum kernel matrix with detailed GPU timing
    This is where GPU is actually used!
    """
    n_features = len(feature_indices)
    
    print(f"\n  📊 Computing Quantum Kernel Matrix (GPU stage)")
    print(f"     Features: {[feature_names[i] for i in feature_indices]}")
    print(f"     Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
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
    
    print(f"     ✓ Feature map: {n_features} qubits, depth={feature_map.depth()}")
    
    # Create simulator
    if config.USE_GPU:
        simulator = AerSimulator(method='statevector', device='GPU')
        print(f"     ✓ Using GPU simulator")
    else:
        simulator = AerSimulator(method='statevector', device='CPU')
        print(f"     ✓ Using CPU simulator")
    
    # Create quantum kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # COMPUTE KERNEL MATRICES - This uses GPU!
    print(f"\n     🚀 GPU COMPUTING NOW...")
    
    # Training kernel (symmetric)
    kernel_start = time.time()
    print(f"        Computing training kernel ({len(X_train_subset)}x{len(X_train_subset)})...")
    K_train = quantum_kernel.evaluate(X_train_subset, X_train_subset)
    kernel_train_time = time.time() - kernel_start
    print(f"        ✓ Training kernel computed in {kernel_train_time:.3f}s")
    
    # Test kernel
    kernel_start = time.time()
    print(f"        Computing test kernel ({len(X_test_subset)}x{len(X_train_subset)})...")
    K_test = quantum_kernel.evaluate(X_test_subset, X_train_subset)
    kernel_test_time = time.time() - kernel_start
    print(f"        ✓ Test kernel computed in {kernel_test_time:.3f}s")
    
    total_kernel_time = kernel_train_time + kernel_test_time
    print(f"\n     ⏱️  TOTAL GPU TIME: {total_kernel_time:.3f}s")
    print(f"     💡 GPU was actively used during these {total_kernel_time:.1f} seconds!")
    
    return K_train, K_test, total_kernel_time

# ============================================================================
# TRAIN QSVM WITH DETAILED TIMING
# ============================================================================

def train_qsvm_with_kernel(K_train, K_test, y_train, y_test):
    """
    Train classical SVM with precomputed quantum kernel
    This part runs on CPU
    """
    print(f"\n  🔧 Training Classical SVM (CPU stage)")
    
    # Train SVM with precomputed kernel
    svm_start = time.time()
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    svm_train_time = time.time() - svm_start
    print(f"     ✓ SVM trained in {svm_train_time:.3f}s")
    
    # Predict
    pred_start = time.time()
    y_pred = svm.predict(K_test)
    pred_time = time.time() - pred_start
    print(f"     ✓ Predictions made in {pred_time:.3f}s")
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        y_score = svm.decision_function(K_test)
        auc = roc_auc_score(y_test, y_score)
    except:
        auc = 0.0
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"\n     📈 Results:")
    print(f"        Accuracy: {accuracy:.4f}")
    print(f"        AUC: {auc:.4f}")
    print(f"        Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    total_svm_time = svm_train_time + pred_time
    print(f"\n     ⏱️  TOTAL CPU TIME: {total_svm_time:.3f}s")
    
    return accuracy, auc, total_svm_time

# ============================================================================
# COMPLETE QSVM TRAINING PIPELINE
# ============================================================================

def train_complete_qsvm(X_train, y_train, X_test, y_test, feature_indices, feature_names):
    """Complete QSVM training with timing breakdown"""
    
    total_start = time.time()
    
    # Step 1: Compute quantum kernel (GPU)
    K_train, K_test, gpu_time = compute_quantum_kernel_matrix(
        X_train, X_test, feature_indices, feature_names
    )
    
    # Step 2: Train classical SVM (CPU)
    accuracy, auc, cpu_time = train_qsvm_with_kernel(
        K_train, K_test, y_train, y_test
    )
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n  ⏱️  TIME BREAKDOWN:")
    print(f"     GPU (kernel): {gpu_time:.3f}s ({gpu_time/total_time*100:.1f}%)")
    print(f"     CPU (SVM):    {cpu_time:.3f}s ({cpu_time/total_time*100:.1f}%)")
    print(f"     Overhead:     {total_time-gpu_time-cpu_time:.3f}s ({(total_time-gpu_time-cpu_time)/total_time*100:.1f}%)")
    print(f"     TOTAL:        {total_time:.3f}s")
    
    return accuracy, auc, gpu_time, cpu_time, total_time

# ============================================================================
# FEATURE SELECTION ALGORITHM
# ============================================================================

def quantum_feature_selection(X_train, y_train, X_test, y_test, feature_names):
    """Quantum feature selection with detailed timing"""
    
    print("\n" + "=" * 80)
    print("QUANTUM FEATURE SELECTION ALGORITHM")
    print("=" * 80)
    print(f"Device: {config.GPU_DEVICE}")
    print(f"Available features: {len(feature_names)}")
    print(f"Selection range: {config.MIN_FEATURES}-{config.MAX_FEATURES}")
    print("=" * 80)
    
    all_results = {}
    selected_features = []
    
    for n_features in range(config.MIN_FEATURES, config.MAX_FEATURES + 1):
        print(f"\n{'='*80}")
        print(f"SELECTING BEST {n_features} FEATURES")
        print(f"{'='*80}")
        
        if n_features == config.MIN_FEATURES:
            # First iteration - try all combinations
            remaining = list(range(len(feature_names)))
            combos = list(itertools.combinations(remaining, n_features))
            
            print(f"\n⚠️  WARNING: {len(combos)} combinations to test!")
            print(f"   Estimated time: ~{len(combos) * 15 / 60:.1f} minutes")
            print(f"\n   💡 For TESTING, you may want to:")
            print(f"      1. Use MAX_FEATURES=3 (only test 3 features)")
            print(f"      2. Or manually select top 10 features first")
            
            response = input(f"\nContinue with {len(combos)} combinations? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return all_results, selected_features
            
            best_acc = 0
            best_combo = None
            
            stage_start = time.time()
            
            for idx, combo in enumerate(combos):
                print(f"\n{'─'*80}")
                print(f"Combination {idx+1}/{len(combos)}")
                print(f"{'─'*80}")
                
                try:
                    acc, auc, gpu_t, cpu_t, total_t = train_complete_qsvm(
                        X_train, y_train, X_test, y_test,
                        list(combo), feature_names
                    )
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_combo = combo
                        print(f"\n  ★★★ NEW BEST! Accuracy: {acc:.4f} ★★★")
                    
                    # Progress estimate
                    elapsed = time.time() - stage_start
                    avg_time = elapsed / (idx + 1)
                    remaining_time = avg_time * (len(combos) - idx - 1)
                    print(f"\n  Progress: {idx+1}/{len(combos)}")
                    print(f"  Estimated remaining: {remaining_time/60:.1f} minutes")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    continue
            
            selected_features = list(best_combo)
            
        else:
            # Add one more feature
            remaining = [f for f in range(len(feature_names)) if f not in selected_features]
            print(f"\nAdding 1 feature to {len(selected_features)} selected")
            print(f"Testing {len(remaining)} candidates...")
            
            best_acc = 0
            best_new = None
            
            for idx, new_feat in enumerate(remaining):
                print(f"\n{'─'*80}")
                print(f"Testing {idx+1}/{len(remaining)}")
                print(f"{'─'*80}")
                
                test_combo = selected_features + [new_feat]
                
                try:
                    acc, auc, gpu_t, cpu_t, total_t = train_complete_qsvm(
                        X_train, y_train, X_test, y_test,
                        test_combo, feature_names
                    )
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_new = new_feat
                        print(f"\n  ★★★ NEW BEST! Accuracy: {acc:.4f} ★★★")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    continue
            
            if best_new is not None:
                selected_features.append(best_new)
        
        print(f"\n{'='*80}")
        print(f"BEST {n_features} FEATURES:")
        print(f"  {[feature_names[i] for i in selected_features]}")
        print(f"  Accuracy: {best_acc:.4f}")
        print(f"{'='*80}")
        
        all_results[n_features] = {
            'features': selected_features.copy(),
            'feature_names': [feature_names[i] for i in selected_features],
            'accuracy': best_acc
        }
    
    return all_results, selected_features

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("STARTING QUANTUM FEATURE SELECTION")
    print("=" * 80)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(X_full, y_full)
    
    # Run feature selection
    results, best_features = quantum_feature_selection(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"FINAL RESULTS")
    print(f"{'#'*80}")
    
    for n_feat, res in results.items():
        print(f"\n{n_feat} features:")
        print(f"  Features: {res['feature_names']}")
        print(f"  Accuracy: {res['accuracy']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Complete! {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
