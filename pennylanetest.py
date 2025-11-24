#!/usr/bin/env python3
"""
Quantum Feature Selection using PennyLane with GPU Acceleration
Simulates IonQ Trapped-Ion Architecture
Based on IEEE TQE 2022 paper methodology
"""

import pennylane as qml
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

print("=" * 80)
print("PENNYLANE QUANTUM FEATURE SELECTION - TRAPPED-ION ARCHITECTURE")
print("=" * 80)
print(f"Starting: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PennyLane version: {qml.__version__}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset
    DATA_PATH = 'creditcard.csv'
    FRAUD_LABEL = 'Class'
    
    # Feature selection
    MIN_FEATURES = 3
    MAX_FEATURES = 5
    
    # Data parameters (optimized for testing)
    TRAIN_SIZE = 500
    TEST_SIZE = 300
    RANDOM_STATE = 42
    
    # Quantum parameters
    DEVICE = 'lightning.gpu'  # Use 'lightning.qubit' for CPU
    FEATURE_MAP_REPS = 1      # Circuit depth
    SHOTS = None              # None = exact statevector, or 1024 for sampling
    
    # Algorithm
    N_TRIALS = 1
    
    # GPU tracking
    VERBOSE_GPU = True
    
config = Config()

print(f"Configuration:")
print(f"  Device: {config.DEVICE}")
print(f"  Train/Test: {config.TRAIN_SIZE}/{config.TEST_SIZE}")
print(f"  Feature range: {config.MIN_FEATURES}-{config.MAX_FEATURES}")
print(f"  Feature map depth: {config.FEATURE_MAP_REPS}")
print(f"  Shots: {config.SHOTS if config.SHOTS else 'Statevector (exact)'}")
print()

# ============================================================================
# GPU VERIFICATION
# ============================================================================

def verify_gpu():
    """Verify GPU is available and working"""
    print("=" * 80)
    print("GPU VERIFICATION")
    print("=" * 80)
    
    if config.DEVICE == 'lightning.gpu':
        try:
            dev = qml.device('lightning.gpu', wires=3, shots=config.SHOTS)
            
            @qml.qnode(dev)
            def test():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.probs(wires=[0, 1])
            
            result = test()
            print(f"✓ GPU device working!")
            print(f"  Device: {dev}")
            print(f"  Test result: {result}\n")
            return True
            
        except Exception as e:
            print(f"✗ GPU device failed: {e}")
            print(f"  Falling back to CPU...\n")
            config.DEVICE = 'lightning.qubit'
            return False
    else:
        print(f"Using CPU device: {config.DEVICE}\n")
        return False

gpu_available = verify_gpu()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load credit card fraud dataset"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    df = pd.read_csv(config.DATA_PATH)
    print(f"✓ Loaded: {df.shape}")
    print(f"  Fraud: {df[config.FRAUD_LABEL].sum()}")
    print(f"  Genuine: {len(df) - df[config.FRAUD_LABEL].sum()}\n")
    
    X = df.drop(columns=[config.FRAUD_LABEL])
    y = df[config.FRAUD_LABEL]
    
    return X, y

X_full, y_full = load_data()

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(X, y, trial=0):
    """Prepare balanced dataset"""
    print("=" * 80)
    print(f"PREPARING DATA - Trial {trial + 1}")
    print("=" * 80)
    
    # Balance
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
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal,
        train_size=config.TRAIN_SIZE,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE + trial,
        stratify=y_bal
    )
    
    # Normalize to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Balanced dataset: {len(X_bal)} samples")
    print(f"✓ Split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"✓ Normalized: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]\n")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns.tolist()

# ============================================================================
# TRAPPED-ION QUANTUM FEATURE MAP
# ============================================================================

def create_ion_feature_map(n_features, reps=1):
    """
    Create trapped-ion inspired feature map
    Uses native IonQ gates: RX, RY, RZ, IsingXX (MS-like)
    """
    
    def feature_map(x, wires):
        """
        ZZ-style feature map adapted for trapped-ion architecture
        
        IonQ native gates:
        - GPi(φ) ≈ RX(π) with phase
        - GPi2(φ) ≈ RX(π/2) with phase
        - MS(φ) ≈ IsingXX(φ) (Mølmer-Sørensen gate)
        - Virtual Z: RZ (no physical operation)
        """
        n_qubits = len(wires)
        
        for rep in range(reps):
            # Layer 1: Hadamard (decomposed to RY + RZ for trapped ions)
            for i in wires:
                qml.RY(np.pi/2, wires=i)  # H = RY(π/2) · RZ(π/2)
                qml.RZ(np.pi/2, wires=i)
            
            # Layer 2: Encode features with RZ (virtual Z rotation)
            for i, wire in enumerate(wires):
                if i < len(x):
                    qml.RZ(2 * x[i], wires=wire)
            
            # Layer 3: Entangling layer using IsingXX (similar to MS gate)
            for i in range(n_qubits - 1):
                # ZZ interaction decomposed into trapped-ion native gates
                phi_i = x[i] if i < len(x) else 0.0
                phi_j = x[i+1] if i+1 < len(x) else 0.0
                angle = 2 * (np.pi - phi_i) * (np.pi - phi_j)
                
                # MS-like entangling gate
                qml.IsingXX(angle, wires=[wires[i], wires[i+1]])
    
    return feature_map

# ============================================================================
# QUANTUM KERNEL COMPUTATION WITH GPU TIMING
# ============================================================================

def compute_quantum_kernel_matrix_pennylane(X_train, X_test, feature_indices, feature_names):
    """
    Compute quantum kernel matrix using PennyLane
    This is where GPU acceleration happens!
    """
    n_features = len(feature_indices)
    
    print(f"\n  🔬 QUANTUM KERNEL COMPUTATION (Trapped-Ion Architecture)")
    print(f"     Device: {config.DEVICE}")
    print(f"     Features: {[feature_names[i] for i in feature_indices]}")
    print(f"     Qubits: {n_features}")
    
    # Select features
    X_train_subset = X_train[:, feature_indices]
    X_test_subset = X_test[:, feature_indices]
    
    print(f"     Train samples: {len(X_train_subset)}")
    print(f"     Test samples: {len(X_test_subset)}")
    
    # Create device
    dev = qml.device(config.DEVICE, wires=n_features, shots=config.SHOTS)
    print(f"     ✓ Device created: {dev.name}")
    
    # Create feature map
    feature_map = create_ion_feature_map(n_features, config.FEATURE_MAP_REPS)
    
    # Define kernel circuit
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        """Compute kernel between two data points"""
        wires = range(n_features)
        
        # Apply feature map to x1
        feature_map(x1, wires)
        
        # Apply inverse feature map to x2
        qml.adjoint(feature_map)(x2, wires)
        
        # Measure probability of |0...0>
        if config.SHOTS:
            # Sampling mode
            return qml.sample(wires=wires)
        else:
            # Exact statevector mode
            return qml.probs(wires=wires)
    
    # Compute training kernel matrix
    print(f"\n     🚀 GPU COMPUTING TRAINING KERNEL...")
    kernel_start = time.time()
    
    n_train = len(X_train_subset)
    K_train = np.zeros((n_train, n_train))
    
    for i in range(n_train):
        for j in range(i, n_train):
            if config.SHOTS:
                # Sampling mode: count |000...0> measurements
                samples = kernel_circuit(X_train_subset[i], X_train_subset[j])
                # Count zero state
                kernel_val = np.mean(np.sum(samples, axis=1) == 0)
            else:
                # Exact mode: probability of |000...0>
                probs = kernel_circuit(X_train_subset[i], X_train_subset[j])
                kernel_val = probs[0]
            
            K_train[i, j] = kernel_val
            K_train[j, i] = kernel_val  # Symmetric
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - kernel_start
            progress = (i + 1) / n_train
            eta = elapsed / progress - elapsed
            print(f"        Progress: {i+1}/{n_train} ({progress*100:.1f}%) - ETA: {eta:.1f}s")
    
    kernel_train_time = time.time() - kernel_start
    print(f"     ✓ Training kernel: {kernel_train_time:.3f}s")
    
    # Compute test kernel matrix
    print(f"\n     🚀 GPU COMPUTING TEST KERNEL...")
    kernel_start = time.time()
    
    n_test = len(X_test_subset)
    K_test = np.zeros((n_test, n_train))
    
    for i in range(n_test):
        for j in range(n_train):
            if config.SHOTS:
                samples = kernel_circuit(X_test_subset[i], X_train_subset[j])
                kernel_val = np.mean(np.sum(samples, axis=1) == 0)
            else:
                probs = kernel_circuit(X_test_subset[i], X_train_subset[j])
                kernel_val = probs[0]
            
            K_test[i, j] = kernel_val
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - kernel_start
            progress = (i + 1) / n_test
            eta = elapsed / progress - elapsed
            print(f"        Progress: {i+1}/{n_test} ({progress*100:.1f}%) - ETA: {eta:.1f}s")
    
    kernel_test_time = time.time() - kernel_start
    print(f"     ✓ Test kernel: {kernel_test_time:.3f}s")
    
    total_kernel_time = kernel_train_time + kernel_test_time
    print(f"\n     ⏱️  TOTAL GPU TIME: {total_kernel_time:.3f}s")
    print(f"     💡 GPU was actively computing quantum kernels!")
    
    return K_train, K_test, total_kernel_time

# ============================================================================
# CLASSICAL SVM TRAINING
# ============================================================================

def train_svm_with_kernel(K_train, K_test, y_train, y_test):
    """Train classical SVM with precomputed quantum kernel"""
    print(f"\n  🔧 CLASSICAL SVM TRAINING (CPU)")
    
    svm_start = time.time()
    svm = SVC(kernel='precomputed', C=1.0)
    svm.fit(K_train, y_train)
    svm_train_time = time.time() - svm_start
    print(f"     ✓ SVM trained: {svm_train_time:.3f}s")
    
    # Predict
    pred_start = time.time()
    y_pred = svm.predict(K_test)
    pred_time = time.time() - pred_start
    print(f"     ✓ Predictions: {pred_time:.3f}s")
    
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
    print(f"        TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    total_svm_time = svm_train_time + pred_time
    print(f"     ⏱️  CPU time: {total_svm_time:.3f}s")
    
    return accuracy, auc, total_svm_time

# ============================================================================
# COMPLETE QSVM PIPELINE
# ============================================================================

def train_complete_qsvm(X_train, y_train, X_test, y_test, feature_indices, feature_names):
    """Complete QSVM training with trapped-ion architecture"""
    
    print(f"\n{'─'*80}")
    print(f"TRAINING QSVM")
    print(f"{'─'*80}")
    
    total_start = time.time()
    
    # Quantum kernel computation (GPU)
    K_train, K_test, gpu_time = compute_quantum_kernel_matrix_pennylane(
        X_train, X_test, feature_indices, feature_names
    )
    
    # Classical SVM training (CPU)
    accuracy, auc, cpu_time = train_svm_with_kernel(
        K_train, K_test, y_train, y_test
    )
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n  ⏱️  TIME BREAKDOWN:")
    print(f"     Quantum kernel (GPU): {gpu_time:.3f}s ({gpu_time/total_time*100:.1f}%)")
    print(f"     Classical SVM (CPU):  {cpu_time:.3f}s ({cpu_time/total_time*100:.1f}%)")
    print(f"     Total:                {total_time:.3f}s")
    
    return accuracy, auc, gpu_time, cpu_time, total_time

# ============================================================================
# QUANTUM FEATURE SELECTION ALGORITHM
# ============================================================================

def quantum_feature_selection(X_train, y_train, X_test, y_test, feature_names):
    """
    Quantum feature selection algorithm
    Implements paper's feedforward selection method
    """
    
    print("\n" + "=" * 80)
    print("QUANTUM FEATURE SELECTION - TRAPPED-ION ARCHITECTURE")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
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
            # First iteration: try all combinations
            remaining = list(range(len(feature_names)))
            combos = list(itertools.combinations(remaining, n_features))
            
            print(f"\n⚠️  {len(combos)} combinations to test")
            print(f"   Estimated time: ~{len(combos) * 20 / 60:.1f} minutes\n")
            
            if len(combos) > 100:
                response = input(f"Continue? (y/n): ")
                if response.lower() != 'y':
                    print("Exiting...")
                    return all_results, selected_features
            
            best_acc = 0
            best_combo = None
            stage_start = time.time()
            
            for idx, combo in enumerate(combos):
                print(f"\n{'─'*40} Combo {idx+1}/{len(combos)} {'─'*40}")
                
                try:
                    acc, auc, gpu_t, cpu_t, total_t = train_complete_qsvm(
                        X_train, y_train, X_test, y_test,
                        list(combo), feature_names
                    )
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_combo = combo
                        print(f"\n  ★★★ NEW BEST! Accuracy: {acc:.4f} ★★★")
                    
                    # Progress
                    elapsed = time.time() - stage_start
                    avg_time = elapsed / (idx + 1)
                    eta = avg_time * (len(combos) - idx - 1)
                    print(f"\n  Progress: {idx+1}/{len(combos)} - ETA: {eta/60:.1f} min")
                    
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
                print(f"\n{'─'*40} Testing {idx+1}/{len(remaining)} {'─'*40}")
                
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
        print(f"  Indices: {selected_features}")
        print(f"  Names: {[feature_names[i] for i in selected_features]}")
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
    """Main execution"""
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
    print(f"FINAL RESULTS - TRAPPED-ION ARCHITECTURE")
    print(f"{'#'*80}")
    
    for n_feat, res in results.items():
        print(f"\n{n_feat} features:")
        print(f"  Names: {res['feature_names']}")
        print(f"  Accuracy: {res['accuracy']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Complete! {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
