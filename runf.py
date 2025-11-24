#!/usr/bin/env python3
"""
OPTIMIZED Quantum Feature Selection for Credit Card Fraud Detection
WITH Classical SVM Comparison, Kernel Alignment, and Circuit Visualization
"""

import numpy as np
import pandas as pd
import time
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.visualization import circuit_drawer

print("=" * 80)
print("QUANTUM FEATURE SELECTION WITH CLASSICAL COMPARISON")
print("=" * 80)
print(f"\nStarting: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset parameters
    DATA_PATH = 'creditcard.csv'
    FRAUD_LABEL = 'Class'
    
    # Feature pre-filter
    FEATURE_PREFILTER_LIST = ['V14', 'V12', 'V4', 'V10', 'V8', 'V13', 'V7']
    
    # Feature selection parameters
    MIN_FEATURES = 7
    MAX_FEATURES = 7
    
    # Dataset sizes
    TRAIN_SIZE = 400
    TEST_SIZE = 100
    RANDOM_STATE = 42
    
    # Quantum parameters
    FEATURE_MAP_TYPE = 'ZZ'
    FEATURE_MAP_REPS = 1
    ENTANGLEMENT = 'linear'
    
    # GPU parameters
    USE_GPU = False
    GPU_DEVICE = 'CPU'
    SHOTS = 1024
    
    # Algorithm parameters
    N_TRIALS = 1
    
    # NEW: Visualization parameters
    SAVE_CIRCUITS = True
    CIRCUIT_OUTPUT_DIR = './quantum_circuits/'
    SAVE_KERNEL_HEATMAPS = True
    KERNEL_HEATMAP_DIR = './kernel_heatmaps/'
    
config = Config()

# Create output directories
import os
if config.SAVE_CIRCUITS:
    os.makedirs(config.CIRCUIT_OUTPUT_DIR, exist_ok=True)
if config.SAVE_KERNEL_HEATMAPS:
    os.makedirs(config.KERNEL_HEATMAP_DIR, exist_ok=True)

print(f"⚡ Configuration:")
print(f"  Training size: {config.TRAIN_SIZE}")
print(f"  Test size: {config.TEST_SIZE}")
print(f"  Feature map: {config.FEATURE_MAP_TYPE}, depth: {config.FEATURE_MAP_REPS}")
print(f"  Entanglement: {config.ENTANGLEMENT}")
print(f"  Selected features: {config.FEATURE_PREFILTER_LIST}")
print(f"  Circuit output: {config.CIRCUIT_OUTPUT_DIR}")
print(f"  Kernel heatmaps: {config.KERNEL_HEATMAP_DIR}\n")

# ============================================================================
# GPU CHECK
# ============================================================================

if config.USE_GPU:
    print("=" * 80)
    print("GPU CHECK")
    print("=" * 80)
    try:
        simulator = AerSimulator()
        if 'GPU' not in simulator.available_devices():
            print("✗ GPU not available, using CPU")
            config.USE_GPU = False
            config.GPU_DEVICE = 'CPU'
        else:
            print("✓ GPU available")
    except:
        config.USE_GPU = False
        config.GPU_DEVICE = 'CPU'
else:
    print("Using CPU (GPU disabled in config)")
    config.GPU_DEVICE = 'CPU'

print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    try:
        df = pd.read_csv(config.DATA_PATH)
        print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if config.FRAUD_LABEL not in df.columns:
            print(f"!!! ERROR: '{config.FRAUD_LABEL}' not found!")
            return None, None
            
        print(f"  Fraud: {df[config.FRAUD_LABEL].sum()}")
        print(f"  Genuine: {len(df) - df[config.FRAUD_LABEL].sum()}")
        
        X = df.drop(columns=[config.FRAUD_LABEL])
        y = df[config.FRAUD_LABEL]
    
        return X, y
        
    except FileNotFoundError:
        print(f"!!! ERROR: File not found at '{config.DATA_PATH}'")
        return None, None

X_full, y_full = load_data()

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
    
    np.random.seed(config.RANDOM_STATE + trial)
    fraud_sample = np.random.choice(fraud_idx, n_fraud, replace=False)
    genuine_sample = np.random.choice(genuine_idx, n_genuine, replace=False)
    
    sample_idx = np.concatenate([fraud_sample, genuine_sample])
    np.random.shuffle(sample_idx)
    
    X_bal = X.loc[sample_idx]
    y_bal = y.loc[sample_idx]
    
    print(f"✓ Balanced: {len(X_bal)} samples ({n_fraud} fraud, {n_genuine} genuine)")
    
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
# NEW: VISUALIZATION FUNCTIONS
# ============================================================================

def print_sample_transactions(X_train, y_train, feature_names):
    """Print example fraud and non-fraud transactions for paper"""
    print("\n" + "=" * 80)
    print("SAMPLE TRANSACTIONS FOR PAPER")
    print("=" * 80)
    
    # Find first fraud and non-fraud
    fraud_idx = np.where(y_train == 1)[0][0]
    nonfraud_idx = np.where(y_train == 0)[0][0]
    
    print(f"\n📊 FRAUD TRANSACTION (Training Sample #{fraud_idx}):")
    print("─" * 80)
    print(f"{'Feature':<15} {'Normalized Value':<20} {'Qubit Rotation (rad)':<20}")
    print("─" * 80)
    for i, (feat, val) in enumerate(zip(feature_names, X_train[fraud_idx])):
        # For ZZFeatureMap, single-qubit rotations are P(2*x_i)
        # Note: The '2*' is often part of the definition. Check your specific feature map.
        # If using standard ZFeatureMap, rotation is just val.
        # For ZZFeatureMap, it's typically 2 * val.
        rotation = 2 * val 
        print(f"{feat:<15} {val:>18.6f}    {rotation:>18.6f}")
    
    print(f"\n📊 NON-FRAUD TRANSACTION (Training Sample #{nonfraud_idx}):")
    print("─" * 80)
    print(f"{'Feature':<15} {'Normalized Value':<20} {'Qubit Rotation (rad)':<20}")
    print("─" * 80)
    for i, (feat, val) in enumerate(zip(feature_names, X_train[nonfraud_idx])):
        rotation = 2 * val
        print(f"{feat:<15} {val:>18.6f}    {rotation:>18.6f}")
    
    print("\n" + "=" * 80)
    
    return fraud_idx, nonfraud_idx

def save_quantum_circuit(X_train, y_train, feature_names, fraud_idx, nonfraud_idx):
    """Save quantum circuit diagrams for paper"""
    print("\n" + "=" * 80)
    print("GENERATING QUANTUM CIRCUIT DIAGRAMS")
    print("=" * 80)
    
    n_features = len(feature_names)
    
    # Create feature map
    if config.FEATURE_MAP_TYPE == 'ZZ':
        feature_map = ZZFeatureMap(
            feature_dimension=n_features,
            reps=config.FEATURE_MAP_REPS,
            entanglement=config.ENTANGLEMENT,
            insert_barriers=True
        )
    else:
        feature_map = ZFeatureMap(
            feature_dimension=n_features,
            reps=config.FEATURE_MAP_REPS
        )
    
    print(f"✓ Feature map: {n_features} qubits, depth={feature_map.depth()}")
    
    # Fraud transaction circuit
    fraud_circuit = feature_map.assign_parameters(X_train[fraud_idx])
    fraud_path = os.path.join(config.CIRCUIT_OUTPUT_DIR, 'fraud_transaction_circuit.png')
    
    # 
    fig = circuit_drawer(fraud_circuit, output='mpl', style='iqp', fold=20)
    plt.title(f"Fraud Transaction Quantum Encoding Circuit\nFeatures: {feature_names}", 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(fraud_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved fraud circuit: {fraud_path}")
    
    # Non-fraud transaction circuit
    nonfraud_circuit = feature_map.assign_parameters(X_train[nonfraud_idx])
    nonfraud_path = os.path.join(config.CIRCUIT_OUTPUT_DIR, 'nonfraud_transaction_circuit.png')
    
    fig = circuit_drawer(nonfraud_circuit, output='mpl', style='iqp', fold=20)
    plt.title(f"Non-Fraud Transaction Quantum Encoding Circuit\nFeatures: {feature_names}", 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(nonfraud_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved non-fraud circuit: {nonfraud_path}")
    
    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    
    circuit_drawer(fraud_circuit, output='mpl', style='iqp', ax=axes[0], fold=20)
    axes[0].set_title(f"FRAUD Transaction\n{feature_names}", fontsize=11, fontweight='bold')
    
    circuit_drawer(nonfraud_circuit, output='mpl', style='iqp', ax=axes[1], fold=20)
    axes[1].set_title(f"NON-FRAUD Transaction\n{feature_names}", fontsize=11, fontweight='bold')
    
    comparison_path = os.path.join(config.CIRCUIT_OUTPUT_DIR, 'circuit_comparison.png')
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison: {comparison_path}")

def calculate_kernel_alignment(K1, K2, name1="Kernel 1", name2="Kernel 2"):
    """Calculate and print kernel alignment score"""
    # This is the "Frobenius" or "Hilbert-Schmidt" alignment
    numerator = np.sum(K1 * K2)
    norm_K1 = np.linalg.norm(K1, 'fro')
    norm_K2 = np.linalg.norm(K2, 'fro')
    
    # Ensure no division by zero
    if norm_K1 == 0 or norm_K2 == 0:
        alignment = 0.0
    else:
        alignment = numerator / (norm_K1 * norm_K2)
    
    print(f"\n{'='*80}")
    print(f"KERNEL ALIGNMENT: {name1} vs {name2}")
    print(f"{'='*80}")
    print(f"  Alignment Score: {alignment:.4f}")
    print(f"  Frobenius <K1,K2>: {numerator:.4f}")
    print(f"  ||K1||_F: {norm_K1:.4f}")
    print(f"  ||K2||_F: {norm_K2:.4f}")
    print(f"\n  Interpretation:")
    if alignment > 0.9:
        print(f"    ► Very high similarity (>0.9) - kernels are nearly identical")
    elif alignment > 0.7:
        print(f"    ► High similarity (0.7-0.9) - similar patterns")
    elif alignment > 0.5:
        print(f"    ► Moderate similarity (0.5-0.7) - quantum captures some different patterns ✓")
    elif alignment > 0.3:
        print(f"    ► Low similarity (0.3-0.5) - quantum captures different patterns ✓✓")
    else:
        print(f"    ► Very low similarity (<0.3) - completely different feature space ✓✓✓")
    
    return alignment

def visualize_kernel_heatmaps(K_classical, K_quantum, feature_names, alignment):
    """Create and save kernel heatmap comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # 
    # Classical kernel
    sns.heatmap(K_classical, cmap='viridis', square=True, 
                cbar_kws={'label': 'Similarity'}, ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title(f'Classical RBF Kernel\n{len(feature_names)} features: {feature_names}', 
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Training Sample Index')
    axes[0].set_ylabel('Training Sample Index')
    
    # 
    # Quantum kernel
    sns.heatmap(K_quantum, cmap='viridis', square=True, 
                cbar_kws={'label': 'Similarity'}, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f'Quantum ZZ Kernel ({config.ENTANGLEMENT} entanglement)\n' + 
                      f'Alignment: {alignment:.4f}', 
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Training Sample Index')
    axes[1].set_ylabel('Training Sample Index')
    
    # Difference
    diff = np.abs(K_classical - K_quantum)
    sns.heatmap(diff, cmap='hot', square=True, 
                cbar_kws={'label': 'Absolute Difference'}, ax=axes[2])
    axes[2].set_title(f'Absolute Difference\n' + 
                      f'Mean: {diff.mean():.4f}, Max: {diff.max():.4f}', 
                      fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Training Sample Index')
    axes[2].set_ylabel('Training Sample Index')
    
    plt.tight_layout()
    
    save_path = os.path.join(config.KERNEL_HEATMAP_DIR, 
                             f'kernel_comparison_{len(feature_names)}features.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved kernel heatmap: {save_path}")

# ============================================================================
# CLASSICAL SVM WITH RBF KERNEL
# ============================================================================

def compute_classical_kernel(X_train, X_test, n_features):
    """Compute classical RBF kernel"""
    # Standard gamma heuristic: 1 / (n_features * X.var())
    gamma = 1.0 / (n_features * X_train.var())
    
    print(f"\n  Computing Classical RBF Kernel...")
    print(f"    Gamma: {gamma:.6f}")
    
    K_train = rbf_kernel(X_train, X_train, gamma=gamma)
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)
    
    print(f"    ✓ K_train: {K_train.shape}, range [{K_train.min():.4f}, {K_train.max():.4f}]")
    print(f"    ✓ K_test: {K_test.shape}, range [{K_test.min():.4f}, {K_test.max():.4f}]")
    
    return K_train, K_test, gamma

def train_classical_svm(K_train, K_test, y_train, y_test):
    """Train classical SVM with precomputed RBF kernel"""
    print(f"\n  Training Classical SVM...")
    
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    
    y_pred = svm.predict(K_test)
    y_score = svm.decision_function(K_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_ratio = fp / tp if (tp + fn) > 0 else np.inf # Note: Check this definition
    
    print(f"\n    📈 Classical SVM Results:")
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      AUC: {auc:.4f}")
    print(f"      Hit Rate: {hit_rate:.4f} (TP / (TP+FN))")
    print(f"      FAR: {false_alarm_ratio:.4f} (FP / TP)")
    print(f"      Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return accuracy, auc, hit_rate, false_alarm_ratio

# ============================================================================
# QUANTUM KERNEL
# ============================================================================

def compute_quantum_kernel_matrix(X_train, X_test, feature_indices, feature_names):
    """Compute quantum kernel matrix"""
    n_features = len(feature_indices)
    
    print(f"\n  📊 Computing Quantum Kernel Matrix")
    print(f"        Features: {[feature_names[i] for i in feature_indices]}")
    print(f"        Train: {len(X_train)}, Test: {len(X_test)}")
    
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
    
    print(f"        ✓ Feature map: {n_features} qubits, depth={feature_map.depth()}")
    
    # Create simulator
    if config.USE_GPU:
        simulator = AerSimulator(method='statevector', device='GPU')
        print(f"        ✓ Using GPU")
    else:
        simulator = AerSimulator(method='statevector', device='CPU')
        print(f"        ✓ Using CPU")
    
    # Create quantum kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # Compute kernels
    print(f"\n        🚀 Computing kernels...")
    
    kernel_start = time.time()
    K_train = quantum_kernel.evaluate(X_train_subset, X_train_subset)
    kernel_train_time = time.time() - kernel_start
    print(f"          ✓ Training kernel: {kernel_train_time:.3f}s")
    
    kernel_start = time.time()
    K_test = quantum_kernel.evaluate(X_test_subset, X_train_subset)
    kernel_test_time = time.time() - kernel_start
    print(f"          ✓ Test kernel: {kernel_test_time:.3f}s")
    
    total_kernel_time = kernel_train_time + kernel_test_time
    print(f"\n        ⏱️  TOTAL TIME: {total_kernel_time:.3f}s")
    
    return K_train, K_test, total_kernel_time

# ============================================================================
# TRAIN QSVM
# ============================================================================

def train_qsvm_with_kernel(K_train, K_test, y_train, y_test):
    """Train SVM with precomputed quantum kernel"""
    print(f"\n  🔧 Training Quantum SVM")
    
    svm_start = time.time()
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    svm_train_time = time.time() - svm_start
    print(f"        ✓ Trained in {svm_train_time:.3f}s")
    
    pred_start = time.time()
    y_pred = svm.predict(K_test)
    pred_time = time.time() - pred_start
    print(f"        ✓ Predicted in {pred_time:.3f}s")
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        y_score = svm.decision_function(K_test)
        auc = roc_auc_score(y_test, y_score)
    except:
        auc = 0.0
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_ratio = fp / tp if (tp + fn) > 0 else np.inf
    
    print(f"\n        📈 Quantum SVM Results:")
    print(f"          Accuracy: {accuracy:.4f}")
    print(f"          AUC: {auc:.4f}")
    print(f"          Hit Rate: {hit_rate:.4f} (TP / (TP+FN))")
    print(f"          FAR: {false_alarm_ratio:.4f} (FP / TP)")
    print(f"          Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    total_svm_time = svm_train_time + pred_time
    print(f"\n        ⏱️  TOTAL CPU TIME: {total_svm_time:.3f}s")
    
    return accuracy, auc, hit_rate, false_alarm_ratio, total_svm_time

# ============================================================================
# COMPLETE TRAINING PIPELINE WITH COMPARISON
# ============================================================================

def train_complete_with_comparison(X_train, y_train, X_test, y_test, 
                                   feature_indices, feature_names, 
                                   is_first_run=False):
    """Complete training with classical vs quantum comparison"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING WITH CLASSICAL-QUANTUM COMPARISON")
    print(f"{'='*80}")
    
    total_start = time.time()
    n_features = len(feature_indices)
    
    # Select feature subset
    X_train_subset = X_train[:, feature_indices]
    X_test_subset = X_test[:, feature_indices]
    selected_feature_names = [feature_names[i] for i in feature_indices]
    
    print(f"Selected {n_features} features: {selected_feature_names}")
    
    # === ONLY ON FIRST RUN: Print samples and save circuits ===
    # This block fulfills your requests for sample values and circuit PNGs
    if is_first_run:
        fraud_idx, nonfraud_idx = print_sample_transactions(
            X_train_subset, y_train, selected_feature_names
        )
        
        if config.SAVE_CIRCUITS:
            save_quantum_circuit(X_train_subset, y_train, selected_feature_names, 
                                 fraud_idx, nonfraud_idx)
    
    # === CLASSICAL SVM ===
    # This block fulfills your request to train a classical SVM
    print(f"\n{'─'*80}")
    print("CLASSICAL SVM (RBF Kernel)")
    print(f"{'─'*80}")
    
    K_train_classical, K_test_classical, gamma = compute_classical_kernel(
        X_train_subset, X_test_subset, n_features
    )
    
    acc_classical, auc_classical, hr_classical, far_classical = train_classical_svm(
        K_train_classical, K_test_classical, y_train, y_test
    )
    
    # === QUANTUM SVM ===
    print(f"\n{'─'*80}")
    print("QUANTUM SVM (ZZ Kernel)")
    print(f"{'─'*80}")
    
    K_train_quantum, K_test_quantum, gpu_time = compute_quantum_kernel_matrix(
        X_train, X_test, feature_indices, feature_names
    )
    
    acc_quantum, auc_quantum, hr_quantum, far_quantum, cpu_time = train_qsvm_with_kernel(
        K_train_quantum, K_test_quantum, y_train, y_test
    )
    
    # === KERNEL ALIGNMENT ===
    # This block fulfills your request for the Kernel Alignment score
    alignment = calculate_kernel_alignment(
        K_train_classical, K_train_quantum,
        "Classical RBF", "Quantum ZZ"
    )
    
    # === VISUALIZATION ===
    if config.SAVE_KERNEL_HEATMAPS and is_first_run:
        visualize_kernel_heatmaps(
            K_train_classical, K_train_quantum, 
            selected_feature_names, alignment
        )
    
    total_time = time.time() - total_start
    
    # === COMPARISON SUMMARY ===
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Model':<20} {'Accuracy':<12} {'AUC':<12} {'Hit Rate':<12} {'FAR':<12}")
    print(f"{'─'*68}")
    print(f"{'Classical RBF':<20} {acc_classical:<12.4f} {auc_classical:<12.4f} "
          f"{hr_classical:<12.4f} {far_classical:<12.4f}")
    print(f"{'Quantum ZZ':<20} {acc_quantum:<12.4f} {auc_quantum:<12.4f} "
          f"{hr_quantum:<12.4f} {far_quantum:<12.4f}")
    print(f"{'─'*68}")
    print(f"\nKernel Alignment: {alignment:.4f}")
    print(f"Total Time: {total_time:.3f}s")
    print(f"{'='*80}")
    
    return acc_quantum, auc_quantum, hr_quantum, far_quantum, gpu_time, cpu_time, total_time

# ============================================================================
# FEATURE SELECTION ALGORITHM (COMPLETED)
# ============================================================================

def quantum_feature_selection(X_train, y_train, X_test, y_test, feature_names):
    """Quantum feature selection with classical comparison"""
    
    print("\n" + "=" * 80)
    print("QUANTUM FEATURE SELECTION ALGORITHM")
    print("=" * 80)
    print(f"Device: {config.GPU_DEVICE}")
    print(f"Full feature list: {len(feature_names)} features")
    print(f"Selection range: {config.MIN_FEATURES}-{config.MAX_FEATURES}")
    
    # Determine which features to test
    if config.FEATURE_PREFILTER_LIST:
        print(f"Using pre-filter list: {config.FEATURE_PREFILTER_LIST}")
        try:
            feature_indices_to_test = [feature_names.index(name) 
                                       for name in config.FEATURE_PREFILTER_LIST]
            print(f"Indices to test: {feature_indices_to_test}")
        except ValueError as e:
            print(f"\n!!! ERROR: Feature not found: {e}")
            return {}, []
    else:
        print("No pre-filter list. Using ALL features.")
        feature_indices_to_test = list(range(len(feature_names)))
        
    print(f"Total features in pool: {len(feature_indices_to_test)}")
    print("=" * 80)
    
    all_results = {}
    selected_features_name = ""
    
    # This flag ensures we only print samples/circuits on the very first run
    is_first_run = True 
    
    # Your config (MIN=7, MAX=7) means this loop only runs once for n_features = 7
    for n_features in range(config.MIN_FEATURES, config.MAX_FEATURES + 1):
        print(f"\n{'='*80}")
        print(f"TESTING {n_features} FEATURES")
        print(f"{'='*80}")
        
        # This logic handles your specific config:
        # It checks if the number of features to test (n_features=7)
        # matches the number of features in your prefilter list (7)
        if n_features == len(feature_indices_to_test):
            combo = tuple(feature_indices_to_test)
            combo_name = str(tuple(feature_names[i] for i in combo))
            
            print(f"Running test for pre-configured combination: {combo_name}")

            # Call the main comparison function
            # This function will trigger all your requests
            acc, auc, hr, far, gpu_time, cpu_time, total_time = \
                train_complete_with_comparison(
                    X_train, y_train, X_test, y_test,
                    combo, feature_names,
                    is_first_run=is_first_run
                )
            
            # We only want to print samples/save circuits on the very first run
            is_first_run = False 
            
            all_results[combo_name] = {
                'auc': auc, 'acc': acc, 'hr': hr, 'far': far,
                'gpu_time': gpu_time, 'cpu_time': cpu_time, 
                'total_time': total_time
            }
            
            # Since this is the only combo, it's the "best" one
            selected_features_name = combo_name
        
        else:
            # This part would normally do SFS, but your config skips it.
            print(f"Skipping {n_features} features (config set to run only for "
                  f"{len(feature_indices_to_test)} features)")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    if selected_features_name in all_results:
        print(f"Best feature set: {selected_features_name}")
        print(f"Metrics: {all_results[selected_features_name]}")
    else:
        print("No results generated. Check your configuration.")
        
    return all_results, selected_features_name

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    if X_full is None or y_full is None:
        print("Exiting due to data loading error.")
        return

    print(f"\nStarting {config.N_TRIALS} trial(s)...")
    all_trial_results = []
    
    start_time = time.time()
    
    for trial in range(config.N_TRIALS):
        # 1. Prepare data for this trial
        X_train_scaled, X_test_scaled, y_train, y_test, all_feature_names = \
            prepare_data(X_full, y_full, trial=trial)
        
        # 2. Run the feature selection / comparison
        #    (With current config, this just runs the one 7-feature set)
        results, best_set = quantum_feature_selection(
            X_train_scaled, y_train, X_test_scaled, y_test, all_feature_names
        )
        
        all_trial_results.append({'trial': trial, 'best_set': best_set, 'results': results})
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print("ALL TRIALS COMPLETE")
    print("=" * 80)
    print(f"Total execution time: {end_time - start_time:.3f}s")
    print("\nFinal Results Summary:")
    print(all_trial_results)

if __name__ == "__main__":
    main()
