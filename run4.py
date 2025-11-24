import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, 
                             recall_score, precision_score, f1_score)
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import seaborn as sns

# Qiskit imports - CORRECTED
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit.visualization import circuit_drawer

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CORRECTED: Quantum Feature Map Analysis for Fraud Detection")
print("="*80)

# ============================================================================
# PART 1: DATA PREPARATION (MATCHING YOUR EXACT SETUP)
# ============================================================================

class FraudDetectionDataset:
    """Handles dataset preparation - EXACT match to your configuration"""
    
    def __init__(self, csv_path, selected_features, random_state=42):
        """
        Args:
            csv_path: Path to creditcard.csv
            selected_features: List of column names (e.g., ['V14', 'V12', 'V4', 'V10', 'V8', 'V13', 'V7'])
            random_state: For reproducibility
        """
        self.df = pd.read_csv(csv_path)
        self.selected_features = selected_features
        self.random_state = random_state
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        print(f"\n✓ Dataset loaded: {len(self.df)} transactions")
        print(f"  Fraud cases: {self.df['Class'].sum()}")
        print(f"  Non-fraud cases: {len(self.df) - self.df['Class'].sum()}")
        print(f"  Selected features ({len(selected_features)}): {selected_features}")
    
    def prepare_balanced_dataset(self, n_fraud_train, n_nonfraud_train, 
                                  n_fraud_test, n_nonfraud_test):
        """
        Creates balanced train/test split - EXACT match to your setup
        
        Returns:
            X_train, X_test, y_train, y_test (all normalized to [-1, 1])
        """
        # Separate fraud and non-fraud
        fraud_df = self.df[self.df['Class'] == 1]
        nonfraud_df = self.df[self.df['Class'] == 0]
        
        # Sample training data
        fraud_train = fraud_df.sample(n=n_fraud_train, random_state=self.random_state)
        nonfraud_train = nonfraud_df.sample(n=n_nonfraud_train, random_state=self.random_state)
        
        # Sample test data (ensure no overlap with training)
        fraud_remaining = fraud_df.drop(fraud_train.index)
        nonfraud_remaining = nonfraud_df.drop(nonfraud_train.index)
        
        fraud_test = fraud_remaining.sample(n=n_fraud_test, random_state=self.random_state)
        nonfraud_test = nonfraud_remaining.sample(n=n_nonfraud_test, random_state=self.random_state)
        
        # Combine and shuffle
        train_df = pd.concat([fraud_train, nonfraud_train]).sample(frac=1, random_state=self.random_state)
        test_df = pd.concat([fraud_test, nonfraud_test]).sample(frac=1, random_state=self.random_state)
        
        # Extract ONLY selected features
        X_train = train_df[self.selected_features].values
        y_train = train_df['Class'].values
        X_test = test_df[self.selected_features].values
        y_test = test_df['Class'].values
        
        # Normalize to [-1, 1] for quantum encoding
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"\n✓ Dataset prepared:")
        print(f"  Train: {len(X_train)} samples (Fraud: {y_train.sum()}, Non-fraud: {len(y_train) - y_train.sum()})")
        print(f"  Test: {len(X_test)} samples (Fraud: {y_test.sum()}, Non-fraud: {len(y_test) - y_test.sum()})")
        print(f"  Feature range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        
        return X_train, X_test, y_train, y_test
    
    def get_example_samples(self, X, y):
        """Get one fraud and one non-fraud example for visualization"""
        fraud_idx = np.where(y == 1)[0][0]
        nonfraud_idx = np.where(y == 0)[0][0]
        
        return {
            'fraud': {'features': X[fraud_idx], 'label': y[fraud_idx], 'index': fraud_idx},
            'non_fraud': {'features': X[nonfraud_idx], 'label': y[nonfraud_idx], 'index': nonfraud_idx}
        }


# ============================================================================
# PART 2: QUANTUM ENCODING VISUALIZATION
# ============================================================================

class QuantumEncodingVisualizer:
    """Visualizes how classical data is encoded into quantum circuits"""
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.n_features = len(feature_names)
    
    def create_encoding_circuit(self, feature_vector, entanglement='linear', reps=1):
        """
        Creates a ZZFeatureMap circuit - EXACT match to your configuration
        
        Args:
            feature_vector: numpy array of shape (n_features,) with values in [-1, 1]
            entanglement: 'linear' or 'full'
            reps: number of repetitions
        
        Returns:
            QuantumCircuit with parameters bound
        """
        # Create feature map - MATCHING YOUR EXACT CONFIGURATION
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_features,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=True  # For clear visualization
        )
        
        # Bind the feature vector to the circuit
        circuit = feature_map.assign_parameters(feature_vector)
        
        return circuit, feature_map
    
    def visualize_encoding(self, examples_dict, entanglement='linear', reps=1, save_path=None):
        """
        Creates side-by-side circuit diagrams for fraud and non-fraud examples
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for idx, (label, data) in enumerate(examples_dict.items()):
            feature_vector = data['features']
            class_label = data['label']
            
            circuit, _ = self.create_encoding_circuit(feature_vector, entanglement, reps)
            
            # Draw circuit
            circuit_drawer(circuit, output='mpl', style='iqp', ax=axes[idx], fold=20)
            
            # Title with feature values
            feature_str = '\n'.join([f'{name}={val:.4f}' 
                                     for name, val in zip(self.feature_names, feature_vector)])
            axes[idx].set_title(
                f"{'FRAUD' if class_label == 1 else 'NON-FRAUD'} Transaction\n\n{feature_str}",
                fontsize=11, fontweight='bold', pad=20
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Encoding visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def print_encoding_explanation(self, examples_dict):
        """Prints detailed explanation of encoding process"""
        print("\n" + "="*80)
        print("QUANTUM ENCODING EXPLANATION")
        print("="*80)
        
        for label, data in examples_dict.items():
            feature_vector = data['features']
            class_label = data['label']
            
            print(f"\n{'FRAUD' if class_label == 1 else 'NON-FRAUD'} Transaction (Sample #{data['index']}):")
            print("-" * 80)
            
            for i, (name, value) in enumerate(zip(self.feature_names, feature_vector)):
                # ZZFeatureMap encodes as: H|0> -> RZ(2*value)|+>
                # Then applies ZZ interactions: RZ(2*(π-x_i)*(π-x_j)) between qubits
                angle_single = 2 * value  # Single-qubit rotation
                print(f"  Qubit {i} ({name:>6}): {value:>7.4f} → RZ({angle_single:>7.4f}) rotation")
            
            print(f"\n  Entanglement Pattern: {examples_dict}")
            print(f"  - Linear: Creates chain of CNOT gates (0→1, 1→2, 2→3, ...)")
            print(f"  - Full: Creates all-to-all CNOT gates (exponentially more complex)")
            print(f"  - ZZ Interaction: exp(i*(π-x_i)*(π-x_j)*Z_i⊗Z_j) between entangled qubits")
            print(f"\n  Final State: |ψ(x)⟩ = U_Φ(x)|0⟩^⊗{self.n_features}")


# ============================================================================
# PART 3: KERNEL COMPARISON FRAMEWORK (CORRECTED)
# ============================================================================

class KernelComparisonFramework:
    """Compares classical RBF kernel with quantum ZZ kernel - CORRECTED VERSION"""
    
    def __init__(self, n_features):
        self.n_features = n_features
    
    def compute_classical_kernel(self, X_train, X_test, gamma='scale'):
        """
        Computes RBF kernel matrices
        
        Args:
            X_train: Training data (n_train, n_features)
            X_test: Test data (n_test, n_features)
            gamma: RBF kernel parameter
        
        Returns:
            K_train: (n_train, n_train)
            K_test: (n_test, n_train)
            gamma: actual gamma value used
        """
        if gamma == 'scale':
            gamma = 1.0 / (self.n_features * X_train.var())
        
        print(f"\n{'='*80}")
        print(f"Computing Classical RBF Kernel")
        print(f"{'='*80}")
        print(f"  Gamma: {gamma:.6f}")
        print(f"  Formula: K(x, x') = exp(-gamma * ||x - x'||²)")
        
        K_train = rbf_kernel(X_train, X_train, gamma=gamma)
        K_test = rbf_kernel(X_test, X_train, gamma=gamma)
        
        print(f"  ✓ K_train shape: {K_train.shape}, range: [{K_train.min():.4f}, {K_train.max():.4f}]")
        print(f"  ✓ K_test shape: {K_test.shape}, range: [{K_test.min():.4f}, {K_test.max():.4f}]")
        
        return K_train, K_test, gamma
    
    def compute_quantum_kernel(self, X_train, X_test, entanglement='linear', reps=1, shots=1024):
        """
        Computes quantum kernel matrices using Qiskit - CORRECTED VERSION
        
        CRITICAL: Must match your exact configuration!
        """
        print(f"\n{'='*80}")
        print(f"Computing Quantum Kernel")
        print(f"{'='*80}")
        print(f"  Entanglement: {entanglement}")
        print(f"  Repetitions: {reps}")
        print(f"  Shots: {shots}")
        
        # Create feature map - EXACT MATCH TO YOUR CODE
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_features,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=False  # Not needed for kernel computation
        )
        
        print(f"  ✓ Feature map created: {self.n_features} qubits, depth={feature_map.depth()}")
        
        # Create simulator - CORRECTED: Use AerSimulator directly
        simulator = AerSimulator(method='statevector', device='CPU')
        
        # Create quantum kernel - EXACT MATCH TO YOUR CODE
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        # Compute kernels
        print(f"\n  Computing training kernel ({X_train.shape[0]}×{X_train.shape[0]})...")
        K_train = quantum_kernel.evaluate(x_vec=X_train)
        
        print(f"  Computing test kernel ({X_test.shape[0]}×{X_train.shape[0]})...")
        K_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
        
        print(f"  ✓ K_train shape: {K_train.shape}, range: [{K_train.min():.4f}, {K_train.max():.4f}]")
        print(f"  ✓ K_test shape: {K_test.shape}, range: [{K_test.min():.4f}, {K_test.max():.4f}]")
        
        return K_train, K_test, quantum_kernel
    
    def calculate_kernel_alignment(self, K1, K2):
        """
        Calculates centered kernel alignment between two kernels
        
        KA = <K1, K2>_F / (||K1||_F * ||K2||_F)
        
        Returns:
            alignment: float in [-1, 1]
                      1 = perfectly aligned (identical kernel structure)
                      0 = orthogonal (completely different)
                     -1 = anti-aligned (opposite patterns)
        """
        # Frobenius inner product
        numerator = np.sum(K1 * K2)
        
        # Frobenius norms
        norm_K1 = np.linalg.norm(K1, 'fro')
        norm_K2 = np.linalg.norm(K2, 'fro')
        
        alignment = numerator / (norm_K1 * norm_K2)
        
        print(f"\n{'='*80}")
        print(f"Kernel Alignment Analysis")
        print(f"{'='*80}")
        print(f"  Frobenius inner product: {numerator:.4f}")
        print(f"  ||K1||_F: {norm_K1:.4f}")
        print(f"  ||K2||_F: {norm_K2:.4f}")
        print(f"  Alignment: {alignment:.4f}")
        print(f"\n  Interpretation:")
        if alignment > 0.9:
            print(f"    ► Kernels are nearly identical (very high similarity)")
        elif alignment > 0.7:
            print(f"    ► Kernels are highly similar")
        elif alignment > 0.5:
            print(f"    ► Kernels have moderate similarity")
            print(f"    ► Good! Quantum captures SOME different patterns")
        elif alignment > 0.3:
            print(f"    ► Kernels have low similarity")
            print(f"    ► Excellent! Quantum captures DIFFERENT patterns")
        else:
            print(f"    ► Kernels are nearly orthogonal")
            print(f"    ► Outstanding! Completely different feature relationships")
        
        return alignment
    
    def visualize_kernel_comparison(self, K_classical, K_quantum, title_suffix="", save_path=None):
        """
        Creates side-by-side heatmaps of kernel matrices with proper interpretation
        
        WHAT THE HEATMAPS SHOW:
        - Bright spots (yellow) = High similarity between two samples
        - Dark spots (purple) = Low similarity between two samples
        - Diagonal is always brightest (sample compared to itself = 1.0)
        - Off-diagonal patterns show how the kernel groups similar transactions
        """
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        
        # Plot 1: Classical kernel
        sns.heatmap(K_classical, cmap='viridis', square=True, 
                    cbar_kws={'label': 'Kernel Value (Similarity)'}, 
                    ax=axes[0], vmin=0, vmax=1)
        axes[0].set_title(f'Classical RBF Kernel\n{title_suffix}\n\n' + 
                         'Bright = Similar transactions\nDark = Different transactions', 
                         fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Training Sample Index')
        axes[0].set_ylabel('Training Sample Index')
        
        # Plot 2: Quantum kernel
        sns.heatmap(K_quantum, cmap='viridis', square=True, 
                    cbar_kws={'label': 'Kernel Value (Similarity)'}, 
                    ax=axes[1], vmin=0, vmax=1)
        axes[1].set_title(f'Quantum ZZ Kernel\n{title_suffix}\n\n' + 
                         'Different pattern = Different feature relationships', 
                         fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Training Sample Index')
        axes[1].set_ylabel('Training Sample Index')
        
        # Plot 3: Difference (shows WHERE kernels disagree)
        diff = np.abs(K_classical - K_quantum)
        sns.heatmap(diff, cmap='hot', square=True, 
                    cbar_kws={'label': 'Absolute Difference'}, ax=axes[2])
        axes[2].set_title(f'Absolute Difference\n(Mean: {diff.mean():.4f}, Max: {diff.max():.4f})\n\n' +
                         'Hot spots = Quantum sees different patterns', 
                         fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Training Sample Index')
        axes[2].set_ylabel('Training Sample Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Kernel comparison saved to {save_path}")
        
        plt.show()
        
        # Print interpretation
        print(f"\n{'='*80}")
        print("HEATMAP INTERPRETATION GUIDE")
        print(f"{'='*80}")
        print("\n1. CLASSICAL & QUANTUM KERNELS (Left and Middle):")
        print("   - Each cell (i,j) shows similarity between sample i and sample j")
        print("   - Value of 1.0 (bright yellow) = identical/very similar")
        print("   - Value of 0.0 (dark purple) = very different")
        print("   - Diagonal is always brightest (sample vs itself)")
        print("\n2. DIFFERENCE MAP (Right):")
        print("   - Shows where quantum kernel disagrees with classical kernel")
        print("   - Hot spots (bright) = quantum sees different relationships")
        print("   - Dark spots = both kernels agree")
        print(f"   - Average difference: {diff.mean():.4f}")
        print(f"   - Maximum difference: {diff.max():.4f}")
        
        return fig


# ============================================================================
# PART 4: MODEL TRAINING AND EVALUATION
# ============================================================================

class ModelEvaluator:
    """Trains and evaluates all models with consistent metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculates all relevant metrics"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),  # Hit Rate
            'f1': f1_score(y_true, y_pred),
            'far': fp / tp if tp > 0 else np.inf,  # False Alarm Ratio
            'confusion_matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def print_metrics(model_name, metrics):
        """Pretty prints metrics"""
        print(f"\n{'='*80}")
        print(f"{model_name} PERFORMANCE METRICS")
        print(f"{'='*80}")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        if 'auc' in metrics:
            print(f"AUC:         {metrics['auc']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f} (Hit Rate - % of frauds caught)")
        print(f"F1-Score:    {metrics['f1']:.4f}")
        print(f"FAR:         {metrics['far']:.4f} (False alarms per true fraud)")
        print(f"\nConfusion Matrix:")
        print(f"                     Predicted")
        print(f"              Non-Fraud  Fraud")
        print(f"Actual Non-Fraud  {metrics['tn']:>4}    {metrics['fp']:>4}")
        print(f"       Fraud      {metrics['fn']:>4}    {metrics['tp']:>4}")
        print(f"\nInterpretation:")
        print(f"  ✓ Correctly caught {metrics['tp']} frauds")
        print(f"  ✗ Missed {metrics['fn']} frauds")
        print(f"  ⚠ {metrics['fp']} false alarms (genuine flagged as fraud)")
        print(f"  ✓ Correctly passed {metrics['tn']} genuine transactions")


# ============================================================================
# PART 5: MAIN EXECUTION SCRIPT (CORRECTED)
# ============================================================================

def main():
    # ========================================================================
    # CONFIGURATION - MATCH YOUR EXACT SETUP
    # ========================================================================
    CSV_PATH = 'creditcard.csv'
    SELECTED_FEATURES = ['V14', 'V12', 'V4', 'V10', 'V8', 'V13', 'V7']  # Your best 7 features
    RANDOM_STATE = 42
    
    # Training configuration - MATCH YOUR SETUP
    TRAIN_CONFIG = {
        'n_fraud_train': 100,
        'n_nonfraud_train': 100,
        'n_fraud_test': 50,
        'n_nonfraud_test': 50
    }
    
    print(f"\nConfiguration:")
    print(f"  Features: {SELECTED_FEATURES}")
    print(f"  Train: {TRAIN_CONFIG['n_fraud_train'] + TRAIN_CONFIG['n_nonfraud_train']} samples")
    print(f"  Test: {TRAIN_CONFIG['n_fraud_test'] + TRAIN_CONFIG['n_nonfraud_test']} samples")
    
    # ========================================================================
    # Initialize dataset handler
    # ========================================================================
    dataset = FraudDetectionDataset(CSV_PATH, SELECTED_FEATURES, RANDOM_STATE)
    
    # Prepare data
    X_train, X_test, y_train, y_test = dataset.prepare_balanced_dataset(**TRAIN_CONFIG)
    
    # ========================================================================
    # STEP 1: Visualize Quantum Encoding
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: QUANTUM ENCODING VISUALIZATION")
    print("="*80)
    
    visualizer = QuantumEncodingVisualizer(SELECTED_FEATURES)
    examples = dataset.get_example_samples(X_train, y_train)
    
    visualizer.print_encoding_explanation(examples)
    visualizer.visualize_encoding(examples, entanglement='linear', reps=1, 
                                   save_path='quantum_encoding_examples_corrected.png')
    
    # ========================================================================
    # STEP 2: Kernel Computation and Comparison
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: KERNEL COMPUTATION AND COMPARISON")
    print("="*80)
    
    kernel_framework = KernelComparisonFramework(n_features=len(SELECTED_FEATURES))
    
    # Compute classical kernel
    K_train_classical, K_test_classical, gamma = kernel_framework.compute_classical_kernel(
        X_train, X_test, gamma='scale'
    )
    
    # Compute quantum kernel (linear entanglement - superconducting)
    K_train_quantum_linear, K_test_quantum_linear, qkernel_linear = kernel_framework.compute_quantum_kernel(
        X_train, X_test, entanglement='linear', reps=1, shots=1024
    )
    
    # Calculate kernel alignment
    alignment_linear = kernel_framework.calculate_kernel_alignment(
        K_train_classical, K_train_quantum_linear
    )
    
    # Visualize kernel comparison
    kernel_framework.visualize_kernel_comparison(
        K_train_classical, K_train_quantum_linear,
        title_suffix=f"Alignment: {alignment_linear:.4f}",
        save_path='kernel_comparison_linear_corrected.png'
    )
    
    # ========================================================================
    # STEP 3: Train and Evaluate QSVM (Superconducting)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: QSVM TRAINING (SUPERCONDUCTING - LINEAR ENTANGLEMENT)")
    print("="*80)
    
    qsvm_linear = SVC(kernel='precomputed', random_state=RANDOM_STATE)
    qsvm_linear.fit(K_train_quantum_linear, y_train)
    
    # Get predictions
    y_pred_qsvm_linear = qsvm_linear.predict(K_test_quantum_linear)
    
    # For AUC, we need decision function (not probability for precomputed kernel)
    y_score_qsvm_linear = qsvm_linear.decision_function(K_test_quantum_linear)
    
    # Calculate metrics
    metrics_qsvm_linear = ModelEvaluator.calculate_metrics(
        y_test, y_pred_qsvm_linear, y_score_qsvm_linear
    )
    ModelEvaluator.print_metrics("QSVM (Superconducting - Linear Entanglement)", metrics_qsvm_linear)
    
    # ========================================================================
    # STEP 4: Train and Evaluate QSVM (Trapped-Ion)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: QSVM TRAINING (TRAPPED-ION - FULL ENTANGLEMENT)")
    print("="*80)
    
    # Compute quantum kernel with full entanglement
    K_train_quantum_full, K_test_quantum_full, qkernel_full = kernel_framework.compute_quantum_kernel(
        X_train, X_test, entanglement='full', reps=1, shots=1024
    )
    
    # Calculate alignment with classical kernel
    alignment_full = kernel_framework.calculate_kernel_alignment(
        K_train_classical, K_train_quantum_full
    )
    
    # Visualize
    kernel_framework.visualize_kernel_comparison(
        K_train_classical, K_train_quantum_full,
        title_suffix=f"Alignment: {alignment_full:.4f}",
        save_path='kernel_comparison_full_corrected.png'
    )
    
    # Train QSVM
    qsvm_full = SVC(kernel='precomputed', random_state=RANDOM_STATE)
    qsvm_full.fit(K_train_quantum_full, y_train)
    
    y_pred_qsvm_full = qsvm_full.predict(K_test_quantum_full)
    y_score_qsvm_full = qsvm_full.decision_function(K_test_quantum_full)
    
    metrics_qsvm_full = ModelEvaluator.calculate_metrics(
        y_test, y_pred_qsvm_full, y_score_qsvm_full
    )
    ModelEvaluator.print_metrics("QSVM (Trapped-Ion - Full Entanglement)", metrics_qsvm_full)
    
    # ========================================================================
    # STEP 5: Train and Evaluate Classical SVM (Fair Comparison)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: CLASSICAL SVM TRAINING (RBF KERNEL)")
    print("="*80)
    
    classical_svm = SVC(kernel='precomputed', random_state=RANDOM_STATE)
    
    # Fit on the precomputed classical kernel
    classical_svm.fit(K_train_classical, y_train)
    
    # Get predictions
    y_pred_classical = classical_svm.predict(K_test_classical)
    
    # Get decision function for AUC
    y_score_classical = classical_svm.decision_function(K_test_classical)
    
    # Calculate metrics
    metrics_classical = ModelEvaluator.calculate_metrics(
        y_test, y_pred_classical, y_score_classical
    )
    ModelEvaluator.print_metrics("Classical SVM (RBF Kernel)", metrics_classical)

    # ========================================================================
    # STEP 6: FINAL RESULTS COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: FINAL MODEL COMPARISON")
    print("="*80)
    
    results = {
        "Classical SVM (RBF)": metrics_classical,
        "QSVM (Linear Entanglement)": metrics_qsvm_linear,
        "QSVM (Full Entanglement)": metrics_qsvm_full
    }
    
    # Create a DataFrame for easy comparison
    df_results = pd.DataFrame(columns=[
        'Model', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 
        'Frauds Caught (TP)', 'False Alarms (FP)'
    ])
    
    for model_name, metrics in results.items():
        new_row = {
            'Model': model_name,
            'AUC': metrics.get('auc', np.nan),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Frauds Caught (TP)': metrics['tp'],
            'False Alarms (FP)': metrics['fp']
        }
        # Use pd.concat instead of deprecated append
        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)

    print("\nFinal Performance Summary:\n")
    print(df_results.to_string(index=False)) # .to_string() prints all columns nicely
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================
if __name__ == "__main__":
    import os
    # Check if the required data file exists before running
    if not os.path.exists('creditcard.csv'):
        print("\n" + "="*80)
        print("ERROR: 'creditcard.csv' not found.")
        print("Please download the 'creditcard.csv' dataset and")
        print("place it in the same directory as this script.")
        print("You can find it on Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("="*80 + "\n")
    else:
        main()
