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

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit.visualization import circuit_drawer

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA PREPARATION
# ============================================================================

class FraudDetectionDataset:
    """Handles dataset preparation with configurable sampling"""
    
    def __init__(self, csv_path, selected_features, random_state=42):
        """
        Args:
            csv_path: Path to creditcard.csv
            selected_features: List of column names to use (e.g., ['V14', 'V12', 'V4', 'V10', 'V8', 'V13'])
            random_state: For reproducibility
        """
        self.df = pd.read_csv(csv_path)
        self.selected_features = selected_features
        self.random_state = random_state
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        print(f"Dataset loaded: {len(self.df)} transactions")
        print(f"Fraud cases: {self.df['Class'].sum()}")
        print(f"Selected features: {selected_features}")
    
    def prepare_balanced_dataset(self, n_fraud_train, n_nonfraud_train, 
                                  n_fraud_test, n_nonfraud_test):
        """
        Creates balanced train/test split
        
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
        
        # Combine
        train_df = pd.concat([fraud_train, nonfraud_train]).sample(frac=1, random_state=self.random_state)
        test_df = pd.concat([fraud_test, nonfraud_test]).sample(frac=1, random_state=self.random_state)
        
        # Extract features
        X_train = train_df[self.selected_features].values
        y_train = train_df['Class'].values
        X_test = test_df[self.selected_features].values
        y_test = test_df['Class'].values
        
        # Normalize to [-1, 1] for quantum encoding
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"\nDataset prepared:")
        print(f"  Train: {len(X_train)} samples (Fraud: {y_train.sum()}, Non-fraud: {len(y_train) - y_train.sum()})")
        print(f"  Test: {len(X_test)} samples (Fraud: {y_test.sum()}, Non-fraud: {len(y_test) - y_test.sum()})")
        
        return X_train, X_test, y_train, y_test
    
    def get_example_samples(self, X, y):
        """Get one fraud and one non-fraud example for visualization"""
        fraud_idx = np.where(y == 1)[0][0]
        nonfraud_idx = np.where(y == 0)[0][0]
        
        return {
            'fraud': {'features': X[fraud_idx], 'label': y[fraud_idx]},
            'non_fraud': {'features': X[nonfraud_idx], 'label': y[nonfraud_idx]}
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
        Creates a ZZFeatureMap circuit for a given feature vector
        
        Args:
            feature_vector: numpy array of shape (n_features,) with values in [-1, 1]
            entanglement: 'linear' or 'full'
            reps: number of repetitions
        
        Returns:
            QuantumCircuit with parameters bound
        """
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_features,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=True
        )
        
        # Bind the feature vector to the circuit
        circuit = feature_map.assign_parameters(feature_vector)
        
        return circuit
    
    def visualize_encoding(self, examples_dict, entanglement='linear', reps=1, save_path=None):
        """
        Creates side-by-side circuit diagrams for fraud and non-fraud examples
        
        Args:
            examples_dict: Output from get_example_samples()
            entanglement: 'linear' or 'full'
            reps: number of repetitions
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        
        for idx, (label, data) in enumerate(examples_dict.items()):
            feature_vector = data['features']
            class_label = data['label']
            
            circuit = self.create_encoding_circuit(feature_vector, entanglement, reps)
            
            # Draw circuit
            circuit_drawer(circuit, output='mpl', style='iqp', ax=axes[idx])
            
            # Title with feature values
            feature_str = ', '.join([f'{name}={val:.3f}' 
                                     for name, val in zip(self.feature_names, feature_vector)])
            axes[idx].set_title(
                f"{'FRAUD' if class_label == 1 else 'NON-FRAUD'} Transaction\n{feature_str}",
                fontsize=12, fontweight='bold', pad=20
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Encoding visualization saved to {save_path}")
        
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
            
            print(f"\n{'FRAUD' if class_label == 1 else 'NON-FRAUD'} Transaction Encoding:")
            print("-" * 40)
            
            for i, (name, value) in enumerate(zip(self.feature_names, feature_vector)):
                # ZZFeatureMap encodes as: H|0> -> RZ(2*value)|+>
                angle = 2 * np.pi * value  # Normalized to [0, 2π] from [-1, 1]
                print(f"  Qubit {i} ({name:>6}): {value:>7.4f} → RZ({angle:>7.4f} rad)")
            
            print(f"\n  Entanglement: Creates 2-qubit interactions via CNOT gates")
            print(f"  State: |ψ⟩ = U_Φ(x)|0⟩^⊗{self.n_features}")


# ============================================================================
# PART 3: KERNEL COMPARISON FRAMEWORK
# ============================================================================

class KernelComparisonFramework:
    """Compares classical RBF kernel with quantum ZZ kernel"""
    
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
        """
        if gamma == 'scale':
            gamma = 1.0 / (self.n_features * X_train.var())
        
        print(f"\nComputing Classical RBF Kernel (gamma={gamma:.6f})...")
        
        K_train = rbf_kernel(X_train, X_train, gamma=gamma)
        K_test = rbf_kernel(X_test, X_train, gamma=gamma)
        
        print(f"  K_train shape: {K_train.shape}")
        print(f"  K_test shape: {K_test.shape}")
        
        return K_train, K_test, gamma
    
    def compute_quantum_kernel(self, X_train, X_test, entanglement='linear', reps=1, shots=1024):
        """
        Computes quantum kernel matrices using Qiskit
        
        Args:
            X_train: Training data (n_train, n_features)
            X_test: Test data (n_test, n_features)
            entanglement: 'linear' or 'full'
            reps: number of repetitions
            shots: number of shots for sampling
        
        Returns:
            K_train: (n_train, n_train)
            K_test: (n_test, n_train)
        """
        print(f"\nComputing Quantum Kernel (entanglement={entanglement}, reps={reps}, shots={shots})...")
        
        # Create feature map
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_features,
            reps=reps,
            entanglement=entanglement
        )
        
        # Create quantum kernel
        backend = Aer.get_backend('qasm_simulator')
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        # Compute kernels
        K_train = quantum_kernel.evaluate(x_vec=X_train)
        K_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
        
        print(f"  K_train shape: {K_train.shape}")
        print(f"  K_test shape: {K_test.shape}")
        
        return K_train, K_test, quantum_kernel
    
    def calculate_kernel_alignment(self, K1, K2):
        """
        Calculates centered kernel alignment between two kernels
        
        KA = <K1, K2>_F / (||K1||_F * ||K2||_F)
        
        Returns:
            alignment: float in [0, 1], where 1 = identical, 0 = orthogonal
        """
        # Frobenius inner product
        numerator = np.sum(K1 * K2)
        
        # Frobenius norms
        norm_K1 = np.linalg.norm(K1, 'fro')
        norm_K2 = np.linalg.norm(K2, 'fro')
        
        alignment = numerator / (norm_K1 * norm_K2)
        
        return alignment
    
    def visualize_kernel_comparison(self, K_classical, K_quantum, title_suffix="", save_path=None):
        """
        Creates side-by-side heatmaps of kernel matrices
        
        Args:
            K_classical: Classical RBF kernel matrix
            K_quantum: Quantum kernel matrix
            title_suffix: Additional text for title
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Classical kernel
        sns.heatmap(K_classical, cmap='viridis', square=True, 
                    cbar_kws={'label': 'Similarity'}, ax=axes[0], vmin=0, vmax=1)
        axes[0].set_title(f'Classical RBF Kernel\n{title_suffix}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Training Sample Index')
        axes[0].set_ylabel('Training Sample Index')
        
        # Plot 2: Quantum kernel
        sns.heatmap(K_quantum, cmap='viridis', square=True, 
                    cbar_kws={'label': 'Similarity'}, ax=axes[1], vmin=0, vmax=1)
        axes[1].set_title(f'Quantum ZZ Kernel\n{title_suffix}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Training Sample Index')
        axes[1].set_ylabel('Training Sample Index')
        
        # Plot 3: Difference
        diff = np.abs(K_classical - K_quantum)
        sns.heatmap(diff, cmap='hot', square=True, 
                    cbar_kws={'label': 'Absolute Difference'}, ax=axes[2])
        axes[2].set_title(f'Absolute Difference\n(Mean: {diff.mean():.4f})', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Training Sample Index')
        axes[2].set_ylabel('Training Sample Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Kernel comparison saved to {save_path}")
        
        plt.show()
        
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
        print(f"\n{'='*60}")
        print(f"{model_name} Performance")
        print(f"{'='*60}")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        if 'auc' in metrics:
            print(f"AUC:         {metrics['auc']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f} (Hit Rate)")
        print(f"F1-Score:    {metrics['f1']:.4f}")
        print(f"FAR:         {metrics['far']:.4f} (FP/TP)")
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Non-Fraud  Fraud")
        print(f"Actual Non-Fraud  {metrics['tn']:>4}    {metrics['fp']:>4}")
        print(f"       Fraud      {metrics['fn']:>4}    {metrics['tp']:>4}")


# ============================================================================
# PART 5: MAIN EXECUTION SCRIPT
# ============================================================================

def main():
    # Configuration
    CSV_PATH = 'creditcard.csv'  # Update this path
    SELECTED_FEATURES = ['V14', 'V12', 'V4', 'V10', 'V8', 'V13', 'V7']  # Your best 6 features
    
    # Initialize dataset handler
    dataset = FraudDetectionDataset(CSV_PATH, SELECTED_FEATURES)
    
    # Prepare data for QSVM (small dataset)
    X_train_qsvm, X_test_qsvm, y_train_qsvm, y_test_qsvm = dataset.prepare_balanced_dataset(
        n_fraud_train=100, n_nonfraud_train=100,
        n_fraud_test=50, n_nonfraud_test=50  # Smaller test set for kernel visualization
    )
    
    # Prepare data for Classical SVM (medium dataset)
    X_train_csvm, X_test_csvm, y_train_csvm, y_test_csvm = dataset.prepare_balanced_dataset(
        n_fraud_train=200, n_nonfraud_train=200,
        n_fraud_test=100, n_nonfraud_test=100
    )
    
    # ========================================================================
    # STEP 1: Visualize Quantum Encoding
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: QUANTUM ENCODING VISUALIZATION")
    print("="*80)
    
    visualizer = QuantumEncodingVisualizer(SELECTED_FEATURES)
    examples = dataset.get_example_samples(X_train_qsvm, y_train_qsvm)
    
    visualizer.print_encoding_explanation(examples)
    visualizer.visualize_encoding(examples, entanglement='linear', reps=1, 
                                   save_path='quantum_encoding_examples.png')
    
    # ========================================================================
    # STEP 2: Kernel Computation and Comparison
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: KERNEL COMPUTATION AND COMPARISON")
    print("="*80)
    
    kernel_framework = KernelComparisonFramework(n_features=len(SELECTED_FEATURES))
    
    # Compute classical kernel
    K_train_classical, K_test_classical, gamma = kernel_framework.compute_classical_kernel(
        X_train_qsvm, X_test_qsvm, gamma='scale'
    )
    
    # Compute quantum kernel (linear entanglement - superconducting)
    K_train_quantum_linear, K_test_quantum_linear, qkernel_linear = kernel_framework.compute_quantum_kernel(
        X_train_qsvm, X_test_qsvm, entanglement='linear', reps=1, shots=1024
    )
    
    # Calculate kernel alignment
    alignment_linear = kernel_framework.calculate_kernel_alignment(
        K_train_classical, K_train_quantum_linear
    )
    
    print(f"\n{'='*60}")
    print(f"Kernel Alignment (Classical vs Quantum-Linear): {alignment_linear:.4f}")
    print(f"{'='*60}")
    print(f"Interpretation:")
    print(f"  - 1.0: Kernels are identical")
    print(f"  - 0.7-0.9: High similarity")
    print(f"  - 0.4-0.7: Moderate similarity (GOOD - shows quantum captures different patterns)")
    print(f"  - <0.4: Low similarity (EXCELLENT - fundamentally different feature space)")
    
    # Visualize kernel comparison
    kernel_framework.visualize_kernel_comparison(
        K_train_classical, K_train_quantum_linear,
        title_suffix=f"Alignment: {alignment_linear:.4f}",
        save_path='kernel_comparison_linear.png'
    )
    
    # ========================================================================
    # STEP 3: Train and Evaluate Classical SVM
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: CLASSICAL SVM TRAINING")
    print("="*80)
    
    # Train on medium dataset
    K_train_csvm_classical, K_test_csvm_classical, _ = kernel_framework.compute_classical_kernel(
        X_train_csvm, X_test_csvm, gamma='scale'
    )
    
    classical_svm = SVC(kernel='precomputed', probability=True, random_state=42)
    classical_svm.fit(K_train_csvm_classical, y_train_csvm)
    
    y_pred_csvm = classical_svm.predict(K_test_csvm_classical)
    y_pred_proba_csvm = classical_svm.predict_proba(K_test_csvm_classical)[:, 1]
    
    metrics_csvm = ModelEvaluator.calculate_metrics(y_test_csvm, y_pred_csvm, y_pred_proba_csvm)
    ModelEvaluator.print_metrics("Classical SVM (RBF Kernel)", metrics_csvm)
    
    # ========================================================================
    # STEP 4: Train and Evaluate QSVM (Superconducting)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: QSVM TRAINING (SUPERCONDUCTING - LINEAR ENTANGLEMENT)")
    print("="*80)
    
    qsvm_linear = SVC(kernel='precomputed', probability=True, random_state=42)
    qsvm_linear.fit(K_train_quantum_linear, y_train_qsvm)
    
    y_pred_qsvm_linear = qsvm_linear.predict(K_test_quantum_linear)
    y_pred_proba_qsvm_linear = qsvm_linear.predict_proba(K_test_quantum_linear)[:, 1]
    
    metrics_qsvm_linear = ModelEvaluator.calculate_metrics(
        y_test_qsvm, y_pred_qsvm_linear, y_pred_proba_qsvm_linear
    )
    ModelEvaluator.print_metrics("QSVM (Superconducting)", metrics_qsvm_linear)
    
    # ========================================================================
    # STEP 5: Train and Evaluate QSVM (Trapped-Ion)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: QSVM TRAINING (TRAPPED-ION - FULL ENTANGLEMENT)")
    print("="*80)
    
    # Compute quantum kernel with full entanglement
    K_train_quantum_full, K_test_quantum_full, qkernel_full = kernel_framework.compute_quantum_kernel(
        X_train_qsvm, X_test_qsvm, entanglement='full', reps=1, shots=1024
    )
    
    # Calculate alignment with classical kernel
    alignment_full = kernel_framework.calculate_kernel_alignment(
        K_train_classical, K_train_quantum_full
    )
    
    print(f"\nKernel Alignment (Classical vs Quantum-Full): {alignment_full:.4f}")
    
    # Visualize
    kernel_framework.visualize_kernel_comparison(
        K_train_classical, K_train_quantum_full,
        title_suffix=f"Alignment: {alignment_full:.4f}",
        save_path='kernel_comparison_full.png'
    )
    
    # Train QSVM
    qsvm_full = SVC(kernel='precomputed', probability=True, random_state=42)
    qsvm_full.fit(K_train_quantum_full, y_train_qsvm)
    
    y_pred_qsvm_full = qsvm_full.predict(K_test_quantum_full)
    y_pred_proba_qsvm_full = qsvm_full.predict_proba(K_test_quantum_full)[:, 1]
    
    metrics_qsvm_full = ModelEvaluator.calculate_metrics(
        y_test_qsvm, y_pred_qsvm_full, y_pred_proba_qsvm_full
    )
    ModelEvaluator.print_metrics("QSVM (Trapped-Ion)", metrics_qsvm_full)
    
    # ========================================================================
    # STEP 6: Summary Comparison Table
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    
    results_df = pd.DataFrame({
        'Model': ['Classical SVM', 'QSVM (Superconducting)', 'QSVM (Trapped-Ion)'],
        'Architecture': ['CPU', 'Linear Entanglement', 'Full Entanglement'],
        'Training Samples': [400, 200, 200],
        'Test Samples': [200, 100, 100],
        'Accuracy': [
            metrics_csvm['accuracy'],
            metrics_qsvm_linear['accuracy'],
            metrics_qsvm_full['accuracy']
        ],
        'AUC': [
            metrics_csvm['auc'],
            metrics_qsvm_linear['auc'],
            metrics_qsvm_full['auc']
        ],
        'Hit Rate (Recall)': [
            metrics_csvm['recall'],
            metrics_qsvm_linear['recall'],
            metrics_qsvm_full['recall']
        ],
        'FAR (FP/TP)': [
            metrics_csvm['far'],
            metrics_qsvm_linear['far'],
            metrics_qsvm_full['far']
        ],
        'Kernel Alignment': [1.0, alignment_linear, alignment_full]
    })
    
    print(results_df.to_string(index=False))
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")


if __name__ == "__main__":
    main()
