"""
Superconducting Noise Sensitivity Analysis
Quantum Kernel-Based Financial Fraud Detection
Architecture: Superconducting Qubits Only
"""

import numpy as np
import json
import logging
from datetime import datetime
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, recall_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from qiskit.primitives import BackendSampler
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, thermal_relaxation_error, ReadoutError
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('superconducting_noise_sensitivity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# BASELINE NOISE PARAMETERS (SUPERCONDUCTING)
# ==============================================================================
BASELINE_NOISE = {
    'T1': 50e-6,              # Amplitude damping time (50 μs)
    'T2': 70e-6,              # Dephasing time (70 μs) - must be <= 2*T1
    'GATE_TIME_1Q': 50e-9,    # Single-qubit gate time (50 ns)
    'GATE_TIME_2Q': 300e-9,   # Two-qubit gate time (300 ns)
    'P_DEP_1Q': 0.001,        # Single-qubit depolarizing probability
    'P_DEP_2Q': 0.01,         # Two-qubit depolarizing probability
    'P_READOUT': 0.02         # Readout error probability
}

# ==============================================================================
# FEATURE SUBSETS (SUPERCONDUCTING ARCHITECTURE)
# ==============================================================================
FEATURE_SUBSETS = {
    'ZZFeatureMap': [14, 7, 4, 19, 20, 17],      # V14, V7, V4, V19, V20, V17
    'PauliFeatureMap': [14, 12, 4, 17, 8],       # V14, V12, V4, V17, V8
    'ZFeatureMap': [4, 7, 11, 14, 17, 19]        # Validated feature list
}

# ==============================================================================
# NOISE SWEEP CONFIGURATIONS
# ==============================================================================
NOISE_SWEEPS = {
    'T1': np.linspace(10e-6, 100e-6, 10),
    'T2': np.linspace(10e-6, 70e-6, 10),  # Will enforce T2 <= 2*T1
    'GATE_TIME_1Q': np.linspace(20e-9, 200e-9, 10),
    'GATE_TIME_2Q': np.linspace(100e-9, 500e-9, 10),
    'P_DEP_1Q': np.linspace(0.0001, 0.005, 10),
    'P_DEP_2Q': np.linspace(0.001, 0.05, 10),
    'P_READOUT': np.linspace(0.001, 0.1, 10)
}

# Dataset configuration
TRAIN_SIZE = 400
TEST_SIZE = 200
RANDOM_STATE = 42

# Output file
OUTPUT_JSON = f"superconducting_noise_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# ==============================================================================
# NOISE MODEL BUILDER
# ==============================================================================
def build_noise_model(noise_params):
    """
    Build superconducting noise model with given parameters.
    Enforces T2 <= 2*T1 constraint.
    """
    noise_model = NoiseModel()
    
    T1 = noise_params['T1']
    T2 = noise_params['T2']
    gate_time_1q = noise_params['GATE_TIME_1Q']
    gate_time_2q = noise_params['GATE_TIME_2Q']
    p_dep_1q = noise_params['P_DEP_1Q']
    p_dep_2q = noise_params['P_DEP_2Q']
    p_readout = noise_params['P_READOUT']
    
    # Enforce T2 <= 2*T1 constraint
    if T2 > 2 * T1:
        T2 = 2 * T1
        logger.warning(f"T2 adjusted to {T2:.2e} to satisfy T2 <= 2*T1 constraint")
    
    # Thermal relaxation for single-qubit gates
    thermal_1q = thermal_relaxation_error(T1, T2, gate_time_1q)
    
    # Depolarizing error for single-qubit gates
    depol_1q = depolarizing_error(p_dep_1q, 1)
    
    # Compose errors for single-qubit gates
    error_1q = thermal_1q.compose(depol_1q)
    
    # Add to single-qubit gates
    noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'u', 'u1', 'u2', 'u3', 'p'])
    
    # Thermal relaxation for two-qubit gates
    thermal_2q = thermal_relaxation_error(T1, T2, gate_time_2q)
    
    # Depolarizing error for two-qubit gates
    depol_2q = depolarizing_error(p_dep_2q, 2)
    
    # Compose errors for two-qubit gates
    error_2q = thermal_2q.compose(depol_2q)
    
    # Add to two-qubit gates
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cy', 'cz', 'swap'])
    
    # Readout error
    readout_error = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
    noise_model.add_all_qubit_readout_error(readout_error)
    
    return noise_model

# ==============================================================================
# KERNEL COMPUTATION
# ==============================================================================
# from qiskit.primitives import BackendSampler

def compute_kernel(X_train, X_test, feature_indices, feature_map_type, noise_model):
    """
    Compute noise-aware quantum kernel using FidelityQuantumKernel
    compatible with Qiskit 1.3.1
    """

    n_features = len(feature_indices)

    # Select features
    Xt_train = X_train[:, feature_indices]
    Xt_test = X_test[:, feature_indices]

    # Build feature map
    if feature_map_type == "ZZFeatureMap":
        fmap = ZZFeatureMap(n_features, reps=2, entanglement='linear')
    elif feature_map_type == "PauliFeatureMap":
        fmap = PauliFeatureMap(
            n_features, reps=2, paulis=['X', 'ZZ'], entanglement='linear'
        )
    else:  # ZFeatureMap
        fmap = ZFeatureMap(n_features, reps=2)

    # Noisy simulator (density matrix required for noise)
    simulator = AerSimulator(
        method="density_matrix",
        noise_model=noise_model
    )

    # BackendSampler WITHOUT transpile_options (Qiskit 1.3.1 compatible)
    sampler = BackendSampler(backend=simulator)

    # Noise-aware quantum kernel
    kernel = FidelityQuantumKernel(
        feature_map=fmap,
        sampler=sampler
    )

    # Compute kernel matrices
    K_train = kernel.evaluate(Xt_train, Xt_train)
    K_test = kernel.evaluate(Xt_test, Xt_train)

    # Optional numerical stabilization
    K_train = (K_train + K_train.T) / 2

    return K_train, K_test


# ==============================================================================
# QSVM TRAINING AND EVALUATION
# ==============================================================================
def run_qsvm(K_train, K_test, y_train, y_test):
    """
    Train QSVM and compute metrics.
    """
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    
    y_pred = svm.predict(K_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, svm.decision_function(K_test))
    except:
        auc = 0.0
    
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    # False Alarm Ratio = FP / (FP + TN)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return auc, acc, recall, far

# ==============================================================================
# WORKER FUNCTION
# ==============================================================================
def worker_noise_sensitivity(args):
    """
    Worker function for noise sensitivity analysis.
    Each worker handles one feature map across all noise parameters.
    """
    worker_id, feature_map_type, X_train, X_test, y_train, y_test = args
    
    logger.info(f"Worker {worker_id} started: {feature_map_type}")
    
    feature_indices = FEATURE_SUBSETS[feature_map_type]
    logger.info(f"Worker {worker_id} using features: {feature_indices}")
    
    results = []
    
    # Iterate over each noise parameter
    for param_name, param_values in NOISE_SWEEPS.items():
        logger.info(f"Worker {worker_id} sweeping parameter: {param_name}")
        
        for param_value in param_values:
            # Create noise parameters with current value
            noise_params = BASELINE_NOISE.copy()
            noise_params[param_name] = param_value
            
            # Enforce T2 constraint if sweeping T1 or T2
            if param_name == 'T1':
                if noise_params['T2'] > 2 * param_value:
                    noise_params['T2'] = 2 * param_value
            elif param_name == 'T2':
                if param_value > 2 * noise_params['T1']:
                    param_value = 2 * noise_params['T1']
                    noise_params['T2'] = param_value
            
            logger.info(f"Worker {worker_id}: {param_name} = {param_value:.6e}")
            
            try:
                # Build noise model
                noise_model = build_noise_model(noise_params)
                
                # Compute kernels
                K_train, K_test = compute_kernel(
                    X_train, X_test, feature_indices, feature_map_type, noise_model
                )
                
                # Train and evaluate
                auc, acc, recall, far = run_qsvm(K_train, K_test, y_train, y_test)
                
                logger.info(f"Worker {worker_id}: AUC={auc:.4f}, Acc={acc:.4f}, Recall={recall:.4f}, FAR={far:.4f}")
                
                # Store result
                result = {
                    'architecture': 'Superconducting',
                    'feature_map': feature_map_type,
                    'features': feature_indices,
                    'baseline_noise': BASELINE_NOISE,
                    'noise_parameter': param_name,
                    'noise_value': float(param_value),
                    'AUC': float(auc),
                    'Accuracy': float(acc),
                    'Recall': float(recall),
                    'False_Alarm_Ratio': float(far)
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error at {param_name}={param_value:.6e}: {str(e)}")
    
    logger.info(f"Worker {worker_id} completed: {len(results)} results")
    return results

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    logger.info("=" * 80)
    logger.info("SUPERCONDUCTING NOISE SENSITIVITY ANALYSIS")
    logger.info("=" * 80)
    
    # Log baseline values
    logger.info("Baseline Noise Parameters:")
    for param, value in BASELINE_NOISE.items():
        logger.info(f"  {param}: {value:.6e}")
    
    # Generate synthetic fraud detection dataset
    logger.info(f"Generating dataset: {TRAIN_SIZE} train, {TEST_SIZE} test")
    X, y = make_classification(
        n_samples=TRAIN_SIZE + TEST_SIZE,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=3,
        flip_y=0.1,
        random_state=RANDOM_STATE
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Prepare worker arguments
    worker_args = []
    for i, feature_map_type in enumerate(['ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap']):
        worker_args.append((i, feature_map_type, X_train, X_test, y_train, y_test))
    
    # Run parallel workers
    logger.info("Starting 3 workers (one per feature map)")
    with Pool(processes=3) as pool:
        all_results = pool.map(worker_noise_sensitivity, worker_args)
    
    # Flatten results
    final_results = []
    for worker_results in all_results:
        final_results.extend(worker_results)
    
    # Save to JSON
    logger.info(f"Saving {len(final_results)} results to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info("JSON write confirmed")
    logger.info("=" * 80)
    logger.info("SUPERCONDUCTING NOISE SENSITIVITY ANALYSIS COMPLETE")
    logger.info("=" * 80)

# ==============================================================================
# DEBUG MODE
# ==============================================================================
def debug_mode():
    """
    Minimal debug test: one parameter, 5 values, AUC only.
    """
    logger.info("=" * 80)
    logger.info("DEBUG MODE: Superconducting Noise Sensitivity")
    logger.info("=" * 80)
    
    logger.info("Baseline Noise Parameters:")
    for param, value in BASELINE_NOISE.items():
        logger.info(f"  {param}: {value:.6e}")
    
    # Small dataset
    X, y = make_classification(n_samples=100, n_features=30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60, test_size=40, random_state=42)
    
    # Test one parameter with 5 values
    feature_map_type = 'ZFeatureMap'
    feature_indices = FEATURE_SUBSETS[feature_map_type]
    param_name = 'T1'
    param_values = np.linspace(20e-6, 80e-6, 5)
    
    logger.info(f"Testing {param_name} with {len(param_values)} values")
    logger.info(f"Feature map: {feature_map_type}, Features: {feature_indices}")
    
    for param_value in param_values:
        noise_params = BASELINE_NOISE.copy()
        noise_params[param_name] = param_value
        
        # Enforce T2 constraint
        if noise_params['T2'] > 2 * param_value:
            noise_params['T2'] = 2 * param_value
        
        logger.info(f"{param_name} = {param_value:.6e}")
        
        noise_model = build_noise_model(noise_params)
        K_train, K_test = compute_kernel(X_train, X_test, feature_indices, feature_map_type, noise_model)
        auc, acc, recall, far = run_qsvm(K_train, K_test, y_train, y_test)
        
        logger.info(f"  AUC = {auc:.4f}")
    
    logger.info("DEBUG MODE COMPLETE")

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--debug':
        debug_mode()
    else:
        main()