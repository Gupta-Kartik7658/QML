import numpy as np
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix

def quantum_kernel_svm_eval(selected_indices, X_train, y_train, X_test, y_test, feature_names,
                            feature_map_type='ZZ', reps=1, entanglement='full', use_gpu=True):
    """
    Evaluate a feature subset using Quantum Kernel + SVM pipeline.
    Returns AUC as reward.
    """
    print(f"Evaluating features: {selected_indices}")
    n_features = len(selected_indices)
    X_train_subset = X_train[:, selected_indices]
    X_test_subset = X_test[:, selected_indices]

    # Build feature map
    if feature_map_type == 'ZZ':
        feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement=entanglement)
    else:
        feature_map = ZFeatureMap(feature_dimension=n_features, reps=reps)

    # Simulator
    if use_gpu:
        simulator = AerSimulator(method='statevector', device='GPU')
    else:
        simulator = AerSimulator(method='statevector', device='CPU')

    # Quantum kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    # Compute kernel matrices
    K_train = quantum_kernel.evaluate(X_train_subset, X_train_subset)
    K_test = quantum_kernel.evaluate(X_test_subset, X_train_subset)

    # Train SVM
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    y_pred = svm.predict(K_test)

    # Try AUC, fallback to accuracy if not possible
    try:
        y_score = svm.decision_function(K_test)
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = accuracy_score(y_test, y_pred)

    # Compute confusion matrix and metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_ratio = fp / tp if tp > 0 else float('inf')

    return auc, accuracy, hit_rate, false_alarm_ratio