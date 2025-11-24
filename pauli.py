#!/usr/bin/env python3
"""
Quantum Kernel Performance: PauliFeatureMap Only
------------------------------------------------
- Uses a reduced balanced dataset (200 train, 100 test)
- Quantum kernel: PauliFeatureMap
- Metrics:
    - Confusion matrix
    - AUC (ROC)
    - Hit rate (TPR)
    - False alarm rate (FPR)
- Visualizations:
    1. Kernel PCA projection
    2. Kernel matrix heatmap
"""

import os, time, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score
from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# ------------------ CONFIG ------------------
DATA_PATH = "creditcard.csv"
LABEL = "Class"
FEATURE_LIST = ['V14','V12','V4','V10','V8','V13','V7']
RANDOM_STATE = 41
PLOT_DIR = "./kernel_visuals/"
N_SAMPLES = 300  # 200 train + 100 test
os.makedirs(PLOT_DIR, exist_ok=True)
# --------------------------------------------

# ------------------ LOAD + BALANCE ------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[LABEL])
    y = df[LABEL]
    fraud_idx = y[y==1].index
    gen_idx = y[y==0].index
    np.random.seed(RANDOM_STATE)
    n_per_class = N_SAMPLES // 2
    fraud_sample = np.random.choice(fraud_idx, n_per_class, replace=False)
    gen_sample = np.random.choice(gen_idx, n_per_class, replace=False)
    idx = np.concatenate([fraud_sample, gen_sample])
    np.random.shuffle(idx)
    X_sel = X.loc[idx]
    y_sel = y.loc[idx]
    return X_sel, y_sel

# ------------------ PREPROCESS ------------------
def preprocess(X, y):
    scaler = MinMaxScaler((-1,1))
    X_scaled = scaler.fit_transform(X)
    X_sub = X_scaled[:, [X.columns.get_loc(f) for f in FEATURE_LIST]]
    return X_sub, y.values

# ------------------ QUANTUM KERNEL ------------------
def compute_pauli_kernel(X):
    fmap = PauliFeatureMap(
        feature_dimension=len(FEATURE_LIST),
        paulis=['X', 'Y', 'Z', 'ZZ'],
        reps=1,
        entanglement='linear'
    )
    qkernel = FidelityQuantumKernel(feature_map=fmap)
    print(f"Computing PauliFeatureMap kernel on {X.shape[0]} samples ...")
    t0 = time.time()
    K = qkernel.evaluate(X, X)
    print(f"✓ PauliFeatureMap kernel computed in {time.time()-t0:.2f}s")
    return K

# ------------------ PERFORMANCE EVALUATION ------------------
def evaluate_kernel_performance(K, y):
    """
    Split dataset into 200 train and 100 test samples (stratified)
    Evaluate performance using precomputed kernel SVM.
    """
    n = len(y)
    idx_train, idx_test = train_test_split(
        np.arange(n), test_size=100, train_size=200,
        stratify=y, random_state=RANDOM_STATE
    )

    K_train = K[np.ix_(idx_train, idx_train)]
    K_test = K[np.ix_(idx_test, idx_train)]

    clf = SVC(kernel="precomputed", probability=True, random_state=RANDOM_STATE)
    clf.fit(K_train, y[idx_train])

    y_pred = clf.predict(K_test)
    y_prob = clf.predict_proba(K_test)[:, 1]

    cm = confusion_matrix(y[idx_test], y_pred)
    auc = roc_auc_score(y[idx_test], y_prob)
    hit_rate = recall_score(y[idx_test], y_pred)
    false_alarm = cm[0, 1] / (cm[0, 0] + cm[0, 1])

    print(f"\n======================================================================")
    print(f"PERFORMANCE METRICS — PauliFeatureMap")
    print(f"======================================================================")
    print(f"Confusion Matrix:\n{cm}")
    print(f"AUC                 : {auc:.4f}")
    print(f"Hit Rate (Recall)   : {hit_rate:.4f}")
    print(f"False Alarm Rate    : {false_alarm:.4f}")
    print(f"======================================================================\n")

    return cm, auc, hit_rate, false_alarm, y[idx_test], y_pred

# ------------------ VISUALIZATION ------------------
def visualize_kpca(K, y):
    kpca = KernelPCA(kernel="precomputed", n_components=2)
    X_kpca = kpca.fit_transform(K)

    plt.figure(figsize=(7, 6))
    plt.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
                c='#1f77b4', label="Genuine", alpha=0.7, edgecolors='k', s=60)
    plt.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
                c='#d62728', label="Fraud", alpha=0.7, edgecolors='k', s=60)
    plt.title("Kernel PCA Projection — PauliFeatureMap", fontsize=14, fontweight='bold')
    plt.xlabel("Kernel PCA Component 1")
    plt.ylabel("Kernel PCA Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "PauliFeatureMap_PCA.png"), dpi=300)
    plt.close()
    print("✓ Saved PauliFeatureMap PCA plot -> PauliFeatureMap_PCA.png")

def visualize_heatmap(K):
    plt.figure(figsize=(6, 5))
    plt.imshow(K, cmap="viridis", aspect='auto', vmin=0, vmax=1)
    plt.title("PauliFeatureMap Kernel Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    cbar = plt.colorbar()
    cbar.set_label("Kernel Similarity", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "PauliFeatureMap_heatmap.png"), dpi=300)
    plt.close()
    print("✓ Saved PauliFeatureMap kernel heatmap -> PauliFeatureMap_heatmap.png")

# ------------------ MAIN ------------------
def main():
    X, y = load_data()
    X_sub, y = preprocess(X, y)

    print(f"\n{'='*60}")
    print(f"✓ Prepared dataset: {X_sub.shape[0]} samples × {X_sub.shape[1]} features")
    print(f"  Fraud cases: {np.sum(y==1)}, Genuine: {np.sum(y==0)}")
    print(f"{'='*60}\n")

    # Compute Pauli kernel
    K_pauli = compute_pauli_kernel(X_sub)

    # Evaluate performance
    cm, auc, hit_rate, false_alarm, y_true, y_pred = evaluate_kernel_performance(K_pauli, y)

    # Save metrics summary
    metrics_df = pd.DataFrame([{
        "Kernel": "PauliFeatureMap",
        "AUC": auc,
        "Hit Rate": hit_rate,
        "False Alarm Rate": false_alarm
    }])
    metrics_df.to_csv(os.path.join(PLOT_DIR, "PauliFeatureMap_performance.csv"), index=False)
    print("✓ Saved performance summary -> PauliFeatureMap_performance.csv")

    # Visualizations
    visualize_kpca(K_pauli, y)
    visualize_heatmap(K_pauli)

    print("\n✓ All results saved in:", PLOT_DIR)

if __name__ == "__main__":
    main()

