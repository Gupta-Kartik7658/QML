#!/usr/bin/env python3
"""
Visual + Performance Comparison of Classical and Quantum Kernels
-----------------------------------------------------------------
- Balanced subset of dataset for quantum kernels
- 3 kernel types: Classical RBF, ZZFeatureMap, PauliFeatureMap
- Visualizations:
    1. Kernel PCA projections
    2. Kernel matrix heatmaps
- Performance Metrics:
    - Confusion matrix
    - AUC (ROC)
    - Hit rate (TPR)
    - False alarm rate (FPR)
"""

import os, time, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# ------------------ CONFIG ------------------
DATA_PATH = "creditcard.csv"
LABEL = "Class"
FEATURE_LIST = ['V14','V12','V4','V10','V8','V13','V7']
RANDOM_STATE = 41
PLOT_DIR = "./kernel_visuals/"
N_SAMPLES = 400  # Reduced sample size for quantum kernels
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

# ------------------ QUANTUM KERNELS ------------------
def compute_quantum_kernel(X, map_type):
    if map_type == "ZZ":
        fmap = ZZFeatureMap(len(FEATURE_LIST), reps=1, entanglement='linear')
    elif map_type == "Pauli":
        fmap = PauliFeatureMap(len(FEATURE_LIST), paulis=['X','Y','Z','ZZ'], reps=1, entanglement='linear')
    else:
        raise ValueError(map_type)
    qkernel = FidelityQuantumKernel(feature_map=fmap)
    print(f"Computing {map_type} kernel on {X.shape[0]} samples ...")
    t0 = time.time()
    K = qkernel.evaluate(X, X)
    print(f"✓ {map_type} kernel computed in {time.time()-t0:.2f}s")
    return K

# ------------------ CLASSICAL RBF ------------------
def compute_rbf_kernel(X):
    gamma = 1 / (X.shape[1] * X.var())
    print(f"Computing RBF kernel (γ={gamma:.6f}) on {X.shape[0]} samples ...")
    K = rbf_kernel(X, X, gamma=gamma)
    return K

# ------------------ KERNEL ALIGNMENT ------------------
def kernel_alignment(K1, K2):
    """Compute centered alignment between two kernel matrices."""
    K1c = K1 - K1.mean(axis=0)[None, :] - K1.mean(axis=1)[:, None] + K1.mean()
    K2c = K2 - K2.mean(axis=0)[None, :] - K2.mean(axis=1)[:, None] + K2.mean()
    return np.sum(K1c * K2c) / np.sqrt(np.sum(K1c**2) * np.sum(K2c**2))

# ------------------ PERFORMANCE EVALUATION ------------------
def evaluate_kernel_performance(K, y, name):
    """
    Train-test split and evaluate performance using precomputed kernel SVM.
    """
    n = len(y)
    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.3, 
                                           stratify=y, random_state=RANDOM_STATE)

    K_train = K[np.ix_(idx_train, idx_train)]
    K_test = K[np.ix_(idx_test, idx_train)]

    clf = SVC(kernel="precomputed", probability=True, random_state=RANDOM_STATE)
    clf.fit(K_train, y[idx_train])

    y_pred = clf.predict(K_test)
    y_prob = clf.predict_proba(K_test)[:, 1]

    cm = confusion_matrix(y[idx_test], y_pred)
    auc = roc_auc_score(y[idx_test], y_prob)
    hit_rate = recall_score(y[idx_test], y_pred)  # TPR
    false_alarm = cm[0, 1] / (cm[0, 0] + cm[0, 1])  # FPR

    print(f"\n======================================================================")
    print(f"PERFORMANCE METRICS — {name}")
    print(f"======================================================================")
    print(f"Confusion Matrix:\n{cm}")
    print(f"AUC                 : {auc:.4f}")
    print(f"Hit Rate (Recall)   : {hit_rate:.4f}")
    print(f"False Alarm Rate    : {false_alarm:.4f}")
    print(f"======================================================================\n")

    return {
        "Kernel": name,
        "AUC": auc,
        "Hit Rate": hit_rate,
        "False Alarm Rate": false_alarm,
        "Confusion Matrix": cm
    }

# ------------------ VISUALIZATION ------------------
def visualize_kpca(K_dict, y, alignments):
    kernels = list(K_dict.keys())
    colors = ['#1f77b4', '#d62728']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, name in enumerate(kernels):
        kpca = KernelPCA(kernel="precomputed", n_components=2)
        X_kpca = kpca.fit_transform(K_dict[name])

        ax = axes[i]
        ax.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
                   c=colors[0], label="Genuine", alpha=0.7, edgecolors='k', s=60)
        ax.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
                   c=colors[1], label="Fraud", alpha=0.7, edgecolors='k', s=60)
        ax.set_xlabel("Kernel PCA Component 1", fontsize=11)
        ax.set_ylabel("Kernel PCA Component 2", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)

        if name == "Classical RBF":
            title = f"{name} Kernel"
        else:
            title = f"{name} Kernel\nAlignment w/ RBF: {alignments[name]:.4f}"
        ax.set_title(title, fontsize=12, fontweight='bold')

        if i == 0:
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    fig.suptitle("Kernel PCA Projections: Classical vs Quantum Feature Maps", 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "kernelPCA_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved PCA comparison -> kernelPCA_comparison.png")

def visualize_heatmaps(K_dict, y, alignments):
    kernels = list(K_dict.keys())
    cmap_heat = "viridis"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, name in enumerate(kernels):
        ax = axes[i]
        im = ax.imshow(K_dict[name], cmap=cmap_heat, aspect='auto', 
                       vmin=0, vmax=1, interpolation='nearest')
        if name == "Classical RBF":
            title = f"{name} Kernel Matrix"
        else:
            title = f"{name} Kernel Matrix\nAlignment w/ RBF: {alignments[name]:.4f}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Index", fontsize=11)
        ax.set_ylabel("Sample Index", fontsize=11)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Kernel Similarity', fontsize=10)
    
    fig.suptitle("Kernel Matrix Heatmaps: Classical RBF vs Quantum Feature Maps", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "kernelHeatmaps_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved kernel heatmap comparison -> kernelHeatmaps_comparison.png")

# ------------------ MAIN ------------------
def main():
    X, y = load_data()
    X_sub, y = preprocess(X, y)

    print(f"\n{'='*60}")
    print(f"✓ Prepared dataset: {X_sub.shape[0]} samples × {X_sub.shape[1]} features")
    print(f"  Fraud cases: {np.sum(y==1)}, Genuine: {np.sum(y==0)}")
    print(f"{'='*60}\n")

    # Compute all kernels
    kernels = {
        "Classical RBF": compute_rbf_kernel(X_sub),
        "ZZFeatureMap": compute_quantum_kernel(X_sub, "ZZ"),
        "PauliFeatureMap": compute_quantum_kernel(X_sub, "Pauli")
    }

    # Compute kernel alignments
    print(f"\n{'='*60}")
    print("KERNEL ALIGNMENT SCORES:")
    print(f"{'='*60}")
    alignments = {}
    for name in ["ZZFeatureMap", "PauliFeatureMap"]:
        score = kernel_alignment(kernels["Classical RBF"], kernels[name])
        alignments[name] = score
        print(f"  {name:20s}: {score:.4f}")
    alignments["Classical RBF"] = 1.0000
    print(f"  {'Classical RBF':20s}: 1.0000 (self)")
    print(f"{'='*60}\n")

    # Evaluate kernel classification performance
    print("Evaluating classification performance for each kernel...")
    metrics = []
    for name, K in kernels.items():
        results = evaluate_kernel_performance(K, y, name)
        metrics.append(results)

    # Summary
    metrics_df = pd.DataFrame([{
        "Kernel": m["Kernel"],
        "AUC": m["AUC"],
        "Hit Rate": m["Hit Rate"],
        "False Alarm Rate": m["False Alarm Rate"]
    } for m in metrics])

    print("\n======================================================================")
    print("SUMMARY OF KERNEL PERFORMANCE")
    print("======================================================================")
    print(metrics_df.to_string(index=False))
    print("======================================================================\n")

    metrics_df.to_csv(os.path.join(PLOT_DIR, "kernel_performance_summary.csv"), index=False)
    print("✓ Saved kernel performance summary -> kernel_performance_summary.csv")

    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_kpca(kernels, y, alignments)
    visualize_heatmaps(kernels, y, alignments)
    print(f"\n✓ All plots and results saved to: {PLOT_DIR}")

if __name__ == "__main__":
    main()

