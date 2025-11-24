#!/usr/bin/env python3
"""
Exhaustive Feature Selection for Fraud Detection
-------------------------------------------------
- Tests all combinations from 3 to 7 features
- Optimizes for: Maximum Hit Rate & Minimum False Alarm Rate
- Uses combined score: Hit Rate - False Alarm Rate
- Trains XGBoost on balanced 400-sample dataset
- Outputs: Best feature sets, performance metrics, visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import os
import time

# ------------------ CONFIG ------------------
DATA_PATH = "creditcard.csv"
LABEL = "Class"
RANDOM_STATE = 41
N_SAMPLES = 600  # 200 fraud + 200 genuine
TEST_SIZE = 0.33

# EXHAUSTIVE FEATURE LIST (10 features)
FEATURE_POOL = ['V14', 'V12','V4' ,'V20', 'Amount', 'V10', 'V8', 'V22', 'V13', 'V7']

# Feature selection range
MIN_FEATURES = 7
MAX_FEATURES = 7

PLOT_DIR = "./feature_selection_results/"
os.makedirs(PLOT_DIR, exist_ok=True)
# --------------------------------------------

# ------------------ LOAD BALANCED DATA ------------------
def load_balanced_data():
    """Load and create balanced dataset with 400 samples"""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[LABEL])
    y = df[LABEL]
    
    fraud_idx = y[y == 1].index
    genuine_idx = y[y == 0].index
    
    np.random.seed(RANDOM_STATE)
    n_per_class = N_SAMPLES // 2
    
    fraud_sample = np.random.choice(fraud_idx, n_per_class, replace=False)
    genuine_sample = np.random.choice(genuine_idx, n_per_class, replace=False)
    
    selected_idx = np.concatenate([fraud_sample, genuine_sample])
    np.random.shuffle(selected_idx)
    
    X_balanced = X.loc[selected_idx]
    y_balanced = y.loc[selected_idx]
    
    return X_balanced.reset_index(drop=True), y_balanced.reset_index(drop=True)

# ------------------ TRAIN AND EVALUATE ------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train XGBoost and return performance metrics"""
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'tree_method': 'hist'
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    TN, FP, FN, TP = cm.ravel()
    
    hit_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Combined score: maximize hit rate, minimize false alarm rate
    combined_score = hit_rate - false_alarm_rate
    
    return {
        'hit_rate': hit_rate,
        'false_alarm_rate': false_alarm_rate,
        'accuracy': accuracy,
        'combined_score': combined_score,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }

# ------------------ FEATURE SELECTION ------------------
def exhaustive_feature_selection(X, y):
    """
    Perform exhaustive feature selection from 3 to 7 features
    """
    results = []
    
    # Split data once for consistency
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print("\n" + "="*80)
    print("EXHAUSTIVE FEATURE SELECTION")
    print("="*80)
    
    for n_features in range(MIN_FEATURES, MAX_FEATURES + 1):
        print(f"\n{'─'*80}")
        print(f"Testing all {n_features}-feature combinations...")
        print(f"{'─'*80}")
        
        # Generate all combinations
        feature_combinations = list(combinations(FEATURE_POOL, n_features))
        n_combinations = len(feature_combinations)
        
        print(f"Total combinations to test: {n_combinations}")
        
        best_score = -np.inf
        best_combo = None
        best_metrics = None
        
        start_time = time.time()
        
        for idx, feature_set in enumerate(feature_combinations, 1):
            feature_list = list(feature_set)
            
            # Extract selected features
            X_train_subset = X_train_full[feature_list]
            X_test_subset = X_test_full[feature_list]
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_subset)
            X_test_scaled = scaler.transform(X_test_subset)
            
            # Train and evaluate
            metrics = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # Store results
            results.append({
                'n_features': n_features,
                'features': feature_list,
                'features_str': ', '.join(feature_list),
                **metrics
            })
            
            # Track best for this feature count
            if metrics['combined_score'] > best_score:
                best_score = metrics['combined_score']
                best_combo = feature_list
                best_metrics = metrics
            
            # Progress indicator
            if idx % max(1, n_combinations // 10) == 0 or idx == n_combinations:
                elapsed = time.time() - start_time
                print(f"  Progress: {idx}/{n_combinations} ({100*idx/n_combinations:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s")
        
        # Print best for this feature count
        print(f"\n✓ Best {n_features}-feature combination:")
        print(f"  Features: {best_combo}")
        print(f"  Hit Rate: {best_metrics['hit_rate']:.4f}")
        print(f"  False Alarm Rate: {best_metrics['false_alarm_rate']:.4f}")
        print(f"  Combined Score: {best_metrics['combined_score']:.4f}")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    
    return pd.DataFrame(results)

# ------------------ VISUALIZATION ------------------
def visualize_results(results_df):
    """Create comprehensive visualizations"""
    
    # 1. Best performance by feature count
    best_by_n = results_df.loc[results_df.groupby('n_features')['combined_score'].idxmax()]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Hit Rate vs Feature Count
    ax1 = axes[0, 0]
    ax1.plot(best_by_n['n_features'], best_by_n['hit_rate'], 
             'o-', linewidth=2, markersize=10, color='green', label='Hit Rate')
    ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hit Rate (TPR)', fontsize=12, fontweight='bold')
    ax1.set_title('Best Hit Rate by Feature Count', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(MIN_FEATURES, MAX_FEATURES + 1))
    
    # Plot 2: False Alarm Rate vs Feature Count
    ax2 = axes[0, 1]
    ax2.plot(best_by_n['n_features'], best_by_n['false_alarm_rate'], 
             'o-', linewidth=2, markersize=10, color='red', label='False Alarm Rate')
    ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax2.set_ylabel('False Alarm Rate (FPR)', fontsize=12, fontweight='bold')
    ax2.set_title('Best False Alarm Rate by Feature Count', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(MIN_FEATURES, MAX_FEATURES + 1))
    
    # Plot 3: Combined Score vs Feature Count
    ax3 = axes[1, 0]
    ax3.plot(best_by_n['n_features'], best_by_n['combined_score'], 
             'o-', linewidth=2, markersize=10, color='blue', label='Combined Score')
    ax3.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Combined Score (HR - FAR)', fontsize=12, fontweight='bold')
    ax3.set_title('Best Combined Score by Feature Count', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(MIN_FEATURES, MAX_FEATURES + 1))
    
    # Plot 4: Accuracy vs Feature Count
    ax4 = axes[1, 1]
    ax4.plot(best_by_n['n_features'], best_by_n['accuracy'], 
             'o-', linewidth=2, markersize=10, color='purple', label='Accuracy')
    ax4.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Best Accuracy by Feature Count', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(MIN_FEATURES, MAX_FEATURES + 1))
    
    plt.suptitle('Feature Selection Analysis: Performance vs Number of Features', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "performance_by_feature_count.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: performance_by_feature_count.png")
    
    # 2. Distribution of scores for each feature count
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['hit_rate', 'false_alarm_rate', 'combined_score']):
        ax = axes[idx]
        data_to_plot = [results_df[results_df['n_features'] == n][metric].values 
                        for n in range(MIN_FEATURES, MAX_FEATURES + 1)]
        
        bp = ax.boxplot(data_to_plot, labels=range(MIN_FEATURES, MAX_FEATURES + 1),
                        patch_artist=True, showfliers=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        metric_name = metric.replace('_', ' ').title()
        ax.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Score Distributions Across All Feature Combinations', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "score_distributions.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: score_distributions.png")

# ------------------ SAVE RESULTS ------------------
def save_results(results_df):
    """Save detailed results to CSV"""
    # Save all results
    results_df.to_csv(os.path.join(PLOT_DIR, "all_combinations_results.csv"), index=False)
    print(f"\n✓ Saved: all_combinations_results.csv")
    
    # Save best results by feature count
    best_by_n = results_df.loc[results_df.groupby('n_features')['combined_score'].idxmax()]
    best_by_n.to_csv(os.path.join(PLOT_DIR, "best_by_feature_count.csv"), index=False)
    print(f"✓ Saved: best_by_feature_count.csv")
    
    return best_by_n

# ------------------ PRINT SUMMARY ------------------
def print_summary(best_by_n):
    """Print comprehensive summary of results"""
    print("\n" + "="*80)
    print("FEATURE SELECTION SUMMARY - BEST COMBINATIONS")
    print("="*80)
    
    for _, row in best_by_n.iterrows():
        print(f"\n{'─'*80}")
        print(f"Best {row['n_features']}-Feature Combination:")
        print(f"{'─'*80}")
        print(f"Features: {row['features_str']}")
        print(f"  Hit Rate (TPR):       {row['hit_rate']:.4f} ({row['hit_rate']*100:.2f}%)")
        print(f"  False Alarm Rate:     {row['false_alarm_rate']:.4f} ({row['false_alarm_rate']*100:.2f}%)")
        print(f"  Combined Score:       {row['combined_score']:.4f}")
        print(f"  Accuracy:             {row['accuracy']:.4f} ({row['accuracy']*100:.2f}%)")
        print(f"  Confusion Matrix:     TP={row['TP']}, TN={row['TN']}, FP={row['FP']}, FN={row['FN']}")
    
    # Overall best
    overall_best = best_by_n.loc[best_by_n['combined_score'].idxmax()]
    print(f"\n{'='*80}")
    print("OVERALL BEST FEATURE COMBINATION")
    print(f"{'='*80}")
    print(f"Number of Features: {overall_best['n_features']}")
    print(f"Features: {overall_best['features_str']}")
    print(f"  Hit Rate:             {overall_best['hit_rate']:.4f} ({overall_best['hit_rate']*100:.2f}%)")
    print(f"  False Alarm Rate:     {overall_best['false_alarm_rate']:.4f} ({overall_best['false_alarm_rate']*100:.2f}%)")
    print(f"  Combined Score:       {overall_best['combined_score']:.4f}")
    print(f"  Accuracy:             {overall_best['accuracy']:.4f} ({overall_best['accuracy']*100:.2f}%)")
    print("="*80 + "\n")

# ------------------ MAIN ------------------
def main():
    print("="*80)
    print("EXHAUSTIVE FEATURE SELECTION FOR FRAUD DETECTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset size: {N_SAMPLES} samples (balanced)")
    print(f"  Feature pool: {FEATURE_POOL}")
    print(f"  Feature range: {MIN_FEATURES} to {MAX_FEATURES} features")
    print(f"  Total combinations: ", end="")
    total_combos = sum(len(list(combinations(FEATURE_POOL, n))) 
                       for n in range(MIN_FEATURES, MAX_FEATURES + 1))
    print(f"{total_combos}")
    print(f"  Optimization metric: Hit Rate - False Alarm Rate")
    
    # Load data
    print(f"\nLoading balanced dataset...")
    X, y = load_balanced_data()
    print(f"✓ Dataset loaded: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Fraud: {np.sum(y==1)}, Genuine: {np.sum(y==0)}")
    
    # Perform feature selection
    start_time = time.time()
    results_df = exhaustive_feature_selection(X, y)
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✓ Feature selection complete! Total time: {total_time:.2f}s")
    print(f"{'='*80}")
    
    # Save results
    best_by_n = save_results(results_df)
    
    # Visualize
    print(f"\nGenerating visualizations...")
    visualize_results(results_df)
    
    # Print summary
    print_summary(best_by_n)
    
    print(f"\n✓ All results saved to: {PLOT_DIR}")

if __name__ == "__main__":
    main()
