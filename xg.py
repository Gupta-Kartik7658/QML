#!/usr/bin/env python3
"""
XGBoost Fraud Detection Model with Feature Importance
------------------------------------------------------
- Train on balanced 600-sample dataset (300 fraud + 300 genuine)
- Uses all features from creditcard.csv
- Outputs: Confusion Matrix, Hit Rate, False Alarm Rate, Feature Rankings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

# ------------------ CONFIG ------------------
DATA_PATH = "creditcard.csv"
LABEL = "Class"
RANDOM_STATE = 40
N_SAMPLES = 600
TEST_SIZE = 0.33
PLOT_DIR = "./xgboost_results/"
import os
os.makedirs(PLOT_DIR, exist_ok=True)
# --------------------------------------------

# ------------------ LOAD BALANCED DATA ------------------
def load_balanced_data():
    """Load and create balanced dataset"""
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
    
    return X_balanced, y_balanced

# ------------------ PREPROCESS ------------------
def preprocess_data(X_train, X_test):
    """Standardize features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ------------------ TRAIN MODEL ------------------
def train_xgboost(X_train, y_train):
    """Train XGBoost classifier"""
    print("\nTraining XGBoost model...")
    
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
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    print("✓ Model training complete!")
    return model

# ------------------ EVALUATE MODEL ------------------
def evaluate_model(model, X_test, y_test):
    """Generate predictions and compute metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    hit_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * hit_rate) / (precision + hit_rate) if (precision + hit_rate) > 0 else 0
    
    return cm, hit_rate, false_alarm_rate, accuracy, precision, f1_score, y_pred

# ------------------ FEATURE IMPORTANCE ------------------
def get_feature_importance(model, feature_names):
    """Extract and rank feature importance"""
    importance_dict = {
        'weight': model.get_booster().get_score(importance_type='weight'),
        'gain': model.get_booster().get_score(importance_type='gain'),
        'cover': model.get_booster().get_score(importance_type='cover')
    }
    
    # Create dataframe with all importance types
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': [importance_dict['weight'].get(f'f{i}', 0) for i in range(len(feature_names))],
        'Gain': [importance_dict['gain'].get(f'f{i}', 0) for i in range(len(feature_names))],
        'Cover': [importance_dict['cover'].get(f'f{i}', 0) for i in range(len(feature_names))]
    })
    
    # Normalize scores to 0-100 scale
    for col in ['Weight', 'Gain', 'Cover']:
        if importance_df[col].sum() > 0:
            importance_df[col] = 100 * importance_df[col] / importance_df[col].sum()
    
    # Sort by Gain (most important metric)
    importance_df = importance_df.sort_values('Gain', ascending=False).reset_index(drop=True)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    
    return importance_df

def plot_feature_importance(importance_df):
    """Create feature importance visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    importance_types = ['Weight', 'Gain', 'Cover']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (imp_type, color) in enumerate(zip(importance_types, colors)):
        ax = axes[idx]
        
        # Sort by current importance type
        sorted_df = importance_df.sort_values(imp_type, ascending=True)
        
        # Only show top 20 features if there are more than 20
        if len(sorted_df) > 20:
            sorted_df = sorted_df.tail(20)
        
        y_pos = np.arange(len(sorted_df))
        
        bars = ax.barh(y_pos, sorted_df[imp_type], color=color, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df['Feature'], fontsize=9)
        ax.set_xlabel(f'{imp_type} Importance Score', fontsize=11, fontweight='bold')
        ax.set_title(f'Feature Importance by {imp_type}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_df[imp_type])):
            if val > 0:
                ax.text(val, i, f' {val:.1f}', va='center', fontsize=8)
    
    plt.suptitle('XGBoost Feature Importance Rankings', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance plot saved to {PLOT_DIR}feature_importance.png")

# ------------------ VISUALIZATION ------------------
def plot_confusion_matrix(cm, hit_rate, false_alarm_rate):
    """Create and save confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    
    total = np.sum(cm)
    annotations = np.array([[f'{cm[i,j]}\n({100*cm[i,j]/total:.1f}%)' 
                            for j in range(cm.shape[1])] 
                           for i in range(cm.shape[0])])
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                xticklabels=['Genuine (0)', 'Fraud (1)'],
                yticklabels=['Genuine (0)', 'Fraud (1)'],
                linewidths=2, linecolor='black')
    
    plt.title('Confusion Matrix - XGBoost Fraud Detection\n' + 
              f'Hit Rate: {hit_rate:.4f} | False Alarm Rate: {false_alarm_rate:.4f}',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {PLOT_DIR}confusion_matrix.png")

# ------------------ PRINT RESULTS ------------------
def print_results(cm, hit_rate, false_alarm_rate, accuracy, precision, f1_score, importance_df):
    """Print detailed results"""
    TN, FP, FN, TP = cm.ravel()
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print(f"{'':20s} | Predicted: Genuine | Predicted: Fraud")
    print("-"*70)
    print(f"{'Actual: Genuine':20s} | {TN:^18d} | {FP:^15d}")
    print(f"{'Actual: Fraud':20s} | {FN:^18d} | {TP:^15d}")
    print("="*70)
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"{'Hit Rate (TPR/Recall)':30s}: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"{'False Alarm Rate (FPR)':30s}: {false_alarm_rate:.4f} ({false_alarm_rate*100:.2f}%)")
    print(f"{'Accuracy':30s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'Precision':30s}: {precision:.4f} ({precision*100:.2f}%)")
    print(f"{'F1-Score':30s}: {f1_score:.4f}")
    print("="*70)
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"• Out of {TP+FN} fraud cases, the model correctly detected {TP} ({hit_rate*100:.1f}%)")
    print(f"• Out of {TN+FP} genuine cases, the model falsely flagged {FP} ({false_alarm_rate*100:.1f}%)")
    print(f"• Overall accuracy: {accuracy*100:.2f}%")
    print("="*70)
    
    # Print feature importance rankings
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE RANKINGS (Top 15)")
    print("="*70)
    print(f"{'Rank':<6} {'Feature':<12} {'Weight':<12} {'Gain':<12} {'Cover':<12}")
    print("-"*70)
    
    # Show top 15 features
    for idx, row in importance_df.head(15).iterrows():
        print(f"{row['Rank']:<6} {row['Feature']:<12} "
              f"{row['Weight']:>10.2f}  {row['Gain']:>10.2f}  {row['Cover']:>10.2f}")
    
    if len(importance_df) > 15:
        print(f"... and {len(importance_df) - 15} more features")
    print("="*70 + "\n")

# ------------------ MAIN ------------------
def main():
    print("="*70)
    print("XGBoost Fraud Detection Model Training")
    print("="*70)
    
    # Load balanced dataset
    print(f"\nLoading balanced dataset ({N_SAMPLES} samples)...")
    X, y = load_balanced_data()
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Fraud cases: {np.sum(y == 1)}")
    print(f"  Genuine cases: {np.sum(y == 0)}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain-Test Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Preprocess
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    # Train model
    model = train_xgboost(X_train_scaled, y_train)
    
    # Evaluate
    print("\nEvaluating model on test set...")
    cm, hit_rate, false_alarm_rate, accuracy, precision, f1_score, y_pred = \
        evaluate_model(model, X_test_scaled, y_test)
    
    # Get feature importance
    print("\nExtracting feature importance...")
    feature_names = X.columns.tolist()
    importance_df = get_feature_importance(model, feature_names)
    
    # Save feature importance to CSV
    importance_df.to_csv(os.path.join(PLOT_DIR, "feature_importance.csv"), index=False)
    print(f"✓ Feature importance saved to {PLOT_DIR}feature_importance.csv")
    
    # Print results
    print_results(cm, hit_rate, false_alarm_rate, accuracy, precision, f1_score, importance_df)
    
    # Visualize
    plot_confusion_matrix(cm, hit_rate, false_alarm_rate)
    plot_feature_importance(importance_df)
    
    print("\n✓ All results generated successfully!")
    print(f"✓ Outputs saved to: {PLOT_DIR}\n")

if __name__ == "__main__":
    main()
