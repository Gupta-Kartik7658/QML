import pandas as pd
import numpy as np
import itertools
from time import time

# --- Scikit-learn & XGBoost Imports ---
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# -----------------------------------------------------------------
# STEP 1: LOAD AND PREPARE DATA (Unchanged)
# -----------------------------------------------------------------

df = pd.read_csv("creditcard.csv")

# Separate features and Class
X = df.drop('Class', axis=1)
y = df['Class']

# 1️⃣ Train-test split first (to avoid data leakage)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2️⃣ Undersample only the training data
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train_orig, y_train_orig)

# 3️⃣ Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test_orig)

# ✅ Results
print("Before undersampling:", y_train_orig.value_counts().to_dict())
print("After undersampling:", pd.Series(y_train_res).value_counts().to_dict())
print("-" * 40)

# -----------------------------------------------------------------
# STEP 2: PREPARE DATA FOR FEATURE SELECTION (Unchanged)
# -----------------------------------------------------------------

# 🔴 CRITICAL FIX: Convert scaled arrays back to DataFrames 
X_train = pd.DataFrame(X_train_scaled, columns=X_train_res.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test_orig.columns)
y_train = y_train_res
y_test = y_test_orig # This is a pandas Series


TOTAL_FEATURES = X_train.shape[1]
ALL_FEATURE_INDICES = list(range(TOTAL_FEATURES))

# --- Count total frauds (positives) and non-frauds (negatives) ---
TOTAL_TEST_FRAUDS = np.sum(y_test)
TOTAL_TEST_NON_FRAUDS = len(y_test) - TOTAL_TEST_FRAUDS

print(f"Data prepared: {TOTAL_FEATURES} total features.")
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Total frauds in test set: {TOTAL_TEST_FRAUDS}")
print(f"Total non-frauds in test set: {TOTAL_TEST_NON_FRAUDS}")
print("-" * 40)

# --- Configuration Parameters ---
# ✅ CHANGED: Set both to 10 to test *only* 10-feature combinations
STARTING_FEATURES = 6 
MAX_FEATURES = 6


# -----------------------------------------------------------------
# STEP 3: DEFINE THE XGBOOST EVALUATION FUNCTION (MODIFIED)
# -----------------------------------------------------------------

def run_xgboost(train_data, test_data, y_train_data, y_test_data, num_features):
    """
    Trains and evaluates an XGBoost classifier.
    Returns: (train_acc, test_acc, train_tps, test_tps, train_fps, test_fps)
    """
    
    try:
        # --- Auto-detect classification type ---
        unique_classes = np.unique(y_train_data)
        num_classes = len(unique_classes)
        
        if num_classes == 2:
            objective = 'binary:logistic'
            eval_metric = 'logloss'
            num_class_param = {}
        else:
            objective = 'multi:softmax'
            eval_metric = 'mlogloss'
            num_class_param = {'num_class': num_classes}

        # 1. Define the XGBoost Model
        xgb_model = XGBClassifier(
            # ✅ CHANGED: Removed 'device='cuda'' to run on CPU
            objective=objective,
            **num_class_param,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            eval_metric=eval_metric,
            random_state=42
        )
        
        # Convert pandas objects to NumPy arrays
        train_data_np = train_data.values
        test_data_np = test_data.values
        y_test_data_np = y_test_data.values 
        
        
        # 2. Train the Model
        xgb_model.fit(train_data_np, y_train_data) 
        
        # 3. Calculate KPIs
        
        # --- Calculate Test Metrics ---
        test_predictions = xgb_model.predict(test_data_np) 
        test_accuracy = accuracy_score(y_test_data_np, test_predictions)
        cm_test = confusion_matrix(y_test_data_np, test_predictions, labels=[0, 1]) 
        test_tps = cm_test[1, 1] 
        test_fps = cm_test[0, 1] 

        # --- Calculate Train (Hit) Metrics ---
        train_predictions = xgb_model.predict(train_data_np) 
        train_accuracy = accuracy_score(y_train_data, train_predictions)
        cm_train = confusion_matrix(y_train_data, train_predictions, labels=[0, 1])
        train_tps = cm_train[1, 1]
        train_fps = cm_train[0, 1]
        
        return train_accuracy, test_accuracy, train_tps, test_tps, train_fps, test_fps
        
    except Exception as e:
        print(f"     [Error] Failed to train model: {e}")
        return 0.0, 0.0, 0, 0, 0, 0

# -----------------------------------------------------------------
# STEP 4: IMPLEMENT THE FEATURE SELECTION ALGORITHM (Unchanged)
# -----------------------------------------------------------------

print("Starting XGBoost Feature Selection Algorithm (on CPU)...")

# Global trackers for the best result found so far
overall_best_accuracy = 0.0 
overall_best_set = []
overall_best_tps = 0    
overall_best_fps = 0    

for p in range(STARTING_FEATURES, MAX_FEATURES + 1):
    start_time = time()
    print(f"\n--- Finding best {p} features ---")
    
    current_loop_best_accuracy = 0.0
    current_loop_best_set = []
    current_loop_best_tps = 0
    current_loop_best_fps = 0 
    
    combinations_to_test = []
    
    if p == STARTING_FEATURES:
        print(f"Testing all {p}-feature combinations...")
        combinations_to_test = list(itertools.combinations(ALL_FEATURE_INDICES, p))
        
    else:
        # This block will not be reached because STARTING_FEATURES == MAX_FEATURES
        print(f"Expanding on best {p-1} set: {overall_best_set}")
        remaining_indices = [
            idx for idx in ALL_FEATURE_INDICES if idx not in overall_best_set
        ]
        
        for new_idx in remaining_indices:
            new_combo = overall_best_set + [new_idx]
            combinations_to_test.append(tuple(new_combo))

    if not combinations_to_test:
        print("No new combinations to test. Stopping.")
        break
        
    print(f"Total combinations to test: {len(combinations_to_test)}")

    for i, combo in enumerate(combinations_to_test):
        combo_list = list(combo)
        
        # Use .iloc to select columns by integer index from the DataFrame
        X_train_subset = X_train.iloc[:, combo_list]
        X_test_subset = X_test.iloc[:, combo_list]
        
        train_acc, test_acc, train_tps, test_tps, train_fps, test_fps = run_xgboost(
            X_train_subset, X_test_subset, 
            y_train, y_test, 
            num_features=p
        )
        
        hit_ratio = (test_tps / TOTAL_TEST_FRAUDS) if TOTAL_TEST_FRAUDS > 0 else 0.0
        ffr = (test_fps / TOTAL_TEST_NON_FRAUDS) if TOTAL_TEST_NON_FRAUDS > 0 else 0.0
        
        print(f"  [{i+1}/{len(combinations_to_test)}] "
              f"Combo: {combo_list} | Test Acc: {test_acc:.4f} "
              f"| Hit Rate: {hit_ratio:.2%} ({test_tps}/{TOTAL_TEST_FRAUDS}) "
              f"| FPR: {ffr:.2%} ({test_fps}/{TOTAL_TEST_NON_FRAUDS}) "
              f"| Train Acc: {train_acc:.4f}")
        
        if test_acc > current_loop_best_accuracy:
            current_loop_best_accuracy = test_acc
            current_loop_best_set = combo_list
            current_loop_best_tps = test_tps 
            current_loop_best_fps = test_fps 
            
    loop_time = time() - start_time
    
    best_hit_ratio = (current_loop_best_tps / TOTAL_TEST_FRAUDS) if TOTAL_TEST_FRAUDS > 0 else 0.0
    best_ffr = (current_loop_best_fps / TOTAL_TEST_NON_FRAUDS) if TOTAL_TEST_NON_FRAUDS > 0 else 0.0
    
    print(f"Best {p}-feature set: {current_loop_best_set} "
          f"| Test Acc: {current_loop_best_accuracy:.4f} "
          f"| Hit Rate: {best_hit_ratio:.2%} ({current_loop_best_tps}/{TOTAL_TEST_FRAUDS}) "
          f"| FPR: {best_ffr:.2%} ({current_loop_best_fps}/{TOTAL_TEST_NON_FRAUDS})")
    print(f"Time taken: {loop_time:.2f} seconds")

    if current_loop_best_accuracy > overall_best_accuracy:
        overall_best_accuracy = current_loop_best_accuracy
        overall_best_set = current_loop_best_set
        overall_best_tps = current_loop_best_tps 
        overall_best_fps = current_loop_best_fps 
    else:
        # This block will not be reached
        print(f"\nTest Accuracy did not improve from "
              f"{overall_best_accuracy:.4f}. Stopping algorithm.")
        break

print("-" * 40)
print("\n--- XGBoost Feature Selection Complete ---")
print(f"Final Best Feature Set:   {overall_best_set}")
print(f"Final Best Test Accuracy: {overall_best_accuracy:.4f}")

final_hit_ratio = (overall_best_tps / TOTAL_TEST_FRAUDS) if TOTAL_TEST_FRAUDS > 0 else 0.0
final_ffr = (overall_best_fps / TOTAL_TEST_NON_FRAUDS) if TOTAL_TEST_NON_FRAUDS > 0 else 0.0

print(f"Hit Rate (TPR):           {final_hit_ratio:.2%} ({overall_best_tps} / {TOTAL_TEST_FRAUDS})")
print(f"False Positive Rate (FPR):{final_ffr:.2%} ({overall_best_fps} / {TOTAL_TEST_NON_FRAUDS})")
print("-" * 40)
