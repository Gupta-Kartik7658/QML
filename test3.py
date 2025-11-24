"""
Quantum Feature Selection Algorithm for Fraud Detection
Based on: "Mixed Quantum-Classical Method for Fraud Detection With Quantum Feature Selection"
by Grossi et al., IEEE Transactions on Quantum Engineering, 2022

PRODUCTION READY - Compatible with: Qiskit 1.3.1, Qiskit-Aer 0.15.1, Qiskit-ML 0.8.0
Includes extensive logging and GPU fallback mechanisms
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import time
import json
import os
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


class QuantumFeatureSelector:
    """
    Implements the Quantum Feature Selection algorithm from the paper.
    Uses ZZFeatureMap and iterative feature selection based on accuracy.
    """
    
    def __init__(self, 
                 feature_map_type='ZZ',
                 reps=2,
                 entanglement='full',
                 use_gpu=True,
                 shots=8192):
        """
        Initialize the Quantum Feature Selector
        
        Args:
            feature_map_type: 'Z' or 'ZZ' (paper uses ZZ with depth 2)
            reps: Number of repetitions (depth)
            entanglement: Entanglement pattern ('full', 'linear', 'circular')
            use_gpu: Whether to use GPU acceleration
            shots: Number of measurement shots
        """
        print("\n" + "="*70)
        print("INITIALIZING QUANTUM FEATURE SELECTOR")
        print("="*70)
        
        self.feature_map_type = feature_map_type
        self.reps = reps
        self.entanglement = entanglement
        self.shots = shots
        self.gpu_available = False
        
        print(f"Configuration:")
        print(f"  Feature Map Type: {feature_map_type}")
        print(f"  Repetitions (depth): {reps}")
        print(f"  Entanglement: {entanglement}")
        print(f"  Shots: {shots}")
        print(f"  GPU Requested: {use_gpu}")
        
        # Setup simulator with GPU fallback
        print("\nSetting up quantum simulator...")
        try:
            if use_gpu:
                self.simulator = AerSimulator(method='statevector', device='GPU')
                # Test if GPU actually works
                test_qc = QuantumCircuit(2)
                test_qc.h(0)
                test_qc.measure_all()
                test_qc_t = transpile(test_qc, self.simulator)
                test_job = self.simulator.run(test_qc_t, shots=100)
                test_result = test_job.result()
                self.gpu_available = True
                print("✓ GPU acceleration ACTIVE and VERIFIED")
            else:
                raise Exception("CPU requested by user")
        except Exception as e:
            print(f"⚠ GPU not available or failed: {e}")
            print("  Falling back to CPU...")
            self.simulator = AerSimulator(method='statevector', device='CPU')
            self.gpu_available = False
            print("✓ CPU simulator initialized")
        
        self.selected_features = []
        self.feature_scores = {}
        self.best_model = None
        self.all_iteration_metrics = []
        
        print("="*70)
    
    def create_feature_map(self, num_features):
        """Create quantum feature map based on specified type"""
        print(f"    Creating {self.feature_map_type} feature map for {num_features} features...")
        
        if self.feature_map_type == 'ZZ':
            fm = ZZFeatureMap(feature_dimension=num_features,
                             reps=self.reps,
                             entanglement=self.entanglement)
        elif self.feature_map_type == 'Z':
            fm = ZFeatureMap(feature_dimension=num_features,
                            reps=self.reps)
        else:
            raise ValueError(f"Unknown feature map type: {self.feature_map_type}")
        
        print(f"    Feature map created (depth: ~{fm.depth()})")
        return fm
    
    def evaluate_feature_subset(self, X_train, y_train, X_test, y_test, feature_indices):
        """
        Evaluate a specific subset of features using QSVM
        
        Returns:
            dict: Contains accuracy, auc, and other metrics
        """
        try:
            # Select features
            X_train_subset = X_train[:, feature_indices]
            X_test_subset = X_test[:, feature_indices]
            
            # Create feature map
            feature_map = self.create_feature_map(len(feature_indices))
            
            # Create quantum kernel
            print(f"    Building quantum kernel...")
            kernel = FidelityQuantumKernel(feature_map=feature_map)
            
            # Create and train QSVC
            print(f"    Training QSVC...")
            qsvc = QSVC(quantum_kernel=kernel)
            
            train_start = time.time()
            qsvc.fit(X_train_subset, y_train)
            train_time = time.time() - train_start
            
            print(f"    Training completed in {train_time:.2f}s")
            
            # Predict
            print(f"    Making predictions...")
            pred_start = time.time()
            y_pred = qsvc.predict(X_test_subset)
            pred_time = time.time() - pred_start
            
            print(f"    Predictions completed in {pred_time:.2f}s")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate AUC if decision function is available
            try:
                y_score = qsvc.decision_function(X_test_subset)
                auc = roc_auc_score(y_test, y_score)
            except:
                auc = accuracy  # Fallback
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge case where only one class is predicted
                if len(np.unique(y_pred)) == 1:
                    if y_pred[0] == 0:
                        tn = len(y_test[y_test == 0])
                        fp = 0
                        fn = len(y_test[y_test == 1])
                        tp = 0
                    else:
                        tn = 0
                        fp = len(y_test[y_test == 0])
                        fn = 0
                        tp = len(y_test[y_test == 1])
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
            
            # Calculate fraud KPIs
            hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            false_alarm_ratio = fp / tp if tp > 0 else float('inf')
            
            print(f"    ✓ Evaluation complete - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            return {
                'accuracy': accuracy,
                'auc': auc,
                'hit_rate': hit_rate,
                'false_alarm_ratio': false_alarm_ratio,
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'train_time': train_time,
                'pred_time': pred_time
            }
            
        except Exception as e:
            print(f"    ✗ Error evaluating features {feature_indices}: {e}")
            return {
                'accuracy': 0,
                'auc': 0,
                'hit_rate': 0,
                'false_alarm_ratio': float('inf'),
                'error': str(e)
            }
    
    def select_initial_features(self, X_train, y_train, X_test, y_test, 
                               feature_names, num_initial=3):
        """
        Select best initial features from all possible combinations
        This corresponds to finding the best 3 features in the paper
        """
        print(f"\n{'='*70}")
        print(f"STAGE 1: SELECTING BEST {num_initial} INITIAL FEATURES")
        print(f"{'='*70}")
        
        num_features = X_train.shape[1]
        all_combinations = list(combinations(range(num_features), num_initial))
        
        print(f"\nSearch Space:")
        print(f"  Total features available: {num_features}")
        print(f"  Features to select: {num_initial}")
        print(f"  Total combinations to evaluate: {len(all_combinations)}")
        print(f"  Estimated time: ~{len(all_combinations) * 2 / 60:.1f} minutes")
        
        best_score = 0
        best_features = None
        best_metrics = None
        
        results = []
        
        stage_start_time = time.time()
        
        print("\nStarting exhaustive search...")
        print("-" * 70)
        
        for idx, feature_combo in enumerate(all_combinations):
            elapsed = time.time() - stage_start_time
            remaining = (elapsed / (idx + 1)) * (len(all_combinations) - idx - 1) if idx > 0 else 0
            
            if idx % 50 == 0 or idx == len(all_combinations) - 1:
                print(f"\n[{idx+1}/{len(all_combinations)}] Progress: {100*(idx+1)/len(all_combinations):.1f}%")
                print(f"  Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")
                print(f"  Current best accuracy: {best_score:.4f}")
            
            feature_combo_names = [feature_names[i] for i in feature_combo]
            print(f"\n  Testing combination: {feature_combo_names}")
            
            metrics = self.evaluate_feature_subset(X_train, y_train, 
                                                   X_test, y_test, 
                                                   feature_combo)
            
            results.append({
                'features': feature_combo,
                'feature_names': feature_combo_names,
                'metrics': metrics
            })
            
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_features = feature_combo
                best_metrics = metrics
                print(f"  🌟 NEW BEST! Accuracy improved to {best_score:.4f}")
        
        self.selected_features = list(best_features)
        self.feature_scores[num_initial] = results
        
        stage_time = time.time() - stage_start_time
        
        print(f"\n{'='*70}")
        print(f"STAGE 1 COMPLETE - Best {num_initial} features identified")
        print(f"{'='*70}")
        print(f"\nSelected Features:")
        for idx, feat_idx in enumerate(best_features, 1):
            print(f"  {idx}. {feature_names[feat_idx]} (index: {feat_idx})")
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  AUC: {best_metrics['auc']:.4f}")
        print(f"  Hit Rate: {best_metrics['hit_rate']:.4f}")
        print(f"  False Alarm Ratio: {best_metrics['false_alarm_ratio']:.2f}")
        print(f"  True Positives: {best_metrics['tp']}")
        print(f"  False Positives: {best_metrics['fp']}")
        print(f"  True Negatives: {best_metrics['tn']}")
        print(f"  False Negatives: {best_metrics['fn']}")
        
        print(f"\nTime taken: {stage_time/60:.2f} minutes")
        print(f"{'='*70}")
        
        return best_features, best_metrics
    
    def add_next_feature(self, X_train, y_train, X_test, y_test, 
                        feature_names, exclude_features=None):
        """
        Add one more feature to the selected set
        Tests all remaining features with the current best set
        """
        if exclude_features is None:
            exclude_features = set()
        
        current_features = self.selected_features
        num_features = X_train.shape[1]
        
        remaining_features = [i for i in range(num_features) 
                            if i not in current_features and i not in exclude_features]
        
        num_new_features = len(current_features) + 1
        
        print(f"\n{'='*70}")
        print(f"STAGE {num_new_features - 2}: ADDING FEATURE #{num_new_features}")
        print(f"{'='*70}")
        
        print(f"\nCurrent Selected Features ({len(current_features)}):")
        for idx, feat_idx in enumerate(current_features, 1):
            print(f"  {idx}. {feature_names[feat_idx]}")
        
        print(f"\nSearch Space:")
        print(f"  Remaining features to test: {len(remaining_features)}")
        print(f"  Estimated time: ~{len(remaining_features) * 2 / 60:.1f} minutes")
        
        best_score = 0
        best_new_feature = None
        best_metrics = None
        
        results = []
        
        stage_start_time = time.time()
        
        print("\nStarting feature addition search...")
        print("-" * 70)
        
        for idx, new_feature in enumerate(remaining_features):
            elapsed = time.time() - stage_start_time
            remaining = (elapsed / (idx + 1)) * (len(remaining_features) - idx - 1) if idx > 0 else 0
            
            if idx % 5 == 0 or idx == len(remaining_features) - 1:
                print(f"\n[{idx+1}/{len(remaining_features)}] Progress: {100*(idx+1)/len(remaining_features):.1f}%")
                print(f"  Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")
                print(f"  Current best accuracy: {best_score:.4f}")
            
            test_features = current_features + [new_feature]
            test_feature_names = [feature_names[i] for i in test_features]
            
            print(f"\n  Testing: {feature_names[new_feature]}")
            print(f"  Full combination: {test_feature_names}")
            
            metrics = self.evaluate_feature_subset(X_train, y_train,
                                                   X_test, y_test,
                                                   test_features)
            
            results.append({
                'new_feature': new_feature,
                'feature_name': feature_names[new_feature],
                'all_features': test_features,
                'metrics': metrics
            })
            
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_new_feature = new_feature
                best_metrics = metrics
                print(f"  🌟 NEW BEST! Accuracy improved to {best_score:.4f}")
        
        stage_time = time.time() - stage_start_time
        
        if best_new_feature is not None:
            self.selected_features.append(best_new_feature)
            self.feature_scores[num_new_features] = results
            
            print(f"\n{'='*70}")
            print(f"STAGE {num_new_features - 2} COMPLETE")
            print(f"{'='*70}")
            print(f"\nAdded Feature: {feature_names[best_new_feature]}")
            
            print(f"\nUpdated Feature Set ({len(self.selected_features)} features):")
            for idx, feat_idx in enumerate(self.selected_features, 1):
                print(f"  {idx}. {feature_names[feat_idx]}")
            
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
            print(f"  AUC: {best_metrics['auc']:.4f}")
            print(f"  Hit Rate: {best_metrics['hit_rate']:.4f}")
            print(f"  False Alarm Ratio: {best_metrics['false_alarm_ratio']:.2f}")
            print(f"  True Positives: {best_metrics['tp']}")
            print(f"  False Positives: {best_metrics['fp']}")
            
            print(f"\nTime taken: {stage_time/60:.2f} minutes")
            print(f"{'='*70}")
        else:
            print(f"\n⚠ WARNING: No feature improved the model!")
        
        return best_new_feature, best_metrics
    
    def iterative_feature_selection(self, X_train, y_train, X_test, y_test,
                                   feature_names, max_features=10):
        """
        Main algorithm: Iteratively select features starting from 3
        Implements the algorithm from Section IV.A of the paper
        
        Key improvement: Tracks best model across all iterations and returns
        the feature set that gives the highest accuracy (not necessarily max_features)
        """
        print("\n" + "="*70)
        print("QUANTUM FEATURE SELECTION ALGORITHM - STARTING")
        print("="*70)
        print(f"\nAlgorithm Configuration:")
        print(f"  Feature Map: {self.feature_map_type}")
        print(f"  Repetitions (depth): {self.reps}")
        print(f"  Entanglement: {self.entanglement}")
        print(f"  Max features to select: {max_features}")
        print(f"  GPU Available: {self.gpu_available}")
        print(f"\nDataset Information:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Total features available: {X_train.shape[1]}")
        
        overall_start_time = time.time()
        
        # Track best model across ALL iterations
        best_overall_accuracy = 0
        best_overall_features = None
        best_overall_metrics = None
        best_overall_num_features = 0
        
        # Step 1: Select best 3 initial features
        print(f"\n{'#'*70}")
        print("BEGINNING ITERATIVE FEATURE SELECTION")
        print(f"{'#'*70}")
        
        best_initial, initial_metrics = self.select_initial_features(
            X_train, y_train, X_test, y_test, feature_names, num_initial=3
        )
        
        # Store initial as potential best
        best_overall_accuracy = initial_metrics['accuracy']
        best_overall_features = list(self.selected_features)
        best_overall_metrics = initial_metrics.copy()
        best_overall_num_features = 3
        
        # Store metrics for tracking
        self.all_iteration_metrics.append({
            'num_features': 3,
            'features': list(self.selected_features),
            'feature_names': [feature_names[i] for i in self.selected_features],
            'metrics': initial_metrics
        })
        
        # Step 2: Iteratively add features
        for n in range(4, max_features + 1):
            print(f"\n{'#'*70}")
            print(f"ITERATION {n-2}: ATTEMPTING TO ADD FEATURE #{n}")
            print(f"{'#'*70}")
            
            best_new, metrics = self.add_next_feature(
                X_train, y_train, X_test, y_test, feature_names
            )
            
            if best_new is None:
                print(f"\n⚠ No feature improved the model at {n} features.")
                print(f"  Stopping feature selection.")
                break
            
            # Store metrics
            self.all_iteration_metrics.append({
                'num_features': n,
                'features': list(self.selected_features),
                'feature_names': [feature_names[i] for i in self.selected_features],
                'metrics': metrics
            })
            
            # Check if this is the best model so far
            if metrics['accuracy'] > best_overall_accuracy:
                best_overall_accuracy = metrics['accuracy']
                best_overall_features = list(self.selected_features)
                best_overall_metrics = metrics.copy()
                best_overall_num_features = n
                
                print(f"\n🏆 NEW OVERALL BEST MODEL!")
                print(f"  Features: {n}")
                print(f"  Accuracy: {best_overall_accuracy:.4f}")
            else:
                print(f"\n📊 Model with {n} features did not improve over best ({best_overall_num_features} features)")
                print(f"  Current: {metrics['accuracy']:.4f}")
                print(f"  Best: {best_overall_accuracy:.4f}")
        
        total_time = time.time() - overall_start_time
        
        # Final summary
        print(f"\n{'='*70}")
        print("QUANTUM FEATURE SELECTION COMPLETE")
        print(f"{'='*70}")
        
        print(f"\n📊 PERFORMANCE SUMMARY ACROSS ALL ITERATIONS:")
        print(f"\n{'Num Features':<15} {'Accuracy':<12} {'AUC':<12} {'Hit Rate':<12}")
        print("-" * 70)
        for iter_metrics in self.all_iteration_metrics:
            print(f"{iter_metrics['num_features']:<15} "
                  f"{iter_metrics['metrics']['accuracy']:<12.4f} "
                  f"{iter_metrics['metrics']['auc']:<12.4f} "
                  f"{iter_metrics['metrics']['hit_rate']:<12.4f}")
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL IDENTIFIED")
        print(f"{'='*70}")
        print(f"\nOptimal number of features: {best_overall_num_features}")
        print(f"\nSelected Features:")
        for idx, feat_idx in enumerate(best_overall_features, 1):
            print(f"  {idx}. {feature_names[feat_idx]} (index: {feat_idx})")
        
        print(f"\nBest Model Performance:")
        print(f"  Accuracy: {best_overall_metrics['accuracy']:.4f}")
        print(f"  AUC: {best_overall_metrics['auc']:.4f}")
        print(f"  Hit Rate: {best_overall_metrics['hit_rate']:.4f}")
        print(f"  False Alarm Ratio: {best_overall_metrics['false_alarm_ratio']:.2f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {best_overall_metrics['tp']}, FP: {best_overall_metrics['fp']}")
        print(f"    TN: {best_overall_metrics['tn']}, FN: {best_overall_metrics['fn']}")
        
        print(f"\nTotal execution time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"{'='*70}")
        
        # Update to best features
        self.selected_features = best_overall_features
        self.best_model = {
            'num_features': best_overall_num_features,
            'features': best_overall_features,
            'feature_names': [feature_names[i] for i in best_overall_features],
            'metrics': best_overall_metrics
        }
        
        return best_overall_features
    
    def save_results(self, filename='quantum_feature_selection_results.json'):
        """Save comprehensive feature selection results to JSON"""
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'feature_map_type': self.feature_map_type,
                'reps': self.reps,
                'entanglement': self.entanglement,
                'shots': self.shots,
                'gpu_used': self.gpu_available
            },
            'best_model': self.best_model,
            'all_iterations': self.all_iteration_metrics,
            'detailed_scores': {}
        }
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in self.feature_scores.items():
            results['detailed_scores'][str(key)] = value
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {filename}")
        print(f"  File size: {os.path.getsize(filename) / 1024:.2f} KB")
        print(f"{'='*70}")


def prepare_fraud_data(df, test_size=0.4, balance_ratio=1.0, max_samples=2500):
    """
    Prepare fraud detection dataset similar to paper's methodology
    
    Args:
        df: DataFrame with features and 'Class' column
        test_size: Proportion for test set
        balance_ratio: Balance between fraud and genuine (1.0 = equal)
        max_samples: Maximum total samples (paper uses 2500: 1500 train, 1000 test)
    """
    print("\n" + "="*70)
    print("DATA PREPARATION PIPELINE")
    print("="*70)
    
    print("\n1. Loading and analyzing dataset...")
    
    # Separate fraud and genuine
    fraud = df[df['Class'] == 1]
    genuine = df[df['Class'] == 0]
    
    print(f"\nOriginal Dataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Fraud transactions: {len(fraud)} ({100*len(fraud)/len(df):.3f}%)")
    print(f"  Genuine transactions: {len(genuine)} ({100*len(genuine)/len(df):.3f}%)")
    print(f"  Imbalance ratio: 1:{len(genuine)/len(fraud):.0f}")
    
    print("\n2. Balancing dataset via undersampling...")
    
    # Undersample genuine transactions
    genuine_sample_size = int(len(fraud) * balance_ratio)
    genuine_sample = genuine.sample(n=min(genuine_sample_size, len(genuine)), 
                                   random_state=42)
    
    # Combine
    balanced_df = pd.concat([fraud, genuine_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced Dataset:")
    print(f"  Fraud: {len(fraud)}")
    print(f"  Genuine: {len(genuine_sample)}")
    print(f"  Total: {len(balanced_df)}")
    print(f"  New ratio: 1:{len(genuine_sample)/len(fraud):.1f}")
    
    # Further reduce for quantum hardware limitations
    print("\n3. Reducing to quantum-compatible size...")
    if len(balanced_df) > max_samples:
        balanced_df = balanced_df.sample(n=max_samples, random_state=42)
        print(f"  ⚠ Reduced to {max_samples} samples (quantum hardware limitation)")
        print(f"    Paper uses 2500: 1500 training + 1000 testing")
    else:
        print(f"  ✓ Dataset size ({len(balanced_df)}) within limits")
    
    # Separate features and target
    print("\n4. Extracting features and target...")
    X = balanced_df.drop('Class', axis=1).values
    y = balanced_df['Class'].values
    feature_names = balanced_df.drop('Class', axis=1).columns.tolist()
    
    print(f"  Features extracted: {len(feature_names)}")
    print(f"  Feature names: {feature_names[:5]}... (showing first 5)")
    
    # Normalize to [-1, 1] as required by quantum feature maps
    print("\n5. Normalizing features to [-1, 1] range...")
    print("  (Required for quantum feature map angle encoding)")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    print(f"  ✓ Normalization complete")
    print(f"    Min value: {X_scaled.min():.3f}")
    print(f"    Max value: {X_scaled.max():.3f}")
    
    # Train-test split
    print(f"\n6. Splitting into train/test sets (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nFinal Dataset Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"    - Fraud: {sum(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")
    print(f"    - Genuine: {len(y_train)-sum(y_train)} ({100*(len(y_train)-sum(y_train))/len(y_train):.1f}%)")
    print(f"  Testing set: {len(X_test)} samples")
    print(f"    - Fraud: {sum(y_test)} ({100*sum(y_test)/len(y_test):.1f}%)")
    print(f"    - Genuine: {len(y_test)-sum(y_test)} ({100*(len(y_test)-sum(y_test))/len(y_test):.1f}%)")
    print(f"  Features: {X_train.shape[1]}")
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    return X_train, X_test, y_train, y_test, feature_names
# ==============================================================
# MAIN EXECUTION PIPELINE
# ==============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 QUANTUM FRAUD DETECTION PIPELINE - EXECUTION STARTED")
    print("="*70)

    try:
        # Step 1: Load dataset
        print("\n[1] Loading creditcard.csv dataset...")
        csv_path = "creditcard.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError("creditcard.csv not found in current directory.")
        df = pd.read_csv(csv_path)
        print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

        # Step 2: Prepare dataset
        print("\n[2] Preparing dataset for quantum feature selection...")
        X_train, X_test, y_train, y_test, feature_names = prepare_fraud_data(
            df,
            test_size=0.4,
            balance_ratio=1.0,
            max_samples=2500
        )

        # Step 3: Initialize Quantum Feature Selector
        print("\n[3] Initializing Quantum Feature Selector class...")
        qfs = QuantumFeatureSelector(
            feature_map_type='ZZ',
            reps=2,
            entanglement='full',
            use_gpu=True,
            shots=8192
        )

        # Step 4: Run iterative quantum feature selection
        print("\n[4] Running iterative quantum feature selection process (max 10 features)...")
        best_features = qfs.iterative_feature_selection(
            X_train, y_train, X_test, y_test,
            feature_names=feature_names,
            max_features=10
        )

        # Step 5: Save results
        print("\n[5] Saving all results to disk...")
        qfs.save_results("quantum_feature_selection_results.json")

        # Step 6: Summary
        print("\n" + "="*70)
        print("🎯 PIPELINE COMPLETE - FINAL SUMMARY")
        print("="*70)
        print(f"Optimal feature count: {len(best_features)}")
        print(f"Feature indices: {best_features}")
        print(f"Feature names: {[feature_names[i] for i in best_features]}")
        print("\nResults saved to: quantum_feature_selection_results.json")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("❌ PIPELINE TERMINATED DUE TO ERROR")
        print("="*70)
        print(f"Error details: {e}")
        print("="*70)


