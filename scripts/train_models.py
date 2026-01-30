#!/usr/bin/env python3
"""
Train Fast ML Models for PromptShield

Models trained:
- L5: Single XGBoost (primary, fast)
- L7: Ensemble (XGBoost + RandomForest + SVM)

Target latency: L5 <5ms, L7 <15ms
"""

import pickle
import json
import time
from pathlib import Path
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")


def load_features(filepath: str):
    """Load extracted features"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data.get('metadata', [])


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost classifier (L5 primary model)
    """
    
    if not HAS_XGBOOST:
        return None
    
    print("\n[XGBoost] Training...")
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',  # Fast histogram-based
        'eval_metric': 'logloss'
    }
    
    model = xgb.XGBClassifier(**params)
    
    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - start
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    results = {
        'model': model,
        'name': 'XGBoost',
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
    }
    
    print(f"  ✅ Trained in {train_time:.2f}s")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    return results


def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Train LightGBM classifier (faster alternative)
    """
    
    if not HAS_LIGHTGBM:
        return None
    
    print("\n[LightGBM] Training...")
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    
    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    train_time = time.time() - start
    
    # Evaluate
    y_pred = model.predict(X_val)
    
    results = {
        'model': model,
        'name': 'LightGBM',
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
    }
    
    print(f"  ✅ Trained in {train_time:.2f}s")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    return results


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest (L7 ensemble member)
    """
    
    print("\n[Random Forest] Training...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all cores
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Evaluate
    y_pred = model.predict(X_val)
    
    results = {
        'model': model,
        'name': 'RandomForest',
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
    }
    
    print(f"  ✅ Trained in {train_time:.2f}s")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    return results


def train_svm(X_train, y_train, X_val, y_val):
    """
    Train SVM (L7 ensemble member)
    """
    
    print("\n[SVM] Training...")
    
    # Use subset for faster training
    subset_size = min(5000, len(X_train))
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,  # Enable probability estimates
        random_state=42
    )
    
    start = time.time()
    model.fit(X_train[indices], y_train[indices])
    train_time = time.time() - start
    
    # Evaluate
    y_pred = model.predict(X_val)
    
    results = {
        'model': model,
        'name': 'SVM',
        'train_time': train_time,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
    }
    
    print(f"  ✅ Trained in {train_time:.2f}s (on {subset_size} samples)")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    return results


def create_ensemble(models, X_val, y_val):
    """
    Create voting ensemble from multiple models
    """
    
    print("\n[Ensemble] Creating voting ensemble...")
    
    # Get predictions from all models
    predictions = []
    probabilities = []
    
    for result in models:
        model = result['model']
        pred = model.predict(X_val)
        predictions.append(pred)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_val)[:, 1]
        else:
            prob = pred.astype(float)
        
        probabilities.append(prob)
    
    # Soft voting (average probabilities)
    ensemble_prob = np.mean(probabilities, axis=0)
    ensemble_pred = (ensemble_prob > 0.5).astype(int)
    
    results = {
        'name': 'Ensemble',
        'accuracy': accuracy_score(y_val, ensemble_pred),
        'precision': precision_score(y_val, ensemble_pred),
        'recall': recall_score(y_val, ensemble_pred),
        'f1': f1_score(y_val, ensemble_pred),
        'models': [r['name'] for r in models]
    }
    
    print(f"  ✅ Ensemble created from: {', '.join(results['models'])}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    return results


def benchmark_inference_speed(model, X_test, num_samples=1000):
    """
    Benchmark model inference speed
    """
    
    # Take subset for benchmarking
    X_subset = X_test[:num_samples]
    
    # Warmup
    for _ in range(10):
        _ = model.predict(X_subset[:10])
    
    # Benchmark
    times = []
    for i in range(num_samples):
        start = time.perf_counter()
        _ = model.predict([X_subset[i]])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
    }


def evaluate_on_test_set(model, X_test, y_test, model_name):
    """
    Final evaluation on test set
    """
    
    print(f"\n[{model_name}] Evaluating on test set...")
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Benchmark speed
    print("\nInference Speed Benchmark:")
    speed = benchmark_inference_speed(model, X_test)
    print(f"  Mean: {speed['mean_ms']:.2f}ms")
    print(f"  Median: {speed['median_ms']:.2f}ms")
    print(f"  P95: {speed['p95_ms']:.2f}ms")
    print(f"  P99: {speed['p99_ms']:.2f}ms")
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': cm.tolist(),
        'latency': speed
    }


def main():
    """
    Main training pipeline
    """
    
    print("=" * 60)
    print("PromptShield ML Model Training")
    print("=" * 60)
    
    data_dir = Path("../promptshield/ml_data")
    models_dir = Path("../promptshield/models")
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("\nLoading datasets...")
    X_train, y_train, _ = load_features(data_dir / "train_features.pkl")
    X_val, y_val, _ = load_features(data_dir / "val_features.pkl")
    X_test, y_test, _ = load_features(data_dir / "test_features.pkl")
    
    print(f"✅ Loaded datasets")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # Train models
    trained_models = []
    
    # L5: Primary model (XGBoost or LightGBM)
    if HAS_XGBOOST:
        xgb_results = train_xgboost(X_train, y_train, X_val, y_val)
        if xgb_results:
            trained_models.append(xgb_results)
    
    if HAS_LIGHTGBM:
        lgb_results = train_lightgbm(X_train, y_train, X_val, y_val)
        if lgb_results:
            trained_models.append(lgb_results)
    
    # L7: Ensemble members
    rf_results = train_random_forest(X_train, y_train, X_val, y_val)
    trained_models.append(rf_results)
    
    svm_results = train_svm(X_train, y_train, X_val, y_val)
    trained_models.append(svm_results)
    
    # Create ensemble
    ensemble_results = create_ensemble(trained_models, X_val, y_val)
    
    # Select best L5 model
    print("\n" + "=" * 60)
    print("Model Selection")
    print("=" * 60)
    
    # Best single model for L5
    best_l5 = max(trained_models, key=lambda x: x['f1'])
    print(f"\n✅ Best L5 Model: {best_l5['name']}")
    print(f"   F1 Score: {best_l5['f1']:.4f}")
    
    # Save L5 model
    l5_path = models_dir / "l5_model.pkl"
    with open(l5_path, 'wb') as f:
        pickle.dump({
            'model': best_l5['model'],
            'name': best_l5['name'],
            'metrics': {k: v for k, v in best_l5.items() if k != 'model'}
        }, f)
    print(f"   Saved to: {l5_path}")
    
    # Save all models for L7 ensemble
    l7_path = models_dir / "l7_ensemble.pkl"
    with open(l7_path, 'wb') as f:
        pickle.dump({
            'models': [r['model'] for r in trained_models],
            'model_names': [r['name'] for r in trained_models],
            'ensemble_metrics': ensemble_results
        }, f)
    print(f"\n✅ L7 Ensemble saved to: {l7_path}")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation (Test Set)")
    print("=" * 60)
    
    test_results = evaluate_on_test_set(best_l5['model'], X_test, y_test, best_l5['name'])
    
    # Save results
    results_summary = {
        'l5_model': best_l5['name'],
        'l5_test_metrics': test_results,
        'l7_ensemble': ensemble_results,
        'all_models': [{k: v for k, v in r.items() if k != 'model'} for r in trained_models]
    }
    
    results_path = models_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"L5 Model: {best_l5['name']}")
    print(f"  - Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"  - Test F1: {test_results['f1']:.4f}")
    print(f"  - Latency (P99): {test_results['latency']['p99_ms']:.2f}ms")
    print(f"\nL7 Ensemble: {len(trained_models)} models")
    print(f"  - Val Accuracy: {ensemble_results['accuracy']:.4f}")
    print(f"  - Val F1: {ensemble_results['f1']:.4f}")


if __name__ == "__main__":
    main()
