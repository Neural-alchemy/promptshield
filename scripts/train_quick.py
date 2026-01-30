#!/usr/bin/env python3
"""
Quick ML training script for PromptShield
Trains 4 models on augmented dataset
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import time

print("=" * 70)
print("🛡️ PromptShield Model Training")
print("=" * 70)

# Paths
data_dir = Path("../ml_data")
models_dir = Path("../models")
models_dir.mkdir(exist_ok=True)

# Load data
print("\n[1/5] Loading augmented dataset...")
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train = load_jsonl(data_dir / "train_augmented.jsonl")
val = load_jsonl(data_dir / "val_augmented.jsonl")
test = load_jsonl(data_dir / "test_augmented.jsonl")

print(f"  Train: {len(train)} samples")
print(f"  Val:   {len(val)} samples")
print(f"  Test:  {len(test)} samples")

# Extract features
print("\n[2/5] Extracting TF-IDF features...")
X_train = [s['prompt'] for s in train]
y_train = [s['label'] for s in train]

X_val = [s['prompt'] for s in val]
y_val = [s['label'] for s in val]

X_test = [s['prompt'] for s in test]
y_test = [s['label'] for s in test]

print(f"  Malicious in train: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

print(f"  ✅ Feature matrix: {X_train_vec.shape}")

# Save vectorizer
with open(models_dir / "tfidf_vectorizer.pkl", 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"  ✅ Saved vectorizer")

# Train models
print("\n[3/5] Training models...")
models = {}

# Random Forest
print("\n  Training Random Forest...")
start = time.time()
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_vec, y_train)
rf_time = time.time() - start
print(f"    ✅ Trained in {rf_time:.1f}s - Val Acc: {rf.score(X_val_vec, y_val):.4f}")
models['random_forest'] = rf

# Gradient Boosting
print("\n  Training Gradient Boosting...")
start = time.time()
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train_vec, y_train)
gb_time = time.time() - start
print(f"    ✅ Trained in {gb_time:.1f}s - Val Acc: {gb.score(X_val_vec, y_val):.4f}")
models['gradient_boosting'] = gb

# Logistic Regression
print("\n  Training Logistic Regression...")
start = time.time()
lr = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)
lr.fit(X_train_vec, y_train)
lr_time = time.time() - start
print(f"    ✅ Trained in {lr_time:.1f}s - Val Acc: {lr.score(X_val_vec, y_val):.4f}")
models['logistic_regression'] = lr

# Ensemble
print("\n  Creating Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('lr', lr)
    ],
    voting='soft'
)
ensemble.fit(X_train_vec, y_train)
print(f"    ✅ Ensemble ready - Val Acc: {ensemble.score(X_val_vec, y_val):.4f}")
models['ensemble'] = ensemble

# Evaluate
print("\n[4/5] Evaluating on test set...")
print("\n" + "=" * 70)

for name, model in models.items():
    print(f"\n{name.upper().replace('_', ' ')}")
    print("-" * 70)
    
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

# Save models
print("\n[5/5] Saving models...")
for name, model in models.items():
    path = models_dir / f"{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ {name}.pkl")

print("\n" + "=" * 70)
print("✅ Training Complete!")
print("=" * 70)
print(f"\nModels saved to: {models_dir.absolute()}")
print(f"\nTotal training time: {rf_time + gb_time + lr_time:.1f}s")
print("\n🎉 Ready to upload to HuggingFace!")
