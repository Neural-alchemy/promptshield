"""
Train models on expanded dataset with GitHub attacks
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from datetime import datetime

def load_data(filepath: Path):
    """Load JSONL dataset"""
    prompts = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                prompts.append(item['prompt'])
                labels.append(item['label'])
    
    return prompts, labels

def main():
    print("🚀 Training Models on Expanded Dataset")
    print("=" * 60)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "ml_data" / "full_dataset_expanded_augmented.jsonl"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"\n📊 Loading dataset from {data_file}...")
    X, y = load_data(data_file)
    print(f"   Total samples: {len(X):,}")
    print(f"   Malicious: {sum(y):,}")
    print(f"   Benign: {len(y) - sum(y):,}")
    
    # Split data
    print(f"\n✂️  Splitting data (70% train, 15% val, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Validation: {len(X_val):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    # Vectorization
    print(f"\n🔤 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    print(f"   Feature dimensions: {X_train_vec.shape[1]}")
    
    # Save vectorizer
    vectorizer_path = models_dir / "tfidf_vectorizer_expanded.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"   Saved vectorizer to {vectorizer_path}")
    
    # Define models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    results = []
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🤖 Training {name}...")
        print(f"{'='*60}")
        
        # Train
        model.fit(X_train_vec, y_train)
        
        # Predict on validation set
        y_val_pred = model.predict(X_val_vec)
        
        # Metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, zero_division=0)
        recall = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        print(f"\n📊 Validation Results:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1 Score:  {f1*100:.2f}%")
        
        # Test set evaluation
        y_test_pred = model.predict(X_test_vec)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy:  {test_accuracy*100:.2f}%")
        print(f"   Precision: {test_precision*100:.2f}%")
        print(f"   Recall:    {test_recall*100:.2f}%")
        print(f"   F1 Score:  {test_f1*100:.2f}%")
        
        # Save model
        model_filename = name.lower().replace(' ', '_') + '_expanded.pkl'
        model_path = models_dir / model_filename
        joblib.dump(model, model_path)
        print(f"\n💾 Saved model to {model_path}")
        
        # Store results
        results.append({
            'model': name,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'model_path': str(model_path)
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Model Comparison (Test Set):\n")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 68)
    for r in results:
        print(f"{r['model']:<20} {r['test_accuracy']*100:>10.2f}%  {r['test_precision']*100:>10.2f}%  {r['test_recall']*100:>10.2f}%  {r['test_f1']*100:>10.2f}%")
    
    # Save results to JSON
    results_file = models_dir / f"training_results_expanded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(X),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
