"""
Overfitting Analysis - Check if models are truly generalizing
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

def load_data(filepath: Path):
    """Load JSONL dataset"""
    prompts = []
    labels = []
    sources = []
    augmented = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                prompts.append(item['prompt'])
                labels.append(item['label'])
                sources.append(item.get('source', 'unknown'))
                augmented.append(item.get('augmented', False))
    
    return prompts, labels, sources, augmented

def check_data_leakage(X, y, sources, augmented):
    """Check for potential data leakage issues"""
    print("\n🔍 CHECKING FOR DATA LEAKAGE")
    print("=" * 60)
    
    # Check augmented data distribution
    total_augmented = sum(augmented)
    total_samples = len(augmented)
    
    print(f"\n📊 Augmented Data:")
    print(f"   Total augmented: {total_augmented:,} ({total_augmented/total_samples*100:.1f}%)")
    print(f"   Total original: {total_samples - total_augmented:,}")
    
    # Check for duplicate prompts
    unique_prompts = len(set(X))
    print(f"\n🔄 Duplicate Check:")
    print(f"   Total prompts: {len(X):,}")
    print(f"   Unique prompts: {unique_prompts:,}")
    print(f"   Duplicates: {len(X) - unique_prompts:,}")
    
    if len(X) - unique_prompts > 0:
        print(f"   ⚠️  WARNING: Found duplicates!")
    else:
        print(f"   ✅ No duplicates found")
    
    # Check source distribution
    source_counts = {}
    for s in sources:
        source_counts[s] = source_counts.get(s, 0) + 1
    
    print(f"\n📦 Data Sources:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {source:30} {count:5,} ({count/total_samples*100:5.1f}%)")

def perform_cross_validation(X, y):
    """Perform k-fold cross-validation"""
    print("\n🔀 K-FOLD CROSS-VALIDATION (5-FOLD)")
    print("=" * 60)
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    X_vec = vectorizer.fit_transform(X)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        scores = cross_val_score(model, X_vec, y, cv=5, scoring='accuracy', n_jobs=-1)
        print(f"   Fold scores: {[f'{s*100:.2f}%' for s in scores]}")
        print(f"   Mean: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")
        
        # Check variance
        if scores.std() > 0.05:
            print(f"   ⚠️  HIGH VARIANCE: Model might be overfitting!")
        elif scores.mean() < 0.95:
            print(f"   ⚠️  LOW ACCURACY: Model might be underfitting!")
        else:
            print(f"   ✅ Good generalization")

def test_on_novel_attacks(X, y, sources, augmented):
    """Test on truly novel attacks (not augmented, from specific sources)"""
    print("\n🆕 NOVEL ATTACK TESTING")
    print("=" * 60)
    
    # Split: Train on augmented + PromptXploit, test on GitHub original attacks
    train_idx = []
    test_idx = []
    
    for i, (src, aug) in enumerate(zip(sources, augmented)):
        # Test on original GitHub attacks only
        if 'github_' in src and not aug and y[i] == 1:
            test_idx.append(i)
        else:
            train_idx.append(i)
    
    if len(test_idx) > 0:
        print(f"\n📊 Split:")
        print(f"   Training samples: {len(train_idx):,}")
        print(f"   Test samples (novel GitHub attacks): {len(test_idx):,}")
        
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        
        # Train and test
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_vec, y_train)
        
        accuracy = model.score(X_test_vec, y_test)
        print(f"\n🎯 Performance on Novel Attacks:")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        
        if accuracy < 0.9:
            print(f"   ⚠️  WARNING: Poor performance on novel data!")
            print(f"   This suggests overfitting to training distribution.")
        else:
            print(f"   ✅ Good generalization to novel attacks")
    else:
        print("   ℹ️  No novel GitHub attacks found for testing")

def analyze_learning_curves(X, y):
    """Generate learning curves to detect overfitting"""
    print("\n📈 LEARNING CURVE ANALYSIS")
    print("=" * 60)
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    X_vec = vectorizer.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_vec, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    
    print(f"\n📊 Training vs Validation Performance:")
    print(f"   {'Size':>8} {'Train Acc':>12} {'Val Acc':>12} {'Gap':>10}")
    print("   " + "-" * 50)
    
    for size, train_acc, val_acc in zip(train_sizes, train_mean, val_mean):
        gap = train_acc - val_acc
        print(f"   {int(size):>8} {train_acc*100:>11.2f}% {val_acc*100:>11.2f}% {gap*100:>9.2f}%")
    
    final_gap = train_mean[-1] - val_mean[-1]
    print(f"\n🎯 Final Train-Val Gap: {final_gap*100:.2f}%")
    
    if final_gap > 0.05:
        print(f"   ⚠️  OVERFITTING DETECTED: Large gap between train and validation")
    elif final_gap < 0.01:
        print(f"   ✅ Excellent generalization: Minimal gap")
    else:
        print(f"   ✅ Good generalization: Acceptable gap")

def main():
    print("🔬 OVERFITTING ANALYSIS")
    print("=" * 60)
    
    # Load data
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "ml_data" / "full_dataset_expanded_augmented.jsonl"
    
    print(f"\n📊 Loading {data_file}...")
    X, y, sources, augmented = load_data(data_file)
    print(f"   Loaded {len(X):,} samples")
    
    # Run all checks
    check_data_leakage(X, y, sources, augmented)
    perform_cross_validation(X, y)
    test_on_novel_attacks(X, y, sources, augmented)
    analyze_learning_curves(X, y)
    
    # Final verdict
    print("\n" + "=" * 60)
    print("📋 OVERFITTING ASSESSMENT")
    print("=" * 60)
    print("""
Key Indicators:
✅ Cross-validation will show if model generalizes to different folds
✅ Novel attack testing shows performance on unseen attack patterns
✅ Learning curves reveal train-test gap
⚠️  High augmentation ratio might inflate scores
⚠️  100% accuracy is suspicious - check for leakage

Recommendations:
1. Review cross-validation variance
2. Test on completely external datasets
3. Consider reducing augmentation ratio
4. Monitor real-world deployment performance
    """)
    print("=" * 60)

if __name__ == "__main__":
    main()
