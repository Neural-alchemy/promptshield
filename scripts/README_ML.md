# PromptShield ML Pipeline

## 🎯 Overview

This pipeline generates training data from **PromptXploit attacks** and trains fast ML models for PromptShield's L5 and L7 security levels.

**Key Features:**
- Uses your existing PromptXploit attacks as malicious samples
- Generates balanced benign samples and edge cases
- Extracts ~150 lightweight features (<1ms)
- Trains fast models (XGBoost, LightGBM, RandomForest, SVM)
- Target latency: L5 <5ms, L7 <15ms

---

## 📊 Dataset Details

### Size Recommendations

| Tier | Malicious | Benign | Edge Cases | Total | Use Case |
|------|-----------|--------|------------|-------|----------|
| **MVP** | 3,500 | 7,000 | 500 | **11,000** | Initial launch |
| **Production** | 10,000 | 20,000 | 2,000 | **32,000** | Production ready |
| **Enterprise** | 25,000+ | 50,000+ | 5,000+ | **80,000+** | Continuous improvement |

### Data Sources

**Malicious Samples:**
- ✅ PromptXploit attacks (automatic)
- ✅ 17 attack categories
- ✅ Real-world attack patterns

**Benign Samples:**
- ✅ Synthetic normal prompts (automatic)
- 🔄 **Optional:** HuggingFace datasets (ShareGPT, Alpaca, etc.)

**Edge Cases:**
- ✅ Prompts that look suspicious but are benign
- ✅ Critical for reducing false positives

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd promptshield/scripts
pip install -r requirements_ml.txt
```

**Required:**
- `numpy`, `scikit-learn`
- `xgboost` OR `lightgbm` (choose at least one)

### 2. Generate Dataset

```bash
python generate_ml_dataset.py
```

**Output:**
```
ml_data/
├── train.jsonl          # Training set (70%)
├── val.jsonl            # Validation set (15%)
├── test.jsonl           # Test set (15%)
├── full_dataset.jsonl   # Complete dataset
└── dataset_stats.json   # Statistics
```

**Expected output:**
```
Loaded 147+ malicious samples from PromptXploit
Generated 294+ benign samples
Generated 44+ edge cases
Total: ~485 samples (will be more if you have more attacks)
```

### 3. Extract Features

```bash
python extract_features.py
```

**Output:**
```
ml_data/
├── train_features.pkl   # ~150 features per sample
├── val_features.pkl
└── test_features.pkl
```

**Features extracted:**
- Statistical: length, entropy, character ratios
- Lexical: keywords, patterns, n-grams
- Behavioral: repetitions, encoding tricks
- TF-IDF: top 100 terms

### 4. Train Models

```bash
python train_models.py
```

**Output:**
```
models/
├── l5_model.pkl         # Best single model (XGBoost or LightGBM)
├── l7_ensemble.pkl      # Ensemble of all models
└── training_results.json # Metrics and benchmarks
```

**Models trained:**
- XGBoost (primary L5)
- LightGBM (alternative L5)
- RandomForest (L7 ensemble)
- SVM (L7 ensemble)

---

## 📈 Expected Performance

### Accuracy Targets

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **L5 (XGBoost)** | 92-95% | 93-96% | 90-94% | 92-95% |
| **L7 (Ensemble)** | 95-98% | 96-99% | 94-97% | 95-98% |

### Latency Targets

| Model | Mean | P95 | P99 |
|-------|------|-----|-----|
| **L5** | 2-3ms | 4-5ms | 6-8ms |
| **L7** | 8-12ms | 14-16ms | 18-22ms |

---

## 🔧 How It Works

### 1. Dataset Generation

```python
# Automatically loads from PromptXploit
malicious = load_promptxploit_attacks("../promptxploit/attacks")
# Returns: [{"prompt": "...", "label": 1, "attack_type": "jailbreak", ...}]

# Generates synthetic benign samples
benign = generate_benign_samples(count=7000)

# Generates edge cases (critical!)
edge_cases = generate_edge_cases(count=500)
```

**Attack categories loaded:**
- Jailbreak, Prompt Injection, PII Extraction
- System Manipulation, Encoding Tricks
- Context Confusion, Agent Manipulation
- RAG Poisoning, Multi-turn Attacks
- And more...

### 2. Feature Extraction

**Fast statistical features (~50 features):**
```python
features = {
    'length': len(prompt),
    'entropy': calculate_entropy(prompt),
    'special_char_ratio': count_special_chars(prompt),
    'has_system_keywords': check_keywords(prompt, SYSTEM_WORDS),
    'has_jailbreak_phrases': check_jailbreak_patterns(prompt),
    # ... ~45 more features
}
```

**TF-IDF features (~100 features):**
- Top 100 most important terms
- Unigrams and bigrams
- Fast lookup (<1ms)

**Total: ~150 features, extracted in <1ms**

### 3. Model Training

**L5: Single Model**
```python
# Primary: XGBoost (best balance of speed/accuracy)
xgb_model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)

# Alternative: LightGBM (faster training)
lgb_model = LGBMClassifier(num_leaves=31, n_estimators=100)
```

**L7: Ensemble**
```python
# Soft voting ensemble
ensemble_pred = average([
    xgboost.predict_proba(X),
    random_forest.predict_proba(X),
    svm.predict_proba(X)
])
```

---

## 🎯 Model Selection

### How Many Models to Train?

**Recommended: 4 models** (1 for L5, 3 for L7 ensemble)

| Level | Models | When to Use | Latency |
|-------|--------|-------------|---------|
| **L5** | 1 model (XGBoost) | Production default | <5ms |
| **L7** | 3-model ensemble | High-security apps | <15ms |

### Which Models?

**For L5 (pick one):**
- ✅ **XGBoost**: Best accuracy, most popular
- ✅ **LightGBM**: Faster training, similar accuracy

**For L7 Ensemble:**
- ✅ XGBoost or LightGBM
- ✅ Random Forest
- ✅ SVM (trained on subset for speed)

**Don't use:**
- ❌ Transformers (too slow: 50-200ms)
- ❌ LSTMs/RNNs (slow, complex)
- ❌ Multiple XGBoost variants (redundant)

---

## 📊 Dataset Size Guidelines

### Minimum (MVP)
```
11,000 samples
├── Malicious: 3,500 (from PromptXploit)
├── Benign: 7,000 (synthetic)
└── Edge cases: 500
```

✅ Good enough for initial launch  
✅ Can train in ~10 minutes  
⚠️ May have higher false positives

### Recommended (Production)
```
32,000 samples
├── Malicious: 10,000
│   ├── PromptXploit: 3,500
│   └── Real attacks: 6,500 (collect from logs)
├── Benign: 20,000
│   ├── Synthetic: 10,000
│   └── Real traffic: 10,000
└── Edge cases: 2,000
```

✅ Production-ready accuracy  
✅ Lower false positives  
✅ Trains in ~30 minutes

### Enterprise (Continuous)
```
80,000+ samples (continuous collection)
```

✅ Best accuracy  
✅ Adapts to new attacks  
✅ Requires retraining pipeline

---

## 🔄 Continuous Improvement

### Weekly Updates
```bash
# 1. Add new PromptXploit attacks
cd ../promptxploit
# Add new attack JSON files

# 2. Regenerate dataset
cd ../promptshield/scripts
python generate_ml_dataset.py

# 3. Retrain models
python extract_features.py
python train_models.py
```

### Collect Real Data
```python
# Log false positives/negatives from production
{
  "prompt": "...",
  "predicted": 0,
  "actual": 1,  # Wrong!
  "timestamp": "..."
}

# Add to training data weekly
```

---

## 🎯 Integration with PromptShield

After training, integrate into PromptShield:

```python
from promptshield import Shield
import pickle

# Load trained model
with open('models/l5_model.pkl', 'rb') as f:
    ml_model = pickle.load(f)

# Use in L5
shield = Shield(level=5)
result = shield.protect_input(
    user_input="Your prompt",
    ml_model=ml_model  # Pass trained model
)
```

---

## ✅ Success Checklist

- [ ] PromptXploit has 100+ attack samples
- [ ] Ran `generate_ml_dataset.py` successfully
- [ ] Generated 10K+ samples total
- [ ] Ran `extract_features.py`
- [ ] Installed `xgboost` or `lightgbm`
- [ ] Trained models with `train_models.py`
- [ ] L5 model achieves >90% F1 score
- [ ] L5 latency is <10ms (P99)
- [ ] Integrated into PromptShield shields.py

---

## 📝 Next Steps

1. **Generate more PromptXploit attacks** (target: 500+)
2. **Collect real benign traffic** from your apps
3. **Run the full pipeline** (dataset → features → train)
4. **Benchmark performance** on test set
5. **Integrate L5 into PromptShield**
6. **A/B test** vs pattern-only approach
7. **Set up weekly retraining** pipeline

---

## 🚨 Troubleshooting

### "No PromptXploit attacks found"
- Check path in `generate_ml_dataset.py`
- Default: `../promptxploit/attacks`
- Ensure attack JSON files exist

### "ImportError: No module named xgboost"
```bash
pip install xgboost
# OR
pip install lightgbm
```

### "Low accuracy (<80%)"
- Need more samples (target: 10K+)
- Check class balance (should be ~2:1 benign:malicious)
- Verify edge cases are labeled correctly

### "Latency too high (>20ms)"
- Use XGBoost instead of SVM
- Reduce `max_depth` in XGBoost
- Use fewer features (top 100 only)

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

**Questions?** Check the training output logs or dataset stats for debugging.
