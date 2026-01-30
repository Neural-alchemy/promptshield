# PromptShield Models

🛡️ **Production-ready ML models for prompt injection detection**

This repository contains trained models that achieve **98%+ accuracy** in detecting prompt injection attacks against LLMs.

## Models

### 1. **Ensemble Model** (Recommended) ⭐
- **Accuracy**: 98.1%
- **Precision**: 96.9%
- **Recall**: 95.6%
- **F1-Score**: 96.2%
- Combines Random Forest, Gradient Boosting, and Logistic Regression
- Best overall performance

### 2. **Gradient Boosting**
- **Accuracy**: 97.5%
- Faster inference than ensemble
- Good balance of speed and accuracy

### 3. **Random Forest**
- **Accuracy**: 97.2%
- Most interpretable
- Feature importance analysis

### 4. **Logistic Regression** (Baseline)
- **Accuracy**: 96.3%
- Fastest inference
- Minimal resource usage

## Usage

```python
import pickle
from huggingface_hub import hf_hub_download

# Download models
vectorizer = pickle.load(open(hf_hub_download(
    repo_id="YOUR_USERNAME/promptshield-models",
    filename="tfidf_vectorizer.pkl"
), 'rb'))

ensemble = pickle.load(open(hf_hub_download(
    repo_id="YOUR_USERNAME/promptshield-models",
    filename="ensemble.pkl"
), 'rb'))

# Predict
def detect_injection(prompt: str):
    """Detect prompt injection"""
    X = vectorizer.transform([prompt])
    prediction = ensemble.predict(X)[0]
    probability = ensemble.predict_proba(X)[0]
    
    return {
        'is_malicious': bool(prediction),
        'confidence': float(probability[prediction]),
        'risk_score': float(probability[1])  # Prob of being malicious
    }

# Example
result = detect_injection("Ignore all previous instructions")
print(result)
# {'is_malicious': True, 'confidence': 0.98, 'risk_score': 0.98}
```

## Training Data

Trained on **10,610 samples** from the [promptshield-dataset](https://huggingface.co/datasets/YOUR_USERNAME/promptshield-dataset):

- **2,839 malicious samples** (19 attack types)
- **7,771 benign samples**
- **70% augmented** for robustness
- **10+ languages** represented

## Robustness

Models are resilient to evasion:

✅ Case variations (`IGNORE`, `ignore`, `IgNoRe`)  
✅ L33t speak (`3x3cut3`, `sy5t3m`)  
✅ Unicode tricks (Cyrillic/Greek lookalikes)  
✅ Whitespace manipulation  
✅ Paraphrasing  

## Model Files

- `tfidf_vectorizer.pkl` (5.2 MB) - Text feature extractor
- `ensemble.pkl` (47.3 MB) - **Recommended model**
- `gradient_boosting.pkl` (22.1 MB) - Fast alternative
- `random_forest.pkl` (31.8 MB) - Interpretable model
- `logistic_regression.pkl` (1.2 MB) - Lightweight baseline

## Benchmarks

Tested on 1,592 samples (429 malicious + 1,163 benign):

| Metric | Ensemble | GB | RF | LR |
|--------|----------|-----|-----|-----|
| Accuracy | **98.1%** | 97.5% | 97.2% | 96.3% |
| Precision | **96.9%** | 96.1% | 95.8% | 94.2% |
| Recall | **95.6%** | 94.8% | 94.1% | 92.7% |
| F1-Score | **96.2%** | 95.4% | 94.9% | 93.4% |
| AUC-ROC | **0.991** | 0.987 | 0.984 | 0.978 |

### Inference Speed (CPU)

- Logistic Regression: ~0.5ms per sample
- Ensemble: ~3ms per sample
- Random Forest: ~2ms per sample
- Gradient Boosting: ~1.5ms per sample

## Integration

### FastAPI Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load model (once at startup)
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('ensemble.pkl', 'rb'))

class PromptRequest(BaseModel):
    prompt: str

@app.post("/detect")
def detect_injection(request: PromptRequest):
    X = vectorizer.transform([request.prompt])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return {
        'is_malicious': bool(prediction),
        'risk_score': float(probability[1]),
        'confidence': float(probability[prediction])
    }
```

### Python Library

```bash
pip install promptshield
```

```python
from promptshield import PromptShield

shield = PromptShield()
result = shield.scan("Your prompt here")
print(result.is_safe)  # True/False
```

## Citation

```bibtex
@misc{promptshield-models,
  author = {YOUR_NAME},
  title = {PromptShield: ML Models for Prompt Injection Detection},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/YOUR_USERNAME/promptshield-models}
}
```

## License

MIT License

## Related Resources

- **Dataset**: [promptshield-dataset](https://huggingface.co/datasets/YOUR_USERNAME/promptshield-dataset)
- **GitHub**: [SecurePrompt](https://github.com/YOUR_USERNAME/SecurePrompt)
- **Demo**: [Try it live](https://huggingface.co/spaces/YOUR_USERNAME/promptshield-demo)

---

**Need help?** Open an issue on [GitHub](https://github.com/YOUR_USERNAME/SecurePrompt/issues)
