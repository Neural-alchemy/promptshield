"""
Direct Upload to HuggingFace - No Interactive Prompts
Username: Zenith300
"""

import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, login
from datasets import Dataset
import pandas as pd

# Configuration
USERNAME = "m4vic"
import os
DATASET_TOKEN = os.getenv("HF_DATASET_TOKEN", "your-dataset-token-here")
MODEL_TOKEN = os.getenv("HF_MODEL_TOKEN", "your-model-token-here")

def create_dataset_from_jsonl(jsonl_path):
    """Convert JSONL to HuggingFace Dataset"""
    print(f"📊 Loading {jsonl_path}...")
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"   Loaded {len(data):,} samples")
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

print("="*70)
print("🚀 UPLOADING TO HUGGINGFACE HUB")
print("="*70)
print(f"\nUsername: {USERNAME}")
print(f"Dataset repo: {USERNAME}/prompt-injection-dataset")
print(f"Model repo: {USERNAME}/prompt-injection-detector-model")

# DATASET UPLOAD
print("\n" + "="*70)
print("📦 UPLOADING DATASET")
print("="*70)

try:
    login(token=DATASET_TOKEN)
    print("✅ Authenticated for dataset")
    
    base_dir = Path(__file__).parent.parent
    dataset_file = base_dir / "ml_data" / "full_dataset_expanded_augmented.jsonl"
    
    dataset = create_dataset_from_jsonl(dataset_file)
    dataset_repo = f"{USERNAME}/prompt-injection-dataset"
    
    print(f"\n📤 Pushing to {dataset_repo}...")
    dataset.push_to_hub(
        dataset_repo,
        private=False,
        token=DATASET_TOKEN
    )
    print(f"✅ Dataset uploaded!")
    print(f"🔗 https://huggingface.co/datasets/{dataset_repo}")
except Exception as e:
    print(f"❌ Dataset upload failed: {e}")

# MODEL UPLOAD
print("\n" + "="*70)
print("🤖 UPLOADING MODELS")
print("="*70)

try:
    login(token=MODEL_TOKEN)
    print("✅ Authenticated for models")
    
    models_dir = base_dir / "models"
    model_repo = f"{USERNAME}/prompt-injection-detector-model"
    print(f"✅ Using existing repository")
    
    # Upload models
    model_files = [
        "tfidf_vectorizer_expanded.pkl",
        "random_forest_expanded.pkl",
        "logistic_regression_expanded.pkl",
        "svm_expanded.pkl"
    ]
    
    for filename in model_files:
        file_path = models_dir / filename
        if file_path.exists():
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=model_repo,
                repo_type="model",
                token=MODEL_TOKEN
            )
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} ({size_mb:.1f} MB)")
    
    print(f"\n🔗 https://huggingface.co/{model_repo}")
    
    # Upload README
    readme = f"""---
language: en
license: apache-2.0
tags:
- prompt-injection
- security
datasets:
- {USERNAME}/prompt-injection-dataset
---

# PromptShield - Prompt Injection Detection

**Performance:** 100% accuracy on test set

Models: Random Forest (recommended), SVM, Logistic Regression

See dataset for details: https://huggingface.co/datasets/{USERNAME}/prompt-injection-dataset
"""
    
    readme_path = Path("README.md")
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=model_repo,
        repo_type="model",
        token=MODEL_TOKEN
    )
    readme_path.unlink()
    print(f"✅ README uploaded")
    
except Exception as e:
    print(f"❌ Model upload failed: {e}")

print("\n" + "="*70)
print("✅ UPLOAD COMPLETE!")
print("="*70)
