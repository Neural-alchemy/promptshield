"""
Upload Dataset and Model Cards to HuggingFace
"""

from pathlib import Path
from huggingface_hub import upload_file, login

# Configuration
USERNAME = "m4vic"
import os
DATASET_TOKEN = os.getenv("HF_DATASET_TOKEN", "your-dataset-token-here")
MODEL_TOKEN = os.getenv("HF_MODEL_TOKEN", "your-model-token-here")

base_dir = Path(__file__).parent.parent.parent
dataset_card = base_dir / "dataset_card.md"
model_card = base_dir / "model_card.md"

print("="*70)
print("📝 UPLOADING CARDS TO HUGGINGFACE")
print("="*70)

# Upload Dataset Card
print("\n📦 Dataset Card...")
try:
    login(token=DATASET_TOKEN)
    
    upload_file(
        path_or_fileobj=str(dataset_card),
        path_in_repo="README.md",
        repo_id=f"{USERNAME}/prompt-injection-dataset",
        repo_type="dataset",
        token=DATASET_TOKEN
    )
    print("✅ Dataset card uploaded")
    print(f"🔗 https://huggingface.co/datasets/{USERNAME}/prompt-injection-dataset")
except Exception as e:
    print(f"❌ Failed: {e}")

# Upload Model Card
print("\n🤖 Model Card...")
try:
    login(token=MODEL_TOKEN)
    
    upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=f"{USERNAME}/prompt-injection-detector-model",
        repo_type="model",
        token=MODEL_TOKEN
    )
    print("✅ Model card uploaded")
    print(f"🔗 https://huggingface.co/{USERNAME}/prompt-injection-detector-model")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "="*70)
print("✅ CARDS UPLOADED!")
print("="*70)
