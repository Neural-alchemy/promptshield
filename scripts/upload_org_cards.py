"""
Upload Organization Card and Updated Dataset/Model Cards
"""

from pathlib import Path
from huggingface_hub import upload_file, login

# Configuration
ORG_NAME = "neuralchemy"
DATASET_REPO = "prompt-injection-benign-dataset"
MODEL_REPO = "prompt-injection-detector-ml-models"
import os
DATASET_TOKEN = os.getenv("HF_DATASET_TOKEN", "your-dataset-token-here")
MODEL_TOKEN = os.getenv("HF_MODEL_TOKEN", "your-model-token-here")

base_dir = Path(__file__).parent.parent.parent
dataset_card = base_dir / "dataset_card.md"
model_card = base_dir / "model_card.md"
org_card = base_dir / "org_card.md"

print("="*70)
print("📝 UPLOADING UPDATED CARDS FOR NEURALCHEMY")
print("="*70)

# Upload Organization Card
print("\n🧪 Organization Card...")
try:
    login(token=DATASET_TOKEN)
    
    # Note: Org cards are uploaded differently - you need to do this manually
    # on the HuggingFace website at: https://huggingface.co/organizations/neuralchemy/edit
    print("⚠️  Organization cards must be uploaded manually")
    print("   Go to: https://huggingface.co/organizations/neuralchemy/edit")
    print("   Copy content from: org_card.md")
    print("   Or I can show you the content to paste")
except Exception as e:
    print(f"❌ Failed: {e}")

# Upload Dataset Card
print("\n📦 Dataset Card...")
try:
    login(token=DATASET_TOKEN)
    
    upload_file(
        path_or_fileobj=str(dataset_card),
        path_in_repo="README.md",
        repo_id=f"{ORG_NAME}/{DATASET_REPO}",
        repo_type="dataset",
        token=DATASET_TOKEN
    )
    print("✅ Dataset card uploaded")
    print(f"🔗 https://huggingface.co/datasets/{ORG_NAME}/{DATASET_REPO}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Upload Model Card
print("\n🤖 Model Card...")
try:
    login(token=MODEL_TOKEN)
    
    upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=f"{ORG_NAME}/{MODEL_REPO}",
        repo_type="model",
        token=MODEL_TOKEN
    )
    print("✅ Model card uploaded")
    print(f"🔗 https://huggingface.co/{ORG_NAME}/{MODEL_REPO}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "="*70)
print("✅ CARDS UPDATED FOR NEURALCHEMY!")
print("="*70)
print("\n📝 Manual Step Required:")
print("   Upload org_card.md content to:")
print("   https://huggingface.co/organizations/neuralchemy/edit")
