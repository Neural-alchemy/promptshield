"""
Dual Upload Strategy: Upload dataset to both m4vic (personal) and neuralchemy (org)
"""

from pathlib import Path
from huggingface_hub import upload_file, login, create_repo
from datasets import Dataset
import pandas as pd
import json

# Configuration
PERSONAL_USERNAME = "m4vic"
ORG_NAME = "neuralchemy"
import os
DATASET_TOKEN = os.getenv("HF_DATASET_TOKEN", "your-dataset-token-here")

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

def create_personal_readme():
    """README for personal profile - points to org version"""
    return """---
language:
- en
license: apache-2.0
tags:
- prompt-injection
- security
- llm-security
size_categories:
- 10K<n<100K
---

# Prompt Injection Detection Dataset

> **📌 Official Version**: This dataset is officially maintained at [neuralchemy/prompt-injection-benign-dataset](https://huggingface.co/datasets/neuralchemy/prompt-injection-benign-dataset)

A comprehensive dataset for detecting prompt injection attacks against LLMs.

## Quick Stats

- **10,674 samples** (2,903 malicious, 7,771 benign)
- **100% detection rate** with trained models
- **Real-world attacks** from PromptXploit + GitHub sources

## Usage

```python
from datasets import load_dataset

# Load from official org repo (recommended)
dataset = load_dataset("neuralchemy/prompt-injection-benign-dataset")

# Or from this personal mirror
dataset = load_dataset("m4vic/prompt-injection-dataset")
```

## Related Resources

- 🤗 **Official Dataset**: [neuralchemy/prompt-injection-benign-dataset](https://huggingface.co/datasets/neuralchemy/prompt-injection-benign-dataset)
- 🤖 **Models**: [neuralchemy/prompt-injection-detector-ml-models](https://huggingface.co/neuralchemy/prompt-injection-detector-ml-models)
- 🐙 **GitHub**: [neuralchemy/SecurePrompt](https://github.com/neuralchemy/SecurePrompt)
- 🌐 **Website**: [neuralchemy.in](https://www.neuralchemy.in/)

## Citation

```bibtex
@misc{neuralchemy2026promptinjection,
  author = {Neuralchemy},
  title = {Prompt Injection Detection Dataset},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/neuralchemy/prompt-injection-benign-dataset}
}
```

## License

Apache 2.0 - Maintained by [Neuralchemy](https://huggingface.co/neuralchemy)
"""

def upload_to_personal():
    """Upload to personal m4vic profile"""
    print("\n" + "="*70)
    print("📦 UPLOADING TO PERSONAL PROFILE (m4vic)")
    print("="*70)
    
    try:
        login(token=DATASET_TOKEN)
        
        base_dir = Path(__file__).parent.parent.parent
        dataset_file = base_dir / "promptshield" / "ml_data" / "full_dataset_expanded_augmented.jsonl"
        
        dataset = create_dataset_from_jsonl(dataset_file)
        personal_repo = f"{PERSONAL_USERNAME}/prompt-injection-dataset"
        
        print(f"\n📤 Pushing to {personal_repo}...")
        
        # Create/update repo
        try:
            create_repo(repo_id=personal_repo, repo_type="dataset", exist_ok=True, private=False, token=DATASET_TOKEN)
        except:
            pass  # Repo might already exist
        
        # Push dataset
        dataset.push_to_hub(
            personal_repo,
            private=False,
            token=DATASET_TOKEN,
            commit_message="Portfolio mirror - See neuralchemy/prompt-injection-benign-dataset for official version"
        )
        
        # Upload custom README
        readme_path = Path("README_personal_temp.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(create_personal_readme())
        
        upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=personal_repo,
            repo_type="dataset",
            token=DATASET_TOKEN
        )
        readme_path.unlink()
        
        print(f"✅ Personal version uploaded!")
        print(f"🔗 https://huggingface.co/datasets/{personal_repo}")
        return True
        
    except Exception as e:
        print(f"❌ Personal upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def upload_to_org():
    """Upload to neuralchemy organization"""
    print("\n" + "="*70)
    print("🏢 UPLOADING TO ORGANIZATION (neuralchemy)")
    print("="*70)
    
    try:
        login(token=DATASET_TOKEN)
        
        base_dir = Path(__file__).parent.parent.parent
        dataset_file = base_dir / "promptshield" / "ml_data" / "full_dataset_expanded_augmented.jsonl"
        
        dataset = create_dataset_from_jsonl(dataset_file)
        org_repo = f"{ORG_NAME}/prompt-injection-benign-dataset"
        
        print(f"\n📤 Pushing to {org_repo}...")
        
        # Push dataset
        dataset.push_to_hub(
            org_repo,
            private=False,
            token=DATASET_TOKEN,
            commit_message="Official organizational release"
        )
        
        # Upload official README (from huggingface folder)
        readme_path = base_dir / "huggingface" / "dataset_card.md"
        if readme_path.exists():
            upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=org_repo,
                repo_type="dataset",
                token=DATASET_TOKEN
            )
        
        print(f"✅ Organization version uploaded!")
        print(f"🔗 https://huggingface.co/datasets/{org_repo}")
        return True
        
    except Exception as e:
        print(f"❌ Organization upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("🚀 DUAL UPLOAD: PERSONAL + ORGANIZATION")
    print("="*70)
    print("\nStrategy:")
    print("  • m4vic/prompt-injection-dataset → Personal portfolio mirror")
    print("  • neuralchemy/prompt-injection-benign-dataset → Official version")
    print()
    
    # Upload to both
    personal_success = upload_to_personal()
    org_success = upload_to_org()
    
    # Summary
    print("\n" + "="*70)
    print("✅ DUAL UPLOAD COMPLETE")
    print("="*70)
    
    if personal_success:
        print(f"\n📦 Personal: https://huggingface.co/datasets/{PERSONAL_USERNAME}/prompt-injection-dataset")
    
    if org_success:
        print(f"🏢 Official: https://huggingface.co/datasets/{ORG_NAME}/prompt-injection-benign-dataset")
    
    print("\n💡 Benefits:")
    print("  ✓ Personal profile gets portfolio credit & downloads")
    print("  ✓ Organization maintains professional canonical version")
    print("  ✓ Both cross-reference each other")
    print("  ✓ Downloads accumulate on both!")

if __name__ == "__main__":
    main()
