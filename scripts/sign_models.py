"""
Sign All Models

Signs all PromptShield ML models with RSA private key.
"""

import glob
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptshield.security.model_signing import sign_model


def main():
    """Sign all model files"""
    print("=" * 60)
    print("PromptShield -Model Signing")
    print("=" * 60)
    print()
    
    # Find all model files
    model_files = glob.glob("models/*.pkl") + glob.glob("promptshield/models/*.pkl")
    
    if not model_files:
        print("⚠️  No model files found!")
        print("Expected locations:")
        print("  - models/*.pkl")
        print("  - promptshield/models/*.pkl")
        return
    
    print(f"Found {len(model_files)} model(s) to sign:")
    for model_file in model_files:
        print(f"  - {model_file}")
    print()
    
    # Check for private key
    private_key_path = "promptshield/security/keys/private_key.pem"
    if not Path(private_key_path).exists():
        print(f"✗ Private key not found: {private_key_path}")
        print()
        print("Generate keys first:")
        print("  python scripts/generate_keys.py")
        return
    
    # Sign each model
    signed_count = 0
    for model_file in model_files:
        try:
            print(f"Signing {model_file}...")
            sig_file = sign_model(model_file, private_key_path)
            signed_count += 1
            print()
        except Exception as e:
            print(f"✗ Failed to sign {model_file}: {e}")
            print()
    
    print("=" * 60)
    print(f"✓ Signed {signed_count}/{len(model_files)} models")
    print("=" * 60)


if __name__ == "__main__":
    main()
