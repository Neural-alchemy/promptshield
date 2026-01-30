"""
Generate RSA Keypair for Model Signing

Creates RSA-2048 key pair for signing PromptShield ML models.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptshield.security.model_signing import generate_keypair


def main():
    """Generate RSA keypair for model signing"""
    print("=" * 60)
    print("PromptShield - RSA Keypair Generation")
    print("=" * 60)
    print()
    
    # Generate keypair
    private_key, public_key = generate_keypair(
        key_size=2048,
        private_key_path="promptshield/security/keys/private_key.pem",
        public_key_path="promptshield/security/keys/public_key.pem"
    )
    
    print()
    print("=" * 60)
    print("✓ Setup Complete!")
    print()
    print("Next Steps:")
    print("1. Sign your models: python scripts/sign_models.py")
    print("2. Test evasion: python scripts/run_evasion_tests.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
