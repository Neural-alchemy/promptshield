#!/usr/bin/env python3
"""
One-command ML pipeline for PromptShield

Usage:
    python build_ml_pipeline.py --full         # Full pipeline (dataset + train)
    python build_ml_pipeline.py --dataset-only # Only generate dataset
    python build_ml_pipeline.py --train-only   # Only train (assumes dataset exists)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"\n✅ {description} - Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - Failed!")
        print(f"Error: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\n[CHECK] Verifying dependencies...")
    
    required = ['numpy', 'sklearn']
    optional = ['xgboost', 'lightgbm']
    
    missing_required = []
    missing_optional = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"  ❌ {package} (REQUIRED)")
    
    for package in optional:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"  ⚠️  {package} (optional but recommended)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print(f"Install with: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print(f"For best performance, install: pip install {' '.join(missing_optional)}")
        print("Continuing anyway...\n")
    
    return True


def verify_promptxploit():
    """Verify PromptXploit attacks directory exists"""
    print("\n[CHECK] Verifying PromptXploit directory...")
    
    attacks_dir = Path("../promptxploit/attacks")
    
    if not attacks_dir.exists():
        print(f"❌ PromptXploit attacks directory not found: {attacks_dir}")
        print("Please check the path in generate_ml_dataset.py")
        return False
    
    # Count attack files
    json_files = list(attacks_dir.glob("**/*.json"))
    
    if not json_files:
        print(f"❌ No attack JSON files found in: {attacks_dir}")
        return False
    
    print(f"✅ Found {len(json_files)} attack files")
    return True


def main():
    parser = argparse.ArgumentParser(description="PromptShield ML Pipeline Builder")
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--dataset-only', action='store_true', help='Only generate dataset')
    parser.add_argument('--train-only', action='store_true', help='Only train models')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency checks')
    
    args = parser.parse_args()
    
    # Default to full pipeline if no option specified
    if not (args.full or args.dataset_only or args.train_only):
        args.full = True
    
    print("=" * 60)
    print("PromptShield ML Pipeline Builder")
    print("=" * 60)
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            print("\n❌ Dependency check failed!")
            print("Install missing packages and try again.")
            sys.exit(1)
        
        if not verify_promptxploit():
            print("\n❌ PromptXploit verification failed!")
            sys.exit(1)
    
    success = True
    
    # Step 1: Generate dataset
    if args.full or args.dataset_only:
        if not run_command(
            f"{sys.executable} generate_ml_dataset.py",
            "Generating dataset from PromptXploit"
        ):
            success = False
    
    # Step 2: Extract features
    if success and (args.full or args.train_only):
        if not run_command(
            f"{sys.executable} extract_features.py",
            "Extracting features"
        ):
            success = False
    
    # Step 3: Train models
    if success and (args.full or args.train_only):
        if not run_command(
            f"{sys.executable} train_models.py",
            "Training ML models"
        ):
            success = False
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print("\nOutputs:")
        print("  Dataset:  ../promptshield/ml_data/")
        print("  Models:   ../promptshield/models/")
        print("  Results:  ../promptshield/models/training_results.json")
        print("\nNext steps:")
        print("  1. Review training_results.json")
        print("  2. Check model performance metrics")
        print("  3. Integrate L5 model into PromptShield")
        print("  4. Benchmark latency in production")
    else:
        print("❌ PIPELINE FAILED!")
        print("=" * 60)
        print("\nCheck the error messages above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
