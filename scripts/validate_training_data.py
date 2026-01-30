"""
Validate Training Dataset

Validates training data before model training to prevent poisoning attacks.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptshield.training import DatasetValidator


def main():
    """Validate training dataset"""
    print("=" * 60)
    print("PromptShield - Training Data Validation")
    print("=" * 60)
    print()
    
    # Try to load existing training data
    try:
        # Look for common data file locations
        data_files = [
            "promptshield/ml_data/train_data.npz",
            "ml_data/train_data.npz",
            "data/train.npz"
        ]
        
        data = None
        data_file = None
        
        for df in data_files:
            if Path(df).exists():
                data = np.load(df)
                data_file = df
                break
        
        if data is None:
            print("⚠️  No training data found!")
            print()
            print("Expected locations:")
            for df in data_files:
                print(f"  - {df}")
            print()
            print("Generating synthetic example instead...")
            print()
            
            # Generate synthetic dataset for demonstration
            np.random.seed(42)
            X = np.random.randn(200, 10)
            y = np.random.randint(0, 2, 200)
            
            # Add some poisoned samples
            X_poison = np.random.randn(20, 10) * 5  # Outliers
            y_poison = np.ones(20, dtype=int)
            
            X = np.vstack([X, X_poison])
            y = np.concatenate([y, y_poison])
            
            print(f"✓ Generated synthetic dataset: {len(X)} samples")
        
        else:
            print(f"✓ Loaded training data from: {data_file}")
            X = data['X']
            y = data['y']
            print(f"  Samples: {len(X)}")
            print(f"  Features: {X.shape[1]}")
        
        print()
        
        # Validate dataset
        print("Running validation checks...")
        print()
        
        validator = DatasetValidator()
        result = validator.validate(X, y)
        
        # Print results
        print("=" * 60)
        print("Validation Results:")
        print("=" * 60)
        print(f"Status: {'✅ VALID' if result['is_valid'] else '❌ INVALID'}")
        print(f"Quality Score: {result['quality_score']:.2f}/1.00")
        print(f"Total Issues: {result['total_issues']}")
        print(f"  Critical: {result['critical_issues']}")
        print(f"  Warnings: {result['warnings']}")
        print()
        print(f"Recommendation: {result['recommendation']}")
        print("=" * 60)
        print()
        
        # Generate detailed report
        if result['total_issues'] > 0:
            print("Detailed Report:")
            print()
            report = validator.generate_report()
            print(report)
            
            # Save report
            report_file = "data_validation_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"✓ Report saved: {report_file}")
            print()
        
        # Offer to clean dataset
        if not result['is_valid'] or result['quality_score'] < 0.9:
            print("Cleaning dataset...")
            X_clean, y_clean = validator.clean_dataset(X, y, auto_fix=True)
            print()
            
            # Re-validate cleaned data
            result2 = validator.validate(X_clean, y_clean)
            print("After Cleaning:")
            print(f"  Quality Score: {result2['quality_score']:.2f}/1.00")
            print(f"  Recommendation: {result2['recommendation']}")
            print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
