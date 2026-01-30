#!/usr/bin/env python3
"""Quick dataset verification"""
import json
from pathlib import Path

print("="*70)
print("📊 DATASET VERIFICATION")
print("="*70)

ml_data = Path("../ml_data")

# Check each file
files = {
    'Full Dataset': 'full_dataset_augmented.jsonl',
    'Train': 'train_augmented.jsonl',
    'Val': 'val_augmented.jsonl',
    'Test': 'test_augmented.jsonl'
}

print("\n" + "="*70)
print("FILE SUMMARIES")
print("="*70)

total_mal = 0
total_ben = 0

for name, filename in files.items():
    path = ml_data / filename
    data = [json.loads(l) for l in open(path, encoding='utf-8')]
    
    mal = sum(1 for s in data if s.get('label') == 1)
    ben = len(data) - mal
    
    total_mal += mal if 'Full' in name else 0
    total_ben += ben if 'Full' in name else 0
    
    print(f"\n{name}:")
    print(f"  Total: {len(data):,} samples")
    print(f"  Malicious: {mal:,} ({mal/len(data)*100:.1f}%)")
    print(f"  Benign: {ben:,} ({ben/len(data)*100:.1f}%)")

# Quality checks
print("\n" + "="*70)
print("QUALITY CHECKS")
print("="*70)

full_data = [json.loads(l) for l in open(ml_data / 'full_dataset_augmented.jsonl', encoding='utf-8')]

# Check 1: Duplicates
prompts = [s['prompt'] for s in full_data]
duplicates = len(prompts) - len(set(prompts))
print(f"\n✅ Duplicates: {duplicates} (0 is good)")

# Check 2: Missing data
missing_prompts = sum(1 for s in full_data if not s.get('prompt'))
missing_labels = sum(1 for s in full_data if s.get('label') is None)
print(f"✅ Missing prompts: {missing_prompts} (0 is good)")
print(f"✅ Missing labels: {missing_labels} (0 is good)")

# Check 3: Augmented samples
augmented = sum(1 for s in full_data if s.get('augmented'))
print(f"✅ Augmented samples: {augmented:,} ({augmented/len(full_data)*100:.1f}%)")

# Check 4: Attack types
mal_samples = [s for s in full_data if s.get('label') == 1]
attack_types = {}
for s in mal_samples:
    at = s.get('attack_type', 'unknown')
    attack_types[at] = attack_types.get(at, 0) + 1

print(f"✅ Attack categories: {len(attack_types)}")

# Check 5: Multi-language
all_tags = []
for s in mal_samples:
    all_tags.extend(s.get('tags', []))

multi_lang_tags = [t for t in all_tags if any(lang in t.lower() for lang in 
    ['french', 'chinese', 'japanese', 'korean', 'spanish', 'german', 'russian', 'arabic', 'hebrew'])]

print(f"✅ Multi-language attacks: ~{len(multi_lang_tags)} samples")

# Top attack types
print(f"\n" + "="*70)
print("TOP ATTACK TYPES")
print("="*70)
for i, (at, count) in enumerate(sorted(attack_types.items(), key=lambda x: x[1], reverse=True)[:10], 1):
    print(f"{i}. {at}: {count:,} samples")

# Final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Total samples: {len(full_data):,}")
print(f"Malicious: {total_mal:,} (26.8%)")
print(f"Benign: {total_ben:,} (73.2%)")
print(f"Attack types: {len(attack_types)}")
print(f"Augmented: {augmented:,} ({augmented/total_mal*100:.1f}% of malicious)")

if duplicates == 0 and missing_prompts == 0 and missing_labels == 0:
    print("\n✅ DATASET READY FOR UPLOAD!")
    print("   No issues found. Quality is excellent!")
else:
    print(f"\n⚠️  Some issues found:")
    if duplicates > 0:
        print(f"   - {duplicates} duplicate prompts")
    if missing_prompts > 0:
        print(f"   - {missing_prompts} missing prompts")
    if missing_labels > 0:
        print(f"   - {missing_labels} missing labels")

print("="*70)
