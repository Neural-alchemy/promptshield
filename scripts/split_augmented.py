#!/usr/bin/env python3
"""Split augmented dataset into train/val/test"""

import json
import random
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Load augmented dataset
print("Loading augmented dataset...")
samples = []
with open('../ml_data/full_dataset_augmented.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

# Shuffle
random.shuffle(samples)

# Split (70/15/15)
train_idx = int(0.7 * len(samples))
val_idx = int(0.85 * len(samples))

train = samples[:train_idx]
val = samples[train_idx:val_idx]
test = samples[val_idx:]

# Count malicious
train_mal = sum(1 for s in train if s.get('label') == 1)
val_mal = sum(1 for s in val if s.get('label') == 1)
test_mal = sum(1 for s in test if s.get('label') == 1)

print(f"\nSplit complete:")
print(f"  Train: {len(train)} ({train_mal} malicious, {len(train)-train_mal} benign)")
print(f"  Val:   {len(val)} ({val_mal} malicious, {len(val)-val_mal} benign)")
print(f"  Test:  {len(test)} ({test_mal} malicious, {len(test)-test_mal} benign)")

# Save
print("\nSaving splits...")
for name, data in [('train_augmented', train), ('val_augmented', val), ('test_augmented', test)]:
    with open(f'../ml_data/{name}.jsonl', 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✅ {name}.jsonl")

print("\n✅ Done! Ready for training.")
