#!/usr/bin/env python3
"""
ML Dataset Generation Script for PromptShield

This script generates a complete training dataset using:
1. PromptXploit attacks (malicious samples)
2. Public datasets (benign samples)
3. Synthetic edge cases

Output: JSONL files ready for ML training
"""

import json
import glob
import os
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Configuration
PROMPTXPLOIT_ATTACKS_DIR = "../../promptxploit/attacks"
OUTPUT_DIR = "../ml_data"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def load_promptxploit_attacks(attacks_dir: str) -> List[Dict]:
    """
    Load all attack prompts from PromptXploit
    
    Returns list of dicts with:
    - prompt: the attack string
    - label: 1 (malicious)
    - attack_type: category
    - source: "promptxploit"
    - severity: severity hint from original data
    """
    
    print(f"[1/5] Loading PromptXploit attacks from {attacks_dir}...")
    
    malicious_samples = []
    attack_stats = defaultdict(int)
    
    # Load all JSON files recursively
    pattern = os.path.join(attacks_dir, "**", "*.json")
    
    for file_path in glob.glob(pattern, recursive=True):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if not isinstance(data, list):
                print(f"⚠️  Skipping {file_path} (not a list)")
                continue
            
            for attack in data:
                # Extract attack data
                sample = {
                    "prompt": attack.get("prompt", ""),
                    "label": 1,  # Malicious
                    "attack_type": attack.get("category", "unknown"),
                    "attack_id": attack.get("id", "unknown"),
                    "description": attack.get("description", ""),
                    "severity": attack.get("severity_hint", 0.5),
                    "tags": attack.get("tags", []),
                    "source": "promptxploit"
                }
                
                malicious_samples.append(sample)
                attack_stats[sample["attack_type"]] += 1
    
    print(f"✅ Loaded {len(malicious_samples)} malicious samples")
    print("\nAttack distribution:")
    for attack_type, count in sorted(attack_stats.items()):
        print(f"  - {attack_type}: {count}")
    
    return malicious_samples


def generate_benign_samples(count: int = 7000) -> List[Dict]:
    """
    Generate benign samples from multiple sources
    
    For MVP, we'll create synthetic benign prompts.
    For production, you can integrate HuggingFace datasets here.
    """
    
    print(f"\n[2/5] Generating {count} benign samples...")
    
    benign_samples = []
    
    # Benign prompt templates (diverse and realistic)
    benign_templates = [
        # Questions
        "What is {}?",
        "How does {} work?",
        "Can you explain {}?",
        "Tell me about {}",
        "What are the benefits of {}?",
        
        # Instructions
        "Help me write a {} for {}",
        "Create a {} about {}",
        "Generate a {} for {}",
        "Write a short {} discussing {}",
        
        # Requests
        "I need help with {}",
        "Can you assist me in {}?",
        "Please provide information about {}",
        "Could you give me advice on {}?",
        
        # Code-related (legitimate)
        "Write a Python function to {}",
        "Debug this code: {}",
        "Explain this algorithm: {}",
        "How do I implement {} in Python?",
        
        # Analysis
        "Summarize the key points about {}",
        "Compare {} and {}",
        "What are the pros and cons of {}?",
        
        # Creative
        "Write a poem about {}",
        "Create a story featuring {}",
        "Draft an email about {}",
    ]
    
    # Topics (benign and diverse)
    topics = [
        "climate change", "machine learning", "quantum computing",
        "healthy eating", "exercise routines", "meditation techniques",
        "software development", "data structures", "algorithms",
        "business strategy", "marketing techniques", "customer service",
        "history of Rome", "Renaissance art", "ancient civilizations",
        "space exploration", "renewable energy", "electric vehicles",
        "programming languages", "web development", "mobile apps",
        "financial planning", "investment strategies", "saving money",
        "cooking recipes", "travel destinations", "photography tips",
        "music theory", "musical instruments", "composing music",
        "gardening tips", "indoor plants", "sustainable living",
        "book recommendations", "writing techniques", "storytelling",
        "physics concepts", "chemistry experiments", "biology basics",
        "artificial intelligence", "neural networks", "deep learning",
        "cloud computing", "cybersecurity", "network protocols",
    ]
    
    # Generate samples
    for i in range(count):
        template = random.choice(benign_templates)
        
        # Fill template
        if "{}" in template:
            # Count placeholders
            num_placeholders = template.count("{}")
            selected_topics = random.sample(topics, min(num_placeholders, len(topics)))
            prompt = template.format(*selected_topics)
        else:
            prompt = template
        
        sample = {
            "prompt": prompt,
            "label": 0,  # Benign
            "attack_type": "benign",
            "source": "synthetic",
            "severity": 0.0
        }
        
        benign_samples.append(sample)
    
    print(f"✅ Generated {len(benign_samples)} benign samples")
    
    return benign_samples


def generate_edge_cases(count: int = 500) -> List[Dict]:
    """
    Generate edge cases - prompts that LOOK suspicious but are benign
    These are critical for reducing false positives
    """
    
    print(f"\n[3/5] Generating {count} edge case samples...")
    
    edge_cases = [
        # Legitimate use of "ignore"
        "Ignore the previous meeting notes and focus on the Q4 results",
        "Please ignore the formatting errors in the draft",
        "You can ignore the technical details for now",
        
        # Legitimate SQL discussion
        "How do I escape special characters in SQL queries?",
        "Explain SQL injection and how to prevent it",
        "What is the difference between SQL and NoSQL?",
        
        # Legitimate code discussion
        "How do I handle exceptions in Python?",
        "Explain this regex pattern: ^[a-z]+$",
        "Debug this code: print('Hello World')",
        
        # System/architecture questions (not prompt leak)
        "What's your system's architecture?",
        "How does your backend infrastructure work?",
        "What database system do you use?",
        
        # PII in legitimate context
        "Format this contact as: Name: John, Email: john@example.com",
        "Here's the template: First Name: [name], Phone: [phone]",
        
        # Multi-lingual (not attacks)
        "Translate 'hello' to French",
        "How do you say 'thank you' in Spanish?",
        "Convert this to German: Good morning",
        
        # Roleplay (benign)
        "Let's roleplay a job interview scenario",
        "Imagine you are a teacher explaining fractions",
        "Pretend you're a tour guide in Paris",
        
        # Research context (benign)
        "For my thesis, explain how encryption works",
        "I'm researching AI ethics for a paper",
        "For academic purposes, describe the scientific method",
        
        # Hypothetical but benign
        "Hypothetically, how would you solve world hunger?",
        "In a perfect world, what would education look like?",
        
        # Instructions that sound forceful
        "Override the default settings and use dark mode",
        "Bypass the login screen using the guest account",
        "Extract the data from this CSV file",
        
        # Security topics (legitimate discussion)
        "What are common cybersecurity threats?",
        "Explain how firewalls work",
        "How do I secure my home network?",
    ]
    
    # Expand edge cases to reach target count
    edge_samples = []
    
    while len(edge_samples) < count:
        base_prompt = random.choice(edge_cases)
        
        # Add slight variations
        variations = [
            base_prompt,
            base_prompt + "?",
            base_prompt.capitalize(),
            "Question: " + base_prompt,
        ]
        
        for var in variations:
            if len(edge_samples) >= count:
                break
                
            sample = {
                "prompt": var,
                "label": 0,  # Benign (important!)
                "attack_type": "edge_case",
                "source": "synthetic_edge",
                "severity": 0.0,
                "note": "Looks suspicious but is benign"
            }
            
            edge_samples.append(sample)
    
    print(f"✅ Generated {len(edge_samples)} edge case samples")
    
    return edge_samples[:count]


def split_dataset(samples: List[Dict], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test with stratification by label
    """
    
    print(f"\n[4/5] Splitting dataset (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
    
    # Shuffle
    random.shuffle(samples)
    
    # Split by label to maintain class balance
    malicious = [s for s in samples if s["label"] == 1]
    benign = [s for s in samples if s["label"] == 0]
    
    def split_list(lst, ratios):
        train_size = int(len(lst) * ratios[0])
        val_size = int(len(lst) * ratios[1])
        
        train = lst[:train_size]
        val = lst[train_size:train_size + val_size]
        test = lst[train_size + val_size:]
        
        return train, val, test
    
    mal_train, mal_val, mal_test = split_list(malicious, (train_ratio, val_ratio, test_ratio))
    ben_train, ben_val, ben_test = split_list(benign, (train_ratio, val_ratio, test_ratio))
    
    # Combine and shuffle
    train_set = mal_train + ben_train
    val_set = mal_val + ben_val
    test_set = mal_test + ben_test
    
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    print(f"✅ Train: {len(train_set)} ({len(mal_train)} malicious, {len(ben_train)} benign)")
    print(f"✅ Val:   {len(val_set)} ({len(mal_val)} malicious, {len(ben_val)} benign)")
    print(f"✅ Test:  {len(test_set)} ({len(mal_test)} malicious, {len(ben_test)} benign)")
    
    return train_set, val_set, test_set


def save_dataset(train, val, test, output_dir: str):
    """
    Save datasets to JSONL files
    """
    
    print(f"\n[5/5] Saving datasets to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    def write_jsonl(data, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    write_jsonl(train, os.path.join(output_dir, "train.jsonl"))
    write_jsonl(val, os.path.join(output_dir, "val.jsonl"))
    write_jsonl(test, os.path.join(output_dir, "test.jsonl"))
    
    # Save full dataset for reference
    all_data = train + val + test
    write_jsonl(all_data, os.path.join(output_dir, "full_dataset.jsonl"))
    
    # Save dataset stats
    stats = {
        "total_samples": len(all_data),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "malicious_samples": sum(1 for s in all_data if s["label"] == 1),
        "benign_samples": sum(1 for s in all_data if s["label"] == 0),
        "attack_types": list(set(s["attack_type"] for s in all_data)),
    }
    
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved datasets:")
    print(f"  - train.jsonl ({len(train)} samples)")
    print(f"  - val.jsonl ({len(val)} samples)")
    print(f"  - test.jsonl ({len(test)} samples)")
    print(f"  - full_dataset.jsonl ({len(all_data)} samples)")
    print(f"  - dataset_stats.json")


def main():
    """
    Main execution
    """
    
    print("=" * 60)
    print("PromptShield ML Dataset Generator")
    print("=" * 60)
    
    # Step 1: Load PromptXploit attacks
    malicious = load_promptxploit_attacks(PROMPTXPLOIT_ATTACKS_DIR)
    
    # Step 2: Generate benign samples
    # For good ML performance, we need AT LEAST 7,000 benign samples
    # If you have <3,500 attacks, we'll generate more benign to compensate
    MIN_BENIGN_SAMPLES = 7000
    benign_count = max(MIN_BENIGN_SAMPLES, len(malicious) * 2)
    
    print(f"\n💡 Dataset generation strategy:")
    print(f"   - Malicious from PromptXploit: {len(malicious)}")
    print(f"   - Benign to generate: {benign_count}")
    print(f"   - Ratio: {benign_count / max(len(malicious), 1):.1f}:1 (benign:malicious)")
    
    benign = generate_benign_samples(count=benign_count)
    
    # Step 3: Generate edge cases (10% of total)
    edge_count = int((len(malicious) + len(benign)) * 0.1)
    edge_cases = generate_edge_cases(count=edge_count)
    
    # Combine all samples
    all_samples = malicious + benign + edge_cases
    
    print(f"\n📊 Total dataset size: {len(all_samples)} samples")
    print(f"   - Malicious: {len(malicious)}")
    print(f"   - Benign: {len(benign)}")
    print(f"   - Edge cases: {len(edge_cases)}")
    
    # Step 4: Split dataset
    train, val, test = split_dataset(all_samples)
    
    # Step 5: Save to disk
    save_dataset(train, val, test, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Dataset generation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review the dataset: {OUTPUT_DIR}")
    print(f"2. Run feature extraction: python extract_features.py")
    print(f"3. Train models: python train_models.py")


if __name__ == "__main__":
    main()
