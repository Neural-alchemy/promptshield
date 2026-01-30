#!/usr/bin/env python3
"""
Convert JSONL prompt injection dataset to PromptXploit format
"""

import json
import re
from collections import defaultdict

# Your JSONL data (paste here)
jsonl_data = """
# PASTE YOUR 500 LINES HERE
"""

def parse_jsonl_to_promptxploit():
    """Convert JSONL format to PromptXploit format"""
    
    # Category mapping
    attack_type_mapping = {
        'jailbreaking': 'jailbreak',
        'role_playing': 'jailbreak',
        'obfuscation': 'encoding',
        'data_leakage': 'training_extraction',
        'code_execution': 'adversarial',
        'none': None,  # Skip benign
    }
    
    attacks_by_category = defaultdict(list)
    stats = defaultdict(int)
    
    # Parse each line
    lines = jsonl_data.strip().split('\n')
    
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
            
        try:
            entry = json.loads(line)
            
            # Skip benign samples
            if entry['label'] == 'benign':
                stats['benign_skipped'] += 1
                continue
            
            # Get attack type
            attack_type = entry.get('attack_type', 'unknown')
            category = attack_type_mapping.get(attack_type, attack_type)
            
            # Convert to PromptXploit format
            attack = {
                "id": f"JSONL-{entry['id']}",
                "category": category,
                "description": entry.get('context', 'Prompt injection attack'),
                "prompt": entry['prompt'],
                "expected_violation": True,
                "severity_hint": estimate_severity(entry),
                "tags": [attack_type, "jsonl_dataset", "real_world"],
                "source": "jsonl_dataset_500",
                "original_id": entry['id']
            }
            
            attacks_by_category[category].append(attack)
            stats['malicious_converted'] += 1
            
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {line[:50]}... - {e}")
            stats['errors'] += 1
    
    return dict(attacks_by_category), stats


def estimate_severity(entry):
    """Estimate severity based on attack type and content"""
    attack_type = entry.get('attack_type', '')
    prompt = entry['prompt'].lower()
    
    # High severity indicators
    if any(keyword in prompt for keyword in ['system', 'admin', 'root', 'delete', 'drop', 'rm -rf']):
        return 0.9
    
    # Medium-high severity
    if attack_type in ['code_execution', 'data_leakage']:
        return 0.8
    
    # Medium severity
    if attack_type in ['jailbreaking', 'role_playing']:
        return 0.7
    
    # Lower severity
    if attack_type == 'obfuscation':
        return 0.6
    
    return 0.5


def main():
    print("Converting JSONL to PromptXploit format...")
    
    attacks_by_category, stats = parse_jsonl_to_promptxploit()
    
    print(f"\n📊 Conversion Stats:")
    print(f"  Malicious converted: {stats['malicious_converted']}")
    print(f"  Benign skipped: {stats['benign_skipped']}")
    print(f"  Errors: {stats.get('errors', 0)}")
    
    print(f"\n📂 Attacks by Category:")
    for category, attacks in sorted(attacks_by_category.items()):
        print(f"  {category}: {len(attacks)} attacks")
    
    # Save to files
    for category, attacks in attacks_by_category.items():
        filename = f"jsonl_{category}_attacks.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(attacks, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved: {filename} ({len(attacks)} attacks)")
    
    # Save combined file
    all_attacks = []
    for attacks in attacks_by_category.values():
        all_attacks.extend(attacks)
    
    with open('jsonl_all_attacks.json', 'w', encoding='utf-8') as f:
        json.dump(all_attacks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved: jsonl_all_attacks.json ({len(all_attacks)} total attacks)")


if __name__ == "__main__":
    main()
