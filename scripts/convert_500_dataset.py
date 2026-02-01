#!/usr/bin/env python3
"""
Convert 500-sample JSONL dataset to PromptXploit format
Automatically categorizes and saves to real_world attacks directory
"""

import json
from pathlib import Path
from collections import defaultdict

# Category mapping from attack_type to PromptXploit categories
ATTACK_TYPE_MAPPING = {
    'jailbreaking': 'jailbreak',
    'role_playing': 'jailbreak',
    'obfuscation': 'encoding',
    'data_leakage': 'training_extraction',
    'code_execution': 'adversarial',
    'none': None,  # Skip benign samples
}

def estimate_severity(entry):
    """Estimate severity based on attack type and content"""
    attack_type = entry.get('attack_type', '')
    prompt = entry['prompt'].lower()
    context = entry.get('context', '').lower()
    
    # Critical severity
    if any(kw in prompt + context for kw in ['delete', 'drop table', 'rm -rf', 'shutdown', 'reboot']):
        return 0.9
    
    # High severity
    if attack_type == 'code_execution':
        if any(kw in prompt for kw in ['system', 'exec', 'eval', 'popen']):
            return 0.85
        return 0.8
    
    # Medium-high severity
    if attack_type == 'data_leakage':
        if any(kw in prompt for kw in ['password', 'key', 'token', 'secret']):
            return 0.85
        return 0.75
    
    # Medium severity
    if attack_type in ['jailbreaking', 'role_playing']:
        return 0.7
    
    # Lower severity
    if attack_type == 'obfuscation':
        return 0.65
    
    return 0.6


def extract_language_from_context(context):
    """Extract language tag from context if it's a multi-language attack"""
    lang_mapping = {
        'french': 'french',
        'chinese': 'chinese',
        'italian': 'italian',
        'japanese': 'japanese',
        'korean': 'korean',
        'spanish': 'spanish',
        'german': 'german',
        'russian': 'russian',
        'arabic': 'arabic',
        'hebrew': 'hebrew',
    }
    
    context_lower = context.lower()
    for lang_name, lang_tag in lang_mapping.items():
        if lang_name in context_lower:
            return lang_tag
    return None


def convert_jsonl_to_promptxploit(jsonl_file, output_dir):
    """Convert JSONL dataset to PromptXploit format"""
    
    print(f"📂 Reading {jsonl_file}...")
    
    attacks_by_category = defaultdict(list)
    stats = {
        'total': 0,
        'malicious': 0,
        'benign_skipped': 0,
        'errors': 0
    }
    
    # Read and parse JSONL
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                stats['total'] += 1
                entry = json.loads(line)
                
                # Skip benign samples
                if entry.get('label') == 'benign':
                    stats['benign_skipped'] += 1
                    continue
                
                # Get category
                attack_type = entry.get('attack_type', 'unknown')
                category = ATTACK_TYPE_MAPPING.get(attack_type, attack_type)
                
                # Build tags
                tags = [attack_type, "jsonl_500", "real_world"]
                
                # Add language tag if multi-language
                lang = extract_language_from_context(entry.get('context', ''))
                if lang:
                    tags.append(f"multi_language")
                    tags.append(lang)
                
                # Convert to PromptXploit format
                attack = {
                    "id": f"JSONL-{entry['id']}",
                    "category": category,
                    "description": entry.get('context', 'Prompt injection attack'),
                    "prompt": entry['prompt'],
                    "expected_violation": True,
                    "severity_hint": estimate_severity(entry),
                    "tags": tags,
                    "source": "jsonl_500_dataset",
                    "original_id": entry['id']
                }
                
                attacks_by_category[category].append(attack)
                stats['malicious'] += 1
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num} parse error: {str(e)[:50]}")
                stats['errors'] += 1
            except Exception as e:
                print(f"⚠️  Line {line_num} error: {str(e)[:50]}")
                stats['errors'] += 1
    
    # Save by category
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Conversion Stats:")
    print(f"  Total lines: {stats['total']}")
    print(f"  Malicious converted: {stats['malicious']}")
    print(f"  Benign skipped: {stats['benign_skipped']}")
    print(f"  Errors: {stats['errors']}")
    
    print(f"\n📂 Attacks by Category:")
    for category, attacks in sorted(attacks_by_category.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {category}: {len(attacks)} attacks")
        
        # Save to file
        filename = output_path / f"jsonl_{category}_attacks.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(attacks, f, indent=2, ensure_ascii=False)
        print(f"    ✅ Saved: {filename.name}")
    
    # Save combined file
    all_attacks = []
    for attacks in attacks_by_category.values():
        all_attacks.extend(attacks)
    
    combined_file = output_path / 'jsonl_all_attacks.json'
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_attacks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ COMPLETE! Saved {len(all_attacks)} attacks")
    print(f"📁 Location: {output_path}")
    
    return stats, attacks_by_category


if __name__ == "__main__":
    import sys
    
    # Paths
    script_dir = Path(__file__).parent
    jsonl_file = script_dir / "full_datset_500.jsonI.txt"
    output_dir = script_dir.parent.parent / "promptxploit" / "attacks" / "real_world"
    
    print("=" * 60)
    print("🚀 Converting 500-Sample JSONL to PromptXploit Format")
    print("=" * 60)
    
    if not jsonl_file.exists():
        print(f"❌ JSONL file not found: {jsonl_file}")
        sys.exit(1)
    
    stats, attacks = convert_jsonl_to_promptxploit(jsonl_file, output_dir)
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Your PromptXploit is now MASSIVELY upgraded!")
    print("=" * 60)
