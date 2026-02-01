"""
Generate Augmented Variations of GitHub Attacks
Applies same augmentation techniques to newly added GitHub attacks
"""

import json
import random
from pathlib import Path
from typing import List, Dict

# Import augmentation functions from generate_ml_dataset
import sys
sys.path.append(str(Path(__file__).parent))

def apply_case_variation(text: str) -> str:
    """Apply random case changes"""
    result = []
    for char in text:
        if char.isalpha():
            result.append(char.upper() if random.random() > 0.5 else char.lower())
        else:
            result.append(char)
    return ''.join(result)

def apply_leetspeak(text: str) -> str:
    """Apply leet speak substitutions"""
    leetmap = {
        'a': ['@', '4'], 'e': ['3', '€'], 'i': ['1', '!', '|'],
        'o': ['0', 'ο'], 's': ['$', '5'], 't': ['7', '+'],
        'l': ['|', '1'], 'b': ['8', 'ь']
    }
    result = []
    for char in text.lower():
        if char in leetmap and random.random() > 0.6:
            result.append(random.choice(leetmap[char]))
        else:
            result.append(char)
    return ''.join(result)

def apply_unicode_substitution(text: str) -> str:
    """Apply Unicode lookalike substitutions"""
    unicodemap = {
        'a': 'α', 'e': 'е', 'i': 'і', 'o': 'ο', 'c': 'с',
        'p': 'р', 's': 'ѕ', 'x': 'х', 'y': 'у'
    }
    result = []
    for char in text.lower():
        if char in unicodemap and random.random() > 0.7:
            result.append(unicodemap[char])
        else:
            result.append(char)
    return ''.join(result)

def apply_whitespace_variation(text: str) -> str:
    """Add extra spaces or tabs"""
    variations = [
        lambda t: t.replace(' ', '  '),  # Double spaces
        lambda t: t.replace(' ', '\t'),  # Tabs
        lambda t: f"   {t}   ",  # Padding
        lambda t: t  # No change
    ]
    return random.choice(variations)(text)

def create_augmented_variations(attack: Dict, num_variations: int = 3) -> List[Dict]:
    """Create augmented variations of an attack"""
    variations = []
    original_prompt = attack['prompt']
    
    augmentation_techniques = [
        ('case_variation', apply_case_variation),
        ('leetspeak', apply_leetspeak),
        ('unicode_substitution', apply_unicode_substitution),
        ('whitespace', apply_whitespace_variation),
    ]
    
    for i in range(num_variations):
        # Apply random combination of techniques
        augmented = original_prompt
        techniques_used = []
        
        # Always apply at least one technique
        technique_name, technique_func = random.choice(augmentation_techniques)
        augmented = technique_func(augmented)
        techniques_used.append(technique_name)
        
        # Optionally apply another
        if random.random() > 0.5 and len(augmentation_techniques) > 1:
            other_techniques = [t for t in augmentation_techniques if t[0] != technique_name]
            technique_name2, technique_func2 = random.choice(other_techniques)
            augmented = technique_func2(augmented)
            techniques_used.append(technique_name2)
        
        # Create variation dict
        variation = attack.copy()
        variation['prompt'] = augmented
        variation['augmented'] = True
        variation['augmentation_id'] = i + 1
        variation['original_prompt'] = original_prompt
        variation['augmentation_techniques'] = techniques_used
        
        variations.append(variation)
    
    return variations

def main():
    print("🎨 Generating Augmented Variations...")
    
    # Paths
    base_dir = Path(__file__).parent.parent
    ml_data_dir = base_dir / "ml_data"
    input_file = ml_data_dir / "full_dataset_expanded.jsonl"
    output_file = ml_data_dir / "full_dataset_expanded_augmented.jsonl"
    
    # Load expanded dataset
    print(f"📊 Loading {input_file}...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"   Loaded {len(data)} samples")
    
    # Find GitHub attacks that need augmentation
    github_attacks = [item for item in data if 'github_' in item.get('source', '') and not item.get('augmented', False)]
    other_samples = [item for item in data if 'github_' not in item.get('source', '') or item.get('augmented', False)]
    
    print(f"🔍 Found {len(github_attacks)} new GitHub attacks to augment")
    
    # Generate augmentations
    augmented_data = []
    for attack in github_attacks:
        # Keep original
        augmented_data.append(attack)
        # Add variations
        variations = create_augmented_variations(attack, num_variations=3)
        augmented_data.extend(variations)
    
    print(f"✨ Generated {len(augmented_data) - len(github_attacks)} augmented variations")
    
    # Combine all data
    final_data = other_samples + augmented_data
    
    # Save
    print(f"💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Statistics
    total_malicious = sum(1 for item in final_data if item['label'] == 1)
    total_benign = sum(1 for item in final_data if item['label'] == 0)
    
    print("\n" + "="*60)
    print("✅ Augmentation Complete!")
    print("="*60)
    print(f"📊 Final Statistics:")
    print(f"   Total samples:     {len(final_data):,}")
    print(f"   - Malicious:       {total_malicious:,}")
    print(f"   - Benign:          {total_benign:,}")
    print(f"\n💾 Saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()
