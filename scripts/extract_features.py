#!/usr/bin/env python3
"""
Fast Feature Extraction for ML Models

Extracts lightweight statistical features from prompts
Target: <1ms per prompt for feature extraction
"""

import json
import math
import re
import pickle
from pathlib import Path
from typing import Dict, List
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Keyword lists for fast lookups
SYSTEM_KEYWORDS = {
    'system', 'instructions', 'rules', 'prompt', 'context', 'directive',
    'override', 'bypass', 'disable', 'ignore', 'forget', 'disregard'
}

INSTRUCTION_WORDS = {
    'do', 'execute', 'run', 'perform', 'act', 'behave', 'pretend',
    'imagine', 'assume', 'roleplay', 'simulate', 'emulate'
}

SQL_KEYWORDS = {
    'select', 'insert', 'update', 'delete', 'drop', 'union',
    'where', 'from', 'join', 'table', 'database', 'exec'
}

CODE_PATTERNS = {
    '<script>', '</script>', '${', '{{', '}}', 'eval(',
    'exec(', '__import__', 'os.system', 'subprocess'
}

SPECIAL_CHARS = set('!@#$%^&*()_+-=[]{}|;:\'",.<>?/\\`~')


def extract_features(prompt: str) -> Dict:
    """
    Extract fast ML features from a prompt
    
    Returns dict with ~50 features
    """
    
    if not prompt:
        prompt = ""
    
    features = {}
    
    # ─────────────────────────────────────────
    # 1. BASIC STRUCTURAL FEATURES
    # ─────────────────────────────────────────
    
    features['length'] = len(prompt)
    features['word_count'] = len(prompt.split())
    features['char_count'] = len(prompt)
    features['line_count'] = len(prompt.split('\n'))
    
    # Character ratios
    if len(prompt) > 0:
        features['uppercase_ratio'] = sum(1 for c in prompt if c.isupper()) / len(prompt)
        features['lowercase_ratio'] = sum(1 for c in prompt if c.islower()) / len(prompt)
        features['digit_ratio'] = sum(1 for c in prompt if c.isdigit()) / len(prompt)
        features['space_ratio'] = sum(1 for c in prompt if c.isspace()) / len(prompt)
        features['special_char_ratio'] = sum(1 for c in prompt if c in SPECIAL_CHARS) / len(prompt)
    else:
        features['uppercase_ratio'] = 0
        features['lowercase_ratio'] = 0
        features['digit_ratio'] = 0
        features['space_ratio'] = 0
        features['special_char_ratio'] = 0
    
    # ─────────────────────────────────────────
    # 2. ENTROPY & COMPLEXITY
    # ─────────────────────────────────────────
    
    # Character entropy
    char_freq = Counter(prompt)
    total_chars = len(prompt)
    
    if total_chars > 0:
        entropy = -sum((count/total_chars) * math.log2(count/total_chars) 
                      for count in char_freq.values())
        features['entropy'] = entropy
    else:
        features['entropy'] = 0
    
    # Average word length
    words = prompt.split()
    if words:
        features['avg_word_length'] = sum(len(w) for w in words) / len(words)
    else:
        features['avg_word_length'] = 0
    
    # ────────────────────────────────────────────
    # 3. LEXICAL FEATURES (Fast Keyword Lookup)
    # ────────────────────────────────────────────
    
    prompt_lower = prompt.lower()
    prompt_words = set(prompt_lower.split())
    
    features['has_system_keywords'] = int(bool(SYSTEM_KEYWORDS & prompt_words))
    features['system_keyword_count'] = len(SYSTEM_KEYWORDS & prompt_words)
    
    features['has_instruction_words'] = int(bool(INSTRUCTION_WORDS & prompt_words))
    features['instruction_word_count'] = len(INSTRUCTION_WORDS & prompt_words)
    
    features['has_sql_keywords'] = int(bool(SQL_KEYWORDS & prompt_words))
    features['sql_keyword_count'] = len(SQL_KEYWORDS & prompt_words)
    
    # ─────────────────────────────────────────
    # 4. PATTERN DETECTION
    # ─────────────────────────────────────────
    
    # Code patterns
    features['has_code_patterns'] = int(any(p in prompt for p in CODE_PATTERNS))
    features['code_pattern_count'] = sum(1 for p in CODE_PATTERNS if p in prompt)
    
    # Encoding tricks
    features['has_base64_like'] = int(re.search(r'[A-Za-z0-9+/=]{20,}', prompt) is not None)
    features['has_hex_encoding'] = int(re.search(r'\\x[0-9a-fA-F]{2}', prompt) is not None)
    features['has_unicode_escape'] = int(re.search(r'\\u[0-9a-fA-F]{4}', prompt) is not None)
    
    # Delimiters and brackets
    features['bracket_count'] = prompt.count('(') + prompt.count('[') + prompt.count('{')
    features['quote_count'] = prompt.count('"') + prompt.count("'")
    features['max_bracket_depth'] = calculate_max_nesting_depth(prompt)
    
    # ─────────────────────────────────────────
    # 5. BEHAVIORAL FEATURES
    # ─────────────────────────────────────────
    
    # Repetition detection
    features['has_repetition'] = int(has_character_repetition(prompt, threshold=5))
    features['max_char_repetition'] = max_consecutive_chars(prompt)
    
    # Multiple spaces
    features['has_multiple_spaces'] = int('  ' in prompt)
    features['max_consecutive_spaces'] = max_consecutive_spaces(prompt)
    
    # Newlines
    features['has_multiple_newlines'] = int('\n\n' in prompt)
    features['newline_count'] = prompt.count('\n')
    
    # ─────────────────────────────────────────
    # 6. SEMANTIC INDICATORS
    # ─────────────────────────────────────────
    
    # Jailbreak phrases
    jailbreak_phrases = [
        'ignore', 'disregard', 'forget', 'override', 'bypass',
        'hypothetically', 'fictional', 'roleplay', 'pretend',
        'without restrictions', 'no rules', 'no limitations'
    ]
    features['jailbreak_phrase_count'] = sum(1 for phrase in jailbreak_phrases 
                                             if phrase in prompt_lower)
    
    # PII indicators
    pii_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
    ]
    features['pii_pattern_count'] = sum(1 for pattern in pii_patterns 
                                       if re.search(pattern, prompt, re.IGNORECASE))
    
    # Question markers
    features['is_question'] = int(prompt.strip().endswith('?'))
    features['question_word_count'] = sum(1 for w in ['what', 'how', 'why', 'when', 'where', 'who']
                                         if w in prompt_lower.split())
    
    # Imperative indicators
    imperative_verbs = ['tell', 'show', 'give', 'provide', 'explain', 'describe', 'write']
    features['imperative_verb_count'] = sum(1 for v in imperative_verbs 
                                           if prompt_lower.split()[0:2].__contains__(v) if prompt_lower.split())
    
    # Length-based features
    features['is_very_short'] = int(len(prompt) < 10)
    features['is_very_long'] = int(len(prompt) > 500)
    
    # N-gram features (simple)
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    features['unique_bigram_ratio'] = len(set(bigrams)) / max(len(bigrams), 1)
    
    return features


def calculate_max_nesting_depth(text: str) -> int:
    """Calculate maximum nesting depth of brackets"""
    depth = 0
    max_depth = 0
    
    for char in text:
        if char in '([{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char in ')]}':
            depth = max(0, depth - 1)
    
    return max_depth


def has_character_repetition(text: str, threshold: int = 5) -> bool:
    """Check if any character is repeated consecutively"""
    for i in range(len(text) - threshold):
        if len(set(text[i:i+threshold])) == 1:
            return True
    return False


def max_consecutive_chars(text: str) -> int:
    """Find maximum consecutive repetition of any character"""
    if not text:
        return 0
    
    max_count = 1
    current_count = 1
    
    for i in range(1, len(text)):
        if text[i] == text[i-1]:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 1
    
    return max_count


def max_consecutive_spaces(text: str) -> int:
    """Find maximum consecutive spaces"""
    max_count = 0
    current_count = 0
    
    for char in text:
        if char == ' ':
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count


def extract_features_batch(dataset_path: str, output_path: str):
    """
    Extract features from entire dataset
    """
    
    print(f"Loading dataset from {dataset_path}...")
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples")
    print("Extracting features...")
    
    # Extract features
    feature_list = []
    labels = []
    metadata = []
    
    for i, sample in enumerate(samples):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(samples)}")
        
        features = extract_features(sample['prompt'])
        feature_list.append(features)
        labels.append(sample['label'])
        metadata.append({
            'attack_type': sample.get('attack_type', 'unknown'),
            'source': sample.get('source', 'unknown')
        })
    
    # Convert to numpy arrays
    feature_names = list(feature_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in feature_list])
    y = np.array(labels)
    
    # Add TF-IDF features
    print("Computing TF-IDF features...")
    prompts = [s['prompt'] for s in samples]
    
    tfidf = TfidfVectorizer(
        max_features=100,  # Top 100 terms
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.95
    )
    
    X_tfidf = tfidf.fit_transform(prompts).toarray()
    
    # Combine features
    X_combined = np.hstack([X, X_tfidf])
    
    print(f"Feature extraction complete!")
    print(f"  - Statistical features: {X.shape[1]}")
    print(f"  - TF-IDF features: {X_tfidf.shape[1]}")
    print(f"  - Total features: {X_combined.shape[1]}")
    
    # Save
    output_data = {
        'X': X_combined,
        'y': y,
        'feature_names': feature_names + [f'tfidf_{i}' for i in range(X_tfidf.shape[1])],
        'metadata': metadata,
        'tfidf_vectorizer': tfidf
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Saved features to {output_path}")


def main():
    """Extract features from train/val/test sets"""
    
    data_dir = Path("../promptshield/ml_data")
    
    print("=" * 60)
    print("Feature Extraction")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        input_file = data_dir / f"{split}.jsonl"
        output_file = data_dir / f"{split}_features.pkl"
        
        print(f"\n[{split.upper()}]")
        extract_features_batch(str(input_file), str(output_file))
    
    print("\n" + "=" * 60)
    print("✅ Feature extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
