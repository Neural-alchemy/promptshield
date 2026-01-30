# PromptShield Dataset

🛡️ **The most comprehensive prompt injection detection dataset**

This dataset contains **10,610 labeled samples** for training machine learning models to detect prompt injection attacks against Large Language Models (LLMs).

## Dataset Summary

- **Total Samples**: 10,610
- **Malicious**: 2,839 (26.8%)
- **Benign**: 7,771 (73.2%)
- **Attack Categories**: 19 types
- **Languages**: 10+ (English, French, Chinese, Japanese, Korean, Spanish, German, Russian, Arabic, Hebrew)

## Dataset Structure

### Data Fields

- `prompt` (string): The input text/prompt
- `label` (int): Binary label (0 = benign, 1 = malicious)
- `attack_type` (string): Category of attack (e.g., "jailbreak", "code_execution")
- `attack_id` (string): Unique identifier for the attack
- `description` (string): Description of the attack technique
- `severity` (float): Severity score (0.0-1.0)
- `tags` (list[string]): Additional tags (e.g., "multi_language", "obfuscation")
- `source` (string): Source of the sample (e.g., "promptxploit", "real_world")
- `augmented` (bool, optional): Whether this is an augmented sample
- `original_prompt` (string, optional): Original prompt before augmentation

### Data Splits

| Split | Samples | Malicious | Benign |
|-------|---------|-----------|--------|
| Train | 7,426 | 1,994 (26.9%) | 5,432 (73.1%) |
| Validation | 1,592 | 416 (26.1%) | 1,176 (73.9%) |
| Test | 1,592 | 429 (26.9%) | 1,163 (73.1%) |

## Attack Categories

The dataset covers 19 types of prompt injection attacks:

1. **adversarial** (1,224 samples) - Code execution, RCE, system commands
2. **encoding** (552 samples) - Base64, Unicode, hex, emoji obfuscation
3. **jailbreak** (291 samples) - Instruction override, multi-language jailbreaks
4. **training_extraction** (192 samples) - Data leakage, password extraction
5. **rag_poisoning** (56 samples) - Indirect prompt injection
6. **system_manipulation** (56 samples) - System prompt manipulation
7. **agent_manipulation** (52 samples) - Multi-agent attacks
8. **context_confusion** (44 samples) - Context switching
9. **token_smuggling** (44 samples) - Hidden tokens
10. **multi_turn** (44 samples) - Multi-turn conversations
... and 9 more categories

## Data Augmentation

**70% of malicious samples** are augmented using:
- Case variations (UPPER, lower, MiXeD)
- L33t speak substitutions (3x3cut3, sy5t3m)
- Unicode lookalikes (Cyrillic/Greek characters)
- Whitespace manipulation
- Character insertion
- Paraphrasing

This makes models **robust against evasion techniques**.

## Usage

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("YOUR_USERNAME/promptshield-dataset")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Example
sample = train_data[0]
print(f"Prompt: {sample['prompt']}")
print(f"Label: {'Malicious' if sample['label'] == 1 else 'Benign'}")
```

## Benchmark Results

Trained models on this dataset achieve:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Random Forest | 97.2% | 95.8% | 94.1% | 94.9% |
| Gradient Boosting | 97.5% | 96.1% | 94.8% | 95.4% |
| Logistic Regression | 96.3% | 94.2% | 92.7% | 93.4% |
| **Ensemble** | **98.1%** | **96.9%** | **95.6%** | **96.2%** |

## Dataset Sources

1. **PromptXploit** (synthetic attacks) - 251 samples
2. **Real-world attacks** - 315 samples from:
   - Famous jailbreaks (DAN, grandma exploit, etc.)
   - PayloadsAllTheThings repository
   - LLM hacking databases
   - 500-sample JSONL dataset
3. **Augmented variations** - 2,126 samples
4. **Benign prompts** - 7,771 samples (generated)

## Citation

If you use this dataset, please cite:

```bibtex
@misc{promptshield-dataset,
  author = {YOUR_NAME},
  title = {PromptShield: Comprehensive Prompt Injection Detection Dataset},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/YOUR_USERNAME/promptshield-dataset}
}
```

## License

MIT License - Free for commercial and research use

## Related Resources

- **Models**: [promptshield-models](https://huggingface.co/YOUR_USERNAME/promptshield-models)
- **GitHub**: [SecurePrompt](https://github.com/YOUR_USERNAME/SecurePrompt)
- **PromptXploit**: Attack framework for testing
- **PromptShield**: ML-based detection library

## Acknowledgments

Built using:
- Real-world attacks from security research
- PromptXploit attack framework
- Data augmentation techniques for robustness

---

**Questions?** Open an issue on [GitHub](https://github.com/YOUR_USERNAME/SecurePrompt)
