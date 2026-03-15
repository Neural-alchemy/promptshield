

# PromptShield

**Secure AI Applications in 3 Lines of Code**

[![PyPI](https://img.shields.io/pypi/v/promptshields.svg)](https://pypi.org/project/promptshields/)
[![Python](https://img.shields.io/pypi/pyversions/promptshields.svg)](https://pypi.org/project/promptshields/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/promptshields)](https://pepy.tech/project/promptshields)

An enterprise-grade, bidirectional LLM security framework. Defend against prompt injection, jailbreaks, data leakage, and PII exposure in production applications.

--------------------------------------------------------------------------------

## Installation

```bash
pip install promptshields
```

## Quick Start

```python
from promptshield import Shield

shield = Shield.balanced()
result = shield.protect_input(user_input, system_prompt)

if result['blocked']:
    print(f"Blocked: {result['reason']} (score: {result['threat_level']:.2f})")
    print(f"Breakdown: {result['threat_breakdown']}")
```

--------------------------------------------------------------------------------

## Features & Capabilities

| Feature | PromptShields | DIY Regex | Paid APIs |
|---------|---------------|-----------|-----------| 
| **Setup Time** | 3 minutes | Weeks | Days |
| **Cost** | Free | Free | $$$$ |
| **Privacy** | 100% Local | Local | Cloud |
| **F1 Score** | 0.97 (RF) / 0.96 (DeBERTa) | ~0.60 | ~0.95 |
| **ML Models** | 4 + DeBERTa | None | Black box |
| **Async** | Native | DIY | Varies |

### Protection Scope
- Prompt injection attacks (direct and indirect)
- Jailbreak attempts (DAN, persona replacement)
- System prompt extraction
- PII leakage and sensitive data exposure
- Session anomalies
- Encoded/obfuscated attacks (Base64, URL, Unicode)

--------------------------------------------------------------------------------

## Security Modes

Choose the right tier for your application latency requirements:

```python
Shield.fast()       # ~1ms  - High throughput (pattern matching only)
Shield.balanced()   # ~2ms  - Production default (patterns + session tracking)
Shield.strict()     # ~7ms  - Sensitive apps (+ 1 ML model + PII detection)
Shield.secure()     # ~12ms - Maximum security (4 ML models ensemble)
```

--------------------------------------------------------------------------------

## Upgrading to v3.0.0

Version 3.0.0 introduces a massive update with the new bidirectional **Output Filter**.

### Output Engine (Data Leakage Prevention)
Prevent sensitive data, PII, and proprietary knowledge from leaking through LLM generations securely before they reach the user.

- **4-Layer Scanning Pipeline:** Defends against data leakage using Bloom Filters, Aho-Corasick exact matching, Honeypot traps, and Embedding-based Semantic Similarity checks.
- **Semantic Leakage Detection:** Natively utilizes `sentence-transformers` to detect when the LLM's output is highly semantically similar to your proprietary system prompts or private databases.
- **Contextual PII Redaction:** A heavily-optimized detection system to proactively redact sensitive information securely.

```python
from promptshield import OutputFilter

filter = OutputFilter(
    system_prompt="You are a secret agent...",
    enforce_pii=True,
    enforce_embeddings=True
)

safe_text, was_redacted = filter.scan_output("My name is John Doe.")
```

### Performance & Hardening
- Complete thread-safety for multi-tenant high-concurrency environments.
- Strict HMAC-SHA256 authenticated webhooks.
- Lazy-loading implementation for heavy dependencies (`numpy`, `sentence-transformers`) for lightning-fast cold starts.

--------------------------------------------------------------------------------

## Developer Experience

### YAML Configuration
Launch shields declaratively without changing application code.
```python
shield = Shield.from_config("promptshield.yml")
```

### Slack and Teams Webhooks
Instantly trigger webhooks whenever high-severity threats are blocked natively.
```python
shield = Shield.balanced(webhook_url="https://hooks.slack.com/...")
```

### Async and FastAPI Support
Native middleware integration for modern web frameworks.
```python
from promptshield import Shield
from promptshield.integrations.fastapi import PromptShieldMiddleware

app.add_middleware(PromptShieldMiddleware, shield=Shield.balanced())
```

--------------------------------------------------------------------------------

## Benchmark Results

Trained on the highly curated [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset):

| Model | F1 | ROC-AUC | FPR | Latency |
|-------|-----|---------|------|---------|
| Random Forest | **0.969** | **0.994** | 6.9% | <1ms |
| Logistic Regression | 0.964 | 0.995 | 6.4% | <1ms |
| Gradient Boosting | 0.961 | 0.994 | 7.9% | <1ms |
| LinearSVC | 0.959 | 0.995 | 10.3% | <1ms |
| DeBERTa-v3-small | 0.959 | 0.950 | 8.5% | ~50ms |

Pre-trained models available on Hugging Face: 
- [prompt-injection-detector](https://huggingface.co/neuralchemy/prompt-injection-detector)
- [prompt-injection-deberta](https://huggingface.co/neuralchemy/prompt-injection-deberta)

--------------------------------------------------------------------------------

## Documentation

Full API reference, guides, and integration details are available at the **[PromptShield Documentation Portal](https://doc.neuralchemy.in)**.

--------------------------------------------------------------------------------

## License

MIT License — see [LICENSE](LICENSE)

**Built by [NeurAlchemy](https://github.com/Neural-alchemy)** — AI Security and LLM Safety Research
