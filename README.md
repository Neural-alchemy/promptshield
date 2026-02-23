# PromptShields

c
**Enterprise LLM Security in 3 Lines of Code**
 
```
Before using our tool, you can test it against prompt injection attacks.

Step 1: Implement the tool.
Step 2: Find different prompt injection attacks from promptshield/attack_db.
Step 3: Use those attacks to test your implementation (see DOCUMENTATION.md).

```


[![PyPI](https://img.shields.io/pypi/v/promptshields.svg)](https://pypi.org/project/promptshields/)
[![Python](https://img.shields.io/pypi/pyversions/promptshields.svg)](https://pypi.org/project/promptshields/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/promptshields)](https://pepy.tech/project/promptshields)

Stop prompt injection, jailbreaks, and data leaks in production LLM applications.

---

## Installation
latest version 2.1.4 
```bash
pip install promptshields
```

## Basic Usage

```python
from promptshield import Shield

shield = Shield.balanced()
result = shield.protect_input(user_input, system_prompt)

if result['blocked']:
    return {"error": "Unsafe input detected"}
```

**That's it.** Production-ready security in 3 lines.

---

## Why PromptShields?

| Feature | PromptShields | DIY Regex | Paid APIs |
|---------|---------------|-----------|-----------|
| **Setup Time** | 3 minutes | Weeks | Days |
| **Cost** | Free | Free | $$$$ |
| **Privacy** | 100% Local | Local | Cloud |
| **Accuracy** | 98% | ~60% | ~95% |
| **ML Models** | Included | None | Black box |

### What We Block
- Prompt injection attacks
- Jailbreak attempts  
- System prompt extraction
- PII leakage
- Session anomalies

---



**Don't use one shield everywhere.** Layer them strategically:

## Security Modes

Choose the right tier for your application:

```python
Shield.fast()       # ~1ms  - High throughput (pattern matching)
Shield.balanced()   # ~2ms  - Production default (patterns + session tracking)
Shield.strict()     # ~7ms  - Sensitive apps (+ 1 ML model + PII detection)
Shield.secure()     # ~12ms - Maximum security (+ 3 ML models ensemble)
```

---

## Documentation

 **[Full Documentation](DOCUMENTATION.md)** - Complete guide with framework integrations

**[Quickstart Guide](QUICKSTART.md)** - Get running in 5 minutes

---

## License

MIT License - see [LICENSE](LICENSE)

PromptShield is released under the MIT License.

Neural Alchemy maintains the canonical upstream implementation.
Forks and third-party copies may not include the latest security
updates, ML-based detection, or attack coverage.


---

## Links

-  [PyPI Package](https://pypi.org/project/promptshields/)
-  [GitHub Repository](https://github.com/Neural-alchemy/promptshield)
-  [Documentation](DOCUMENTATION.md)

---

**Built by [Neuralchemy](https://github.com/Neural-alchemy)**
