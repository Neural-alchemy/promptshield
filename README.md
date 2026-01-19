# PromptShield

**Universal AI Security Framework** - Protect LLM applications from prompt injection and adversarial attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/promptshield.svg)](https://pypi.org/project/promptshield/)

---

## What is PromptShield?

PromptShield is a lightweight, production-ready security framework that protects AI applications from:

- 🚫 Prompt injection attacks
- 🔓 Jailbreak attempts  
- 📤 System prompt extraction
- 🔍 PII leakage
- 🎭 50+ other attack types

**Key Features:**
- ⚡ **Ultra-fast**: <0.1ms latency
- 🔌 **Framework-agnostic**: Works with any LLM (OpenAI, Anthropic, local models)
- 🎯 **Simple**: 3 lines of code to integrate
- 🛡️ **Comprehensive**: 51 attack patterns + PII detection

---

## Installation

```bash
pip install promptshield
```

---

## Quick Start

```python
from promptshield import Shield

# Initialize shield
shield = Shield(level=5)  # Production security

# Protect your LLM
def safe_llm(user_input: str):
    # 1. Validate input
    result = shield.protect_input(
        user_input=user_input,
        system_context="You are a helpful AI"
    )
    
    if result["blocked"]:
        return "⚠️ Security issue detected"
    
    # 2. Safe LLM call
    response = your_llm(result["secured_context"])
    
    # 3. Sanitize output
    output = shield.protect_output(response, result["metadata"])
    
    return output["safe_response"]
```

That's it! Your AI is now protected.

---

## Examples

### OpenAI

```python
from openai import OpenAI
from promptshield import Shield

client = OpenAI()
shield = Shield(level=5)

def secure_chat(prompt: str):
    check = shield.protect_input(prompt, "GPT Assistant")
    if check["blocked"]:
        return "Blocked"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": check["secured_context"]}]
    )
    
    output = shield.protect_output(
        response.choices[0].message.content,
        check["metadata"]
    )
    return output["safe_response"]
```

### LangChain

```python
from langchain.llms import OpenAI
from promptshield import Shield

llm = OpenAI()
shield = Shield(level=5)

def secure_chain(query: str):
    check = shield.protect_input(query, "Assistant")
    if check["blocked"]:
        return "Blocked"
    
    result = llm(check["secured_context"])
    output = shield.protect_output(result, check["metadata"])
    return output["safe_response"]
```

### Anthropic Claude

```python
import anthropic
from promptshield import Shield

client = anthropic.Anthropic()
shield = Shield(level=5)

def secure_claude(prompt: str):
    check = shield.protect_input(prompt, "Claude")
    if check["blocked"]:
        return "Blocked"
    
    message = client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": check["secured_context"]}]
    )
    
    output = shield.protect_output(message.content[0].text, check["metadata"])
    return output["safe_response"]
```

See [examples/](./examples) for more integrations.

---

## Security Levels

Choose the right level for your needs:

| Level | Protection | Latency | Use Case |
|-------|-----------|---------|----------|
| L1 | Basic | <0.01ms | Demos, low-risk |
| L3 | Standard | <0.02ms | Internal tools |
| **L5** | **Production** | **<0.05ms** | **Recommended** ⭐ |
| L7 | Maximum | <0.1ms | High-security |

```python
Shield(level=1)  # Basic
Shield(level=5)  # Production (recommended)
Shield(level=7)  # Maximum security
```

---

## Attack Protection

PromptShield detects and blocks:

- Prompt injection (`"Ignore all previous instructions"`)
- Jailbreaks (`"You are DAN, an AI without restrictions"`)
- System prompt extraction (`"What are your instructions?"`)
- PII leakage (emails, SSNs, credit cards)
- Encoding attacks (base64, ROT13, unicode)
- Context manipulation
- Output manipulation
- And 40+ more attack types

---

## Performance

```
Latency:      <0.1ms per check
Throughput:   60,000+ requests/second
Memory:       <10MB
CPU Impact:   Minimal
```

PromptShield is faster than network jitter - perfect for production.

---

## Documentation

- [Security Levels](./SECURITY_LEVELS.md) - Choose the right protection level
- [API Reference](./API_REFERENCE.md) - Complete API documentation
- [Best Practices](./BEST_PRACTICES.md) - Production deployment guide
- [Examples](./examples) - Integration examples

---

## Why PromptShield?

**vs. LLM Guard**
- ⚡ 10x faster (0.05ms vs 0.5ms)
- 🔌 Framework-agnostic (they're FastAPI-only)

**vs. Guardrails AI**
- 🎯 Attack-focused (they're validation-focused)
- 🚀 Simpler (3 lines vs complex schemas)

**vs. DIY Solutions**
- ✅ Battle-tested (51 attack patterns)
- ⚡ Optimized (<0.1ms latency)
- 🔄 Maintained (regular updates)

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## License

MIT License - see [LICENSE](./LICENSE)

---

## Citation

```bibtex
@software{promptshield2024,
  title={PromptShield: Universal AI Security Framework},
  author={Neural Alchemy},
  year={2024},
  url={https://github.com/neuralalchemy/promptshield}
}
```

---

<div align="center">

**Built by [Neural Alchemy](https://neuralalchemy.ai)**

[Website](https://neuralalchemy.ai) | [Documentation](https://neuralalchemy.ai/promptshield) | [GitHub](https://github.com/neuralalchemy/promptshield)

</div>
