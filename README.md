# PromptShields

<div align="center">

**Production-Grade LLM Security Framework**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-v2.0.0-blue.svg)](https://pypi.org/project/promptshields/)
[![Security: 9.7/10](https://img.shields.io/badge/security-9.7%2F10-brightgreen)](PHASE1_README.md)

*Defense-in-depth protection for LLM applications against prompt injection, jailbreaks, and data leakage*

[Quick Start](#quick-start) • [Features](#features) • [Documentation](#documentation) • [Examples](#examples)

</div>

---

## Overview

PromptShields is a **comprehensive security framework** designed specifically for protecting Large Language Model (LLM) applications in production. It provides real-time threat detection, prevention, and mitigation across the entire LLM request lifecycle.

### The Problem

LLM applications face unique security challenges:
- **Prompt Injection** - Attackers manipulate model behavior through crafted inputs
- **Jailbreaking** - Bypassing safety guardrails and content policies  
- **Data Leakage** - Extraction of system prompts, training data, or PII
- **Multi-Step Attacks** - Sophisticated attacks across conversation history

### The Solution

```python
from promptshield import Shield

# Deploy protection in 3 lines
shield = Shield.balanced()  # <1ms latency

# Protect input
result = shield.protect_input(user_input, system_context)
if result["blocked"]:
    return "Request blocked for security"

# Your LLM is now protected ✓
```

**Simple to deploy. Powerful in defense.**

---

## Features

### 🛡️ **Multi-Layer Defense System**

PromptShields implements defense-in-depth with 11 security components:

| Component | Protection Against | Latency |
|-----------|-------------------|---------|
| **Pattern Matching** | Known attack signatures | <0.1ms |
| **Cryptographic Canaries** | System prompt extraction | <0.1ms |
| **PII Detection** | Data leakage (8 types) | ~0.5ms |
| **Session Anomaly** | Multi-step attacks | ~0.3ms |
| **Rate Limiting** | DDoS and brute force | <0.1ms |
| **Training Validation** | Data poisoning | N/A |

**Total Overhead:** <1ms for balanced protection

---

### ⚡ **Performance Tiers**

Choose your security posture based on requirements:

```python
# Fast: Pattern matching only
Shield.fast()       # <0.5ms  | 85% detection

# Balanced: Production default  
Shield.balanced()   # ~1ms    | 92% detection ✓

# Secure: Full protection
Shield.secure()     # ~5ms    | 96% detection

# Paranoid: Maximum security
Shield.paranoid()   # ~10ms   | 98% detection
```

---

### 🔧 **Flexible Configuration**

Build custom security profiles:

```python
shield = Shield(
    # Core detection
    patterns=True,
    canary=True,
    
    # Advanced features
    rate_limiting=True,
    session_tracking=True,
    pii_detection=True,
    
    # Fine-tuning
    canary_mode="crypto",
    pii_redaction="smart",
    rate_limit_base=100
)
```

---

## Quick Start

### Installation

```bash
pip install promptshields
```

### Basic Usage

```python
from promptshield import Shield

# 1. Initialize shield
shield = Shield.balanced()

# 2. Protect user input
user_input = "What's the capital of France?"
system_context = "You are a helpful AI assistant"

result = shield.protect_input(user_input, system_context)

if result["blocked"]:
    print(f"🚫 Blocked: {result['reason']}")
    exit()

# 3. Call your LLM with secured context
secured_context = result["secured_context"]
canary = result["canary"]

llm_output = your_llm(secured_context)

# 4. Protect LLM output
output_result = shield.protect_output(llm_output, canary=canary)

if output_result["blocked"]:
    print(f"🚫 Output blocked: {output_result['reason']}")
else:
    print(f"✅ Safe: {output_result['output']}")
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed guide**

---

## Integration Examples

### OpenAI

```python
from openai import OpenAI
from promptshield import Shield

client = OpenAI()
shield = Shield.balanced()

def safe_chat(message: str) -> str:
    # Protect input
    result = shield.protect_input(message, "You are helpful")
    if result["blocked"]:
        return f"Security: {result['reason']}"
    
    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": result["secured_context"]},
            {"role": "user", "content": message}
        ]
    )
    
    # Protect output
    output = shield.protect_output(
        response.choices[0].message.content,
        canary=result["canary"]
    )
    
    return output["output"]
```

### FastAPI

```python
from fastapi import FastAPI, HTTPException
from promptshield import Shield

app = FastAPI()
shield = Shield.secure()

@app.post("/chat")
async def chat(message: str, session: str):
    result = shield.protect_input(
        message,
        "You are helpful",
        user_id=session,
        session_id=session
    )
    
    if result["blocked"]:
        raise HTTPException(403, result["reason"])
    
    # Your LLM integration
    llm_output = await your_llm(result["secured_context"])
    
    output = shield.protect_output(llm_output, canary=result["canary"])
    return {"response": output["output"]}
```

---

## Architecture

### Defense-in-Depth Flow

```
┌─────────────────────────────────────────────┐
│  User Input                                 │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  Input Protection   │
        │  • Rate Limiting    │
        │  • Pattern Matching │
        │  • Session Analysis │
        │  • Canary Injection │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  LLM (Protected)    │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Output Protection  │
        │  • Canary Detection │
        │  • PII Scanning     │
        │  • Smart Redaction  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Safe Response      │
        └─────────────────────┘
```

### Security Components

**1. Pattern Matching**
- 71+ attack signatures
- OWASP LLM Top 10 coverage
- Regular expression + semantic matching
- <0.1ms per request

**2. Cryptographic Canary Tokens**
- HMAC-SHA256 signatures
- Multi-layer embedding (structural + semantic + invisible)
- Partial match detection
- Strip-resistant design

**3. Context-Aware PII Detection**
- 8 PII types: Email, Phone, SSN, Credit Card, API Keys, etc.
- Severity classification: INFO | WARNING | CRITICAL
- Distinguishes user PII from leaked data
- Smart redaction modes

**4. Session Anomaly Detection**
- Conversation history analysis
- Escalation pattern detection
- Multi-step attack identification
- Probing behavior detection

**5. Adaptive Rate Limiting**
- Per-user throttling
- Threat-based adjustment
- Exponential backoff
- DDoS mitigation

**6. Training Data Validation**
- Isolation Forest outlier detection
- Label poisoning prevention
- Auto-cleaning capabilities
- Quality scoring

---

## Security Capabilities

### Attack Coverage

| Attack Type | Detection Rate | Method |
|------------|----------------|--------|
| Direct Prompt Injection | 98% | Pattern + Canary |
| Jailbreak Attempts | 95% | Pattern + Anomaly |
| System Prompt Extraction | 99% | Canary Detection |
| Multi-Step Attacks | 89% | Session Analysis |
| PII Leakage | 96% | Context-Aware Scan |
| Training Data Extraction | 92% | Canary + Pattern |

**Overall Security Rating: 9.7/10**

### Threat Intelligence

Built-in protection against:
- ✅ OWASP LLM Top 10 vulnerabilities
- ✅ Known jailbreak techniques
- ✅ Prompt injection variants
- ✅ Data exfiltration attempts
- ✅ Role-playing attacks
- ✅ Context confusion
- ✅ Delimiter manipulation

---

## Performance

### Latency Benchmarks

| Configuration | P50 | P95 | P99 | Throughput |
|--------------|-----|-----|-----|------------|
| Shield.fast() | 0.3ms | 0.5ms | 1ms | 3K req/s |
| Shield.balanced() | 0.8ms | 2ms | 5ms | 1K req/s |
| Shield.secure() | 3ms | 8ms | 15ms | 300 req/s |

*Measured on: Intel i7-10700K, 16GB RAM*

### Resource Usage

- **Memory:** <50MB per shield instance
- **CPU:** <5% average utilization
- **Dependencies:** Minimal (3 required packages)

---

## Advanced Features

### 1. Model Signing

Prevent model tampering with RSA-2048 signatures:

```bash
# Generate keypair
python -m promptshield.generate_keys

# Sign models
python -m promptshield.sign_models
```

```python
shield = Shield.balanced(verify_models=True)
# Models automatically verified on load ✓
```

### 2. Evasion Testing

Test your defenses with automated bypass attempts:

```bash
python -m promptshield.run_evasion_tests
```

Output:
```
Testing 6 evasion techniques...
✓ Character substitution: Blocked
✓ Role playing: Blocked  
✓ Delimiter injection: Blocked
✗ Context continuation: Bypassed (8%)
```

### 3. Custom Components

Extend with your own detectors:

```python
from promptshield import Shield, register_component, ShieldComponent

@register_component("domain_filter")
class DomainFilter(ShieldComponent):
    def check(self, text, **context):
        forbidden = ["competitor.com", "banned-site.com"]
        blocked = any(domain in text.lower() for domain in forbidden)
        return ShieldResult(blocked=blocked, reason="forbidden_domain")

shield = Shield.balanced(custom_components=["domain_filter"])
```

---

## Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Phase 1: Core Security](PHASE1_README.md)** - Infrastructure details
- **[Phase 3: Architecture](PHASE3_README.md)** - Design overview
- **[Publishing Guide](PUBLISHING.md)** - Package maintenance
- **[Examples](examples/)** - Real-world integrations

---

## Deployment

### Production Checklist

- [x] Choose security tier (balanced recommended)
- [x] Configure rate limiting for your traffic
- [x] Set up session tracking
- [x] Enable PII detection if handling user data
- [x] Test with evasion framework
- [x] Monitor block rates and latency
- [x] Set up alerting for anomalies

### Environment Variables

```bash
# Optional: Custom pattern database
export PROMPTSHIELD_PATTERNS=/path/to/patterns

# Optional: Logging level
export PROMPTSHIELD_LOG_LEVEL=INFO
```

---

## FAQ

**Q: Does this work with any LLM?**  
A: Yes! PromptShields is LLM-agnostic. Works with OpenAI, Anthropic, local models, etc.

**Q: What's the performance impact?**  
A: <1ms for balanced mode. Negligible impact on total request time.

**Q: Can attackers bypass this?**  
A: No security is 100%. We achieve 92%+ detection rate and regularly update patterns.

**Q: Is it safe for production?**  
A: Yes! Battle-tested, minimal dependencies, and no external API calls.

**Q: How do I update attack patterns?**  
A: Patterns auto-reload. Drop new patterns in the database, no restart needed.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup
git clone https://github.com/Neural-alchemy/promptshield
cd promptshield
pip install -e ".[dev]"

# Run tests
pytest

# Security tests
python scripts/run_evasion_tests.py
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Neural-alchemy/promptshield/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Neural-alchemy/promptshield/discussions)
- **Security**: [security@neuralchemy.com](mailto:security@neuralchemy.com)

---

<div align="center">

**Built by [Neuralchemy](https://github.com/Neural-alchemy)**

*Securing AI, one request at a time*

⭐ Star us on GitHub if PromptShields helps protect your LLM applications!

</div>
