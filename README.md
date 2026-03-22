<p align="center">
  <img src="https://raw.githubusercontent.com/openclay-ai/openclay/main/docs/assets/logo.png" alt="OpenClay Logo" width="120"/>
</p>

<h1 align="center">OpenClay</h1>

<p align="center">
  <strong>Secure First → Execute Second.</strong><br/>
  A Neural Alchemy project. The universal, zero-trust execution framework for LLM agents.
</p>

<p align="center">
  <a href="https://pypi.org/project/openclay/"><img alt="PyPI" src="https://img.shields.io/pypi/v/openclay.svg"></a>
  <a href="https://github.com/neuralchemy/openclay"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://doc.neuralchemy.in"><img alt="Docs" src="https://img.shields.io/badge/docs-neuralchemy.in-orange"></a>
<a href="https://pepy.tech/projects/openclay"><img alt="OpenClay Downloads" src="https://static.pepy.tech/badge/openclay"></a>
<a href="https://pepy.tech/projects/promptshields"><img alt="PromptShields Legacy Downloads" src="https://static.pepy.tech/badge/promptshields"></a>
</p>

---

## Why OpenClay?

Every modern AI framework—LangChain, CrewAI, LlamaIndex—is built on an optimistic assumption: **trust the input, trust the tools, trust the memory.** OpenClay operates on the opposite principle.

> **You do not build an agent and then bolt on security.  
> You define a Security Policy, and the agent executes *inside* it.**

OpenClay wraps every single step — tool calls, memory reads/writes, model inputs and outputs — in a multi-layered shield before any execution ever happens.

---

## Installation

```bash
pip install openclay
```

Optional extras:

```bash
pip install openclay[ml]      # Scikit-learn ensemble models (RF, SVM, LR, GBT)
pip install openclay[embed]   # Sentence-Transformers for semantic similarity
pip install openclay[search]  # DuckDuckGo web fallback for output leakage detection
pip install openclay[all]     # Everything
```

---

## OpenClay v0.2.0 — The Secure Runtime 🚀

With v0.2.0, OpenClay graduates from a "shield library" to a **secure execution runtime**.

### 1. `ClayRuntime` (Protecting Agents & Callables)

Instead of manually calling a shield before and after execution, you wrap your execution logic in `ClayRuntime`. It forces inputs and outputs to pass through explicit trust boundaries.

```python
from openclay import ClayRuntime

# Create an execution environment with a strict policy
runtime = ClayRuntime(policy="strict")

# Shields fire automatically before input and after output
result = runtime.run(my_llm_chain, "Analyze this data", context=system_prompt)

if result.blocked:
    print(f"Blocked by layer: {result.trace.layer} — Reason: {result.trace.reason}")
else:
    print(result.output)
```

**Drop-in shielding for LangChain / CrewAI:**
```python
wrapped_agent = runtime.wrap(langchain_agent)
safe_result = wrapped_agent.run("research AI security")
```

**Explicit Exceptions:**
```python
# Bypass the input shield for a specific block of code
with runtime.disable("input"):
    result = runtime.run(my_chain, unsafe_input)
```

### 2. `ClayTool` (Securing Tool Blind-spots)

The biggest unaddressed vulnerability in agentic AI is blindly trusting external tools. If an LLM executes a database query and the DB returns a malicious SQL injection or prompt hijack, the agent is compromised.

`@ClayTool` intercepts and sanitizes tool outputs *before* they return to the agent's context.

```python
from openclay import ClayTool, Shield

@ClayTool(shield=Shield.balanced())
def search_web(query: str):
    return api.search(query)  # Returned data is automatically scanned!

try:
    search_web("malicious query")
except ToolOutputBlocked as e:
    print(f"Tool output was malicious! Blocked by rule: {e.trace.rule}")
```

---

## openclay.shields — Core Threat Detection Engine ✅

`openclay.shields` is the battle-tested security core of OpenClay, evolved from [PromptShield](https://github.com/neuralchemy/promptshield) v3.0 *(now deprecated — see [migration guide](#migration-promptshield--openclay) below)*.

### The Protection Pipeline

| Layer | Technology | What it catches | Latency |
|---|---|---|---|
| **1. Pattern Engine** | 600+ Aho-Corasick patterns | Injections, jailbreaks, encoding attacks | ~0.1ms |
| **2. Rate Limiter** | Adaptive per-user throttle | Flood / brute-force attacks | ~0.1ms |
| **3. Session Anomaly** | Sliding-window divergence | Multi-turn orchestrated attacks | ~0.5ms |
| **4. ML Ensemble** | TF-IDF + RF / LR / SVM / GBT | Semantic injection variants | ~5-10ms |
| **5. DeBERTa Classifier** | Fine-tuned transformer | Zero-day semantic threats | ~50ms |
| **6. Canary Tokens** | Cryptographic HMAC canaries | System prompt exfiltration | ~0.2ms |
| **7. PII Detector** | Contextual named-entity rules | Sensitive data leakage | ~1ms |
| **8. Output Engine** | Bloom filter + Aho-Corasick + embeddings | Leaked sensitive terms in LLM output | ~2ms |

---

### Low-Level Shield APIs

You can still use the core primitives manually if preferred:

```python
from openclay import Shield

# Balanced preset — production default (~1-2ms)
shield = Shield.balanced()

result = shield.protect_input(
    user_input="Ignore your previous instructions and...",
    system_context="You are a helpful assistant."
)

if result["blocked"]:
    print(f"🛡️ Blocked! Reason: {result['reason']}, Threat level: {result['threat_level']:.2f}")
else:
    print("✅ Input is safe.")
```

### Shield Presets

Four built-in presets to match your latency / security trade-off:

```python
# ⚡ Fast   — pattern-only, <1ms. Great for high-throughput APIs.
shield = Shield.fast()

# ⚖️ Balanced — patterns + session tracking, ~1-2ms. Production default.
shield = Shield.balanced()

# 🔒 Strict  — patterns + 1 ML model (Logistic Regression) + rate limiting + PII, ~7-10ms.
shield = Shield.strict()

# 🛡️ Secure  — all layers + full ML ensemble (RF + LR + SVM + GBT), ~12-15ms.
shield = Shield.secure()
```

---

## The OpenClay Ecosystem

| Module | Status | Description |
|---|---|---|
| `openclay.shields` | ✅ **Ready** | Core threat detection engine |
| `openclay.runtime` | ✅ **v0.2.0** | Secure execution wrapper (`ClayRuntime`) |
| `openclay.tools` | ✅ **v0.2.0** | `@ClayTool` decorator for output interception |
| `openclay.tracing` | ✅ **v0.2.0** | `Trace` explainability for every blocked action |
| `openclay.memory` | 🚧 Draft | Pre-write and pre-read poisoning prevention |
| `openclay.policies` | 🚧 Draft | `StrictPolicy`, `ModeratePolicy`, `CustomPolicy` |

---

## Migration: PromptShield → OpenClay

`promptshields` (v3.0.1) is now **sunset** and will receive no further updates.

### Step 1 — Update your dependency

```diff
- pip install promptshields
+ pip install openclay
```

### Step 2 — Update your imports

```diff
- from promptshield import Shield
+ from openclay import Shield

- from promptshield.integrations.langchain import PromptShieldCallbackHandler
+ from openclay.shields.integrations.langchain import OpenClayCallbackHandler

- from promptshield.integrations.fastapi import PromptShieldMiddleware
+ from openclay.shields.integrations.fastapi import OpenClayMiddleware

- from promptshield.integrations.litellm import PromptShieldLiteLLMCallback
+ from openclay.shields.integrations.litellm import OpenClayLiteLLMCallback

- from promptshield.integrations.crewai import PromptShieldCrewInterceptor
+ from openclay.shields.integrations.crewai import OpenClayCrewInterceptor
```

### Step 3 — Rename class usages

```diff
- PromptShieldCallbackHandler(shield=shield)
+ OpenClayCallbackHandler(shield=shield)

- PromptShieldMiddleware
+ OpenClayMiddleware

- PromptShieldLiteLLMCallback(shield=shield)
+ OpenClayLiteLLMCallback(shield=shield)

- PromptShieldCrewInterceptor(shield=shield)
+ OpenClayCrewInterceptor(shield=shield)
```

> [!NOTE]
> The core `Shield` API is fully backwards-compatible. Only the integration class names changed.

---

## Links

- 📦 [PyPI — `openclay`](https://pypi.org/project/openclay/)
- 📦 [PyPI — `promptshields` (deprecated)](https://pypi.org/project/promptshields/)
- 📖 [Documentation](https://doc.neuralchemy.in)
- 🤗 [Hugging Face — DeBERTa Model](https://huggingface.co/neuralchemy/prompt-injection-deberta)
- 🐛 [GitHub Issues](https://github.com/neuralchemy/openclay/issues)

---

<p align="center">
  Built with ❤️ by <a href="https://neuralchemy.in">Neural Alchemy</a>
</p>
