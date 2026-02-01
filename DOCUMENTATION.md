# PromptShields Documentation

Complete guide for integrating PromptShields into your LLM applications.

---

## Table of Contents

### Getting Started
- [Quick Installation](#quick-installation)
- [Basic Usage](#basic-usage)
- [Shield Modes](#shield-modes)

### Framework Integrations
- [LangChain Integration](#langchain-integration)
- [LangGraph Integration](#langgraph-integration)
- [OpenAI Integration](#openai-integration)
- [LlamaIndex Integration](#llamaindex-integration)
- [Custom Frameworks](#custom-frameworks)

### Advanced Usage
- [Configuration Options](#configuration-options)
- [Multi-Agent Systems](#multi-agent-systems)
- [Session Tracking](#session-tracking)
- [Custom Thresholds](#custom-thresholds)

### API Reference
- [Shield Class](#shield-class)
- [Response Format](#response-format)
- [Available Models](#available-models)

---

## Quick Installation

```bash
pip install promptshields
```

**Requirements:** Python 3.8+

---

## Basic Usage

```python
from promptshield import Shield

# Create shield
shield = Shield.balanced()

# Protect input
result = shield.protect_input(
    user_input="Your user's input here",
    system_context="Your system prompt here"
)

# Check result
if result['blocked']:
    print(f"Blocked: {result['reason']}")
else:
    # Safe to proceed
    response = your_llm_call(user_input)
```

---

## Shield Modes

### Shield.fast()
**Speed:** ~1ms | **Use Case:** High-throughput APIs

```python
shield = Shield.fast()
```

**Features:**
- Pattern matching (71 attack patterns)

---

### Shield.balanced() ⭐
**Speed:** ~2ms | **Use Case:** Production default

```python
shield = Shield.balanced()
```

**Features:**
- Pattern matching
- Session anomaly tracking

---

### Shield.strict()
**Speed:** ~7ms | **Use Case:** Sensitive applications

```python
shield = Shield.strict()
```

**Features:**
- Pattern matching
- Session tracking
- 1 ML model (logistic regression)
- PII detection & redaction
- Rate limiting

---

### Shield.secure()
**Speed:** ~12ms | **Use Case:** Maximum security

```python
shield = Shield.secure()
```

**Features:**
- All strict features
- 3 ML models (ensemble)
- Canary token protection

---

## LangChain Integration

### Basic Wrapper

```python
from langchain.chains import LLMChain
from promptshield import Shield

shield = Shield.strict()

class SecureLLMChain(LLMChain):
    def run(self, *args, **kwargs):
        user_input = kwargs.get('input', '')
        
        # Validate input
        result = shield.protect_input(user_input, self.prompt.template)
        if result['blocked']:
            raise ValueError(f"Security violation: {result['reason']}")
        
        # Execute chain
        return super().run(*args, **kwargs)
```

### Agent Executor Protection

```python
from langchain import agents
from promptshield import Shield

shield = Shield.balanced()

# Wrap agent execution
original_run = agent_executor.run

def secure_run(input_text):
    result = shield.protect_input(input_text, "")
    if result['blocked']:
        return f"Error: {result['reason']}"
    return original_run(input_text)

agent_executor.run = secure_run
```

---

## LangGraph Integration

```python
from langgraph.graph import StateGraph
from promptshield import Shield

shield = Shield.strict()

def secure_node(state):
    """Secure graph node"""
    user_message = state['messages'][-1]
    
    # Validate before processing
    result = shield.protect_input(user_message, state.get('system_prompt', ''))
    if result['blocked']:
        return {"error": result['reason'], "blocked": True}
    
    # Process normally
    return your_processing_logic(state)

# Add to graph
graph = StateGraph()
graph.add_node("secure_input", secure_node)
```

---

## OpenAI Integration

### Drop-in Replacement

```python
import openai
from promptshield import Shield

shield = Shield.balanced()

def secure_chat(messages, **kwargs):
    """Secure OpenAI chat wrapper"""
    last_message = messages[-1]['content']
    
    # Validate
    result = shield.protect_input(last_message, "")
    if result['blocked']:
        return {
            "error": result['reason'],
            "threat_level": result['threat_level']
        }
    
    # Call OpenAI
    return openai.ChatCompletion.create(
        messages=messages,
        **kwargs
    )
```

---

## LlamaIndex Integration

```python
from llama_index import VectorStoreIndex
from promptshield import Shield

shield = Shield.balanced()

def secure_query(index, query_text):
    """Secure query wrapper"""
    result = shield.protect_input(query_text, "")
    
    if result['blocked']:
        return f"Query blocked: {result['reason']}"
    
    return index.query(query_text)
```

---

## Custom Frameworks

### Generic Agent Wrapper

```python
from promptshield import Shield

class SecureAgent:
    def __init__(self, system_prompt):
        self.shield = Shield.strict()
        self.system_prompt = system_prompt
    
    def execute(self, user_task):
        # Validate input
        result = self.shield.protect_input(user_task, self.system_prompt)
        
        if result['blocked']:
            return self.handle_security_violation(result)
        
        # Execute task
        return self.process_task(user_task)
    
    def handle_security_violation(self, result):
        # Log, alert, or return error
        return {"error": "Security violation", "details": result['reason']}
```

---

## Configuration Options

### Custom Shield

```python
shield = Shield(
    patterns=True,                      # Enable pattern matching
    models=["logistic_regression"],     # ML models to use
    model_threshold=0.7,                # Confidence threshold (0.0-1.0)
    session_tracking=True,              # Track user sessions
    pii_detection=True,                 # Detect PII
    rate_limiting=True,                 # Rate limit users
    canary=False,                       # Canary tokens
)
```

### Override Presets

```python
# Add ML to balanced mode
shield = Shield.balanced(models=["svm"])

# Remove ML from strict
shield = Shield.strict(models=None)

# Adjust threshold
shield = Shield.secure(model_threshold=0.9)
```

---

## Multi-Agent Systems

### Layered Defense

```python
from promptshield import Shield

# Different shields for different trust levels
user_shield = Shield.secure()      # User input (untrusted)
agent_shield = Shield.balanced()   # Agent-to-agent (semi-trusted)
internal_shield = Shield.fast()    # Internal APIs (trusted)

def process_request(user_input):
    # Layer 1: User input
    if user_shield.protect_input(user_input, ctx)['blocked']:
        return {"error": "Invalid input"}
    
    # Layer 2: Agent processing
    agent_output = agent.process(user_input)
    if agent_shield.protect_input(agent_output, ctx)['blocked']:
        alert_security_team()
        return {"error": "Security violation"}
    
    # Layer 3: Internal check
    if internal_shield.protect_input(agent_output, "")['blocked']:
        log_anomaly()
    
    return {"data": agent_output}
```

---

## Session Tracking

```python
shield = Shield.strict()

# Track user sessions
for message in conversation:
    result = shield.protect_input(
        user_input=message,
        system_context=prompt,
        user_id="user_12345"  # Session identifier
    )
    
    if result['blocked']:
        # Alert if anomaly detected (e.g., repeated attacks)
        handle_security_event(result)
```

---

## Custom Thresholds

```python
# Conservative (fewer false positives)
shield = Shield(
    models=["random_forest"],
    model_threshold=0.9  # Higher threshold
)

# Aggressive (catch more attacks)
shield = Shield(
    models=["logistic_regression", "svm"],
    model_threshold=0.5  # Lower threshold
)
```

---

## Shield Class

### Initialization

```python
Shield(
    patterns: bool = True,
    models: List[str] = None,
    model_threshold: float = 0.7,
    session_tracking: bool = False,
    pii_detection: bool = False,
    rate_limiting: bool = False,
    canary: bool = False,
)
```

### Methods

#### `protect_input(user_input, system_context, user_id=None)`

Validates user input against configured protections.

**Parameters:**
- `user_input` (str): User's message/input
- `system_context` (str): System prompt or context
- `user_id` (str, optional): User identifier for session tracking

**Returns:** `dict` - See [Response Format](#response-format)

---

## Response Format

```python
{
    "blocked": bool,           # Should input be blocked?
    "reason": str,             # Why? ("pattern_match", "ml_model", "pii_detected", etc.)
    "threat_level": float,     # Threat score (0.0 - 1.0)
    "metadata": dict,          # Additional context
}
```

### Example Responses

**Blocked (Pattern Match):**
```python
{
    "blocked": True,
    "reason": "pattern_match",
    "threat_level": 0.85,
    "metadata": {"component": "pattern_matcher", "rule": "injection_001"}
}
```

**Blocked (ML Model):**
```python
{
    "blocked": True,
    "reason": "ml_model",
    "threat_level": 0.92,
    "metadata": {"component": "ml", "models_triggered": 2}
}
```

**Allowed:**
```python
{
    "blocked": False,
    "reason": "safe",
    "threat_level": 0.12,
    "metadata": {}
}
```

---

## Available Models

### ML Models

- `logistic_regression` - Fast, good accuracy
- `random_forest` - Balanced speed/accuracy
- `svm` - Slower, best accuracy

### Usage

```python
# Single model
shield = Shield(models=["logistic_regression"])

# Ensemble (recommended for strict/secure)
shield = Shield(models=["logistic_regression", "random_forest", "svm"])
```

---

## Troubleshooting

### Import Errors

```bash
# Wrong package name
pip install promptshields  # Note the 's'
```

### Performance Issues

```python
# Use faster mode
shield = Shield.fast()

# Or disable ML
shield = Shield.strict(models=None)
```

### Too Many False Positives

```python
# Increase threshold
shield = Shield(model_threshold=0.9)
```

---

## Support

- 🐙 [GitHub Issues](https://github.com/Neural-alchemy/promptshield/issues)
- 📚 [Back to README](README.md)
