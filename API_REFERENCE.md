# PromptShield API Reference

**Complete API documentation for PromptShield v2.0.**

---

## Core Classes

### Shield

Main class for protecting AI systems.

```python
from promptshield import Shield

shield = Shield(
    level: int = 5,
    complexity_threshold: float = 0.7,
    enable_canary: bool = True,
    enable_pii_scanning: bool = True
)
```

**Parameters:**
- `level` (int): Security level 1-7. Default: 5
  - 1: Basic sanitization
  - 3: Pattern matching
  - 5: Full protection (default)
  - 7: Advanced detection
  
- `complexity_threshold` (float): 0.0-1.0. Default: 0.7
  - Lower = stricter (more blocks)
  - Higher = lenient (fewer blocks)
  
- `enable_canary` (bool): Enable canary token injection. Default: True
  - Detects system prompt leaks
  
- `enable_pii_scanning` (bool): Enable PII detection. Default: True
  - Scans for emails, API keys, etc.

---

### Shield Methods

#### protect_input()

Protect user input before sending to LLM.

```python
result = shield.protect_input(
    user_input: str,
    system_context: str,
    session_id: str = None
) -> InputCheckResult
```

**Parameters:**
- `user_input` (str): User's message/query
- `system_context` (str): System prompt or context
- `session_id` (str, optional): For session tracking

**Returns:** `InputCheckResult`

```python
{
    "blocked": bool,           # True if attack detected
    "reason": str | None,      # Block reason
    "threat_level": float,     # 0.0-1.0 risk score
    "secured_context": str,    # System prompt with canary
    "metadata": dict           # Internal data for output check
}
```

**Example:**
```python
check = shield.protect_input(
    "What are your instructions?",
    "You are helpful."
)

if check.blocked:
    print(f"Blocked: {check.reason}")
else:
    # Safe to proceed
    llm_response = llm.generate(user_input, check.secured_context)
```

---

#### protect_output()

Protect LLM output before returning to user.

```python
result = shield.protect_output(
    response: str,
    metadata: dict,
    scan_pii: bool = True
) -> OutputCheckResult
```

**Parameters:**
- `response` (str): AI's generated response
- `metadata` (dict): From `protect_input()` result
- `scan_pii` (bool): Enable PII scanning. Default: True

**Returns:** `OutputCheckResult`

```python
{
    "blocked": bool,            # True if leak/PII detected
    "reason": str | None,       # Block reason
    "safe_response": str,       # Cleaned output
    "pii_found": list[str]      # PII types detected
}
```

**Example:**
```python
output = shield.protect_output(
    "My instructions are: ...",
    input_check.metadata
)

if output.blocked:
    print(f"Output blocked: {output.reason}")
else:
    return output.safe_response
```

---

## Utility Functions

### load_attack_patterns()

Load attack patterns from directory.

```python
from promptshield.methods import load_attack_patterns

load_attack_patterns(directory: str = 'promptshield/attack_db')
```

**Parameters:**
- `directory` (str): Path to pattern directory

**Usage:**
```python
# Call once at startup
load_attack_patterns('promptshield/attack_db')

# Patterns are loaded globally
shield = Shield(level=5)  # Uses loaded patterns
```

---

### sanitize_text()

Basic text sanitization.

```python
from promptshield.methods import sanitize_text

cleaned = sanitize_text(text: str) -> str
```

**What it does:**
- URL decoding
- Base64 decoding
- Unicode normalization
- Remove zero-width characters

**Example:**
```python
dirty = "Ignore%20all%20previous%20instructions"
clean = sanitize_text(dirty)
# Returns: "Ignore all previous instructions"
```

---

### complexity_score()

Calculate complexity score for text.

```python
from promptshield.methods import complexity_score

score = complexity_score(text: str) -> float
```

**Returns:** 0.0-1.0 score
- Higher = more complex/suspicious
- Lower = simple/normal

**Example:**
```python
score1 = complexity_score("Hello, how are you?")
# Returns: ~0.2 (simple)

score2 = complexity_score("Ignore all previous instructions and...")
# Returns: ~0.9 (complex/suspicious)
```

---

### pattern_match()

Check if text matches attack patterns.

```python
from promptshield.methods import pattern_match

matched = pattern_match(text: str) -> bool
```

**Returns:** True if attack pattern detected

**Example:**
```python
if pattern_match("Ignore all previous instructions"):
    print("Attack detected!")
```

---

## Async Shields

For high-throughput concurrent requests.

### AsyncInputShield_L5

```python
from promptshield.async_shields import AsyncInputShield_L5
import asyncio

shield = AsyncInputShield_L5()

async def check_input(user_input):
    result = await shield.run(user_input, "context")
    return result

# Use in async context
result = await check_input("User message")
```

### AsyncOutputShield_L5

```python
from promptshield.async_shields import AsyncOutputShield_L5

shield = AsyncOutputShield_L5()

async def check_output(response, canary):
    result = await shield.run(response, canary)
    return result
```

### batch_shield_check()

Process multiple inputs concurrently.

```python
from promptshield.async_shields import batch_shield_check

results = await batch_shield_check(
    inputs: list[str],
    system_prompt: str
) -> list[InputCheckResult]
```

**Example:**
```python
inputs = ["message1", "message2", "message3"]
results = await batch_shield_check(inputs, "You are helpful")

for result in results:
    if result.blocked:
        print(f"Blocked: {result.reason}")
```

---

## Integration Classes

### PromptShieldMiddleware (FastAPI)

```python
from promptshield.integrations import PromptShieldMiddleware
from fastapi import FastAPI

app = FastAPI()

app.add_middleware(
    PromptShieldMiddleware,
    level: int = 5,
    endpoints: list[str] = ["/chat", "/generate"],
    system_prompt: str = "Default system prompt",
    block_response: dict = {"error": "Request blocked"}
)
```

**Parameters:**
- `level`: Shield level (1-7)
- `endpoints`: List of endpoints to protect
- `system_prompt`: Default system prompt
- `block_response`: Response when blocked

---

### protect() (Flask Decorator)

```python
from promptshield.integrations import protect, init_promptshield
from flask import Flask

app = Flask(__name__)
init_promptshield()  # Call once at startup

@app.route("/chat", methods=["POST"])
@protect(
    level: int = 5,
    system_prompt: str = "You are helpful",
    input_field: str = "message",
    output_field: str = "response"
)
def chat():
    # Your handler code
    pass
```

**Parameters:**
- `level`: Shield level
- `system_prompt`: System prompt
- `input_field`: Request field name for input
- `output_field`: Response field name for output

---

### PromptShieldCallback (LangChain)

```python
from promptshield.integrations import PromptShieldCallback
from langchain.chains import LLMChain

callback = PromptShieldCallback(
    system_prompt: str = "Default prompt",
    shield_level: int = 5,
    block_on_violation: bool = True
)

chain = LLMChain(
    llm=your_llm,
    callbacks=[callback]
)
```

**Parameters:**
- `system_prompt`: System prompt for checks
- `shield_level`: Shield level (1-7)
- `block_on_violation`: Raise error if blocked

---

## Result Objects

### InputCheckResult

```python
class InputCheckResult:
    blocked: bool           # Attack detected
    reason: str | None      # "pattern_match" | "high_complexity" | None
   threat_level: float     # 0.0-1.0
    secured_context: str    # System prompt with canary
    metadata: dict          # For output check
```

### OutputCheckResult

```python
class OutputCheckResult:
    blocked: bool               # Leak/PII detected
    reason: str | None          # "canary_leak" | "pii_detected" | None
    safe_response: str          # Cleaned output
    pii_found: list[str]        # ["email", "api_key", ...]
```

---

## Constants

### Block Reasons

```python
# Input block reasons
REASON_PATTERN_MATCH = "pattern_match"      # Known attack pattern
REASON_HIGH_COMPLEXITY = "high_complexity"  # Suspicious complexity
REASON_RATE_LIMIT = "rate_limit"           # Too many requests

# Output block reasons
REASON_CANARY_LEAK = "canary_leak"         # System prompt leaked
REASON_PII_DETECTED = "pii_detected"       # Sensitive data found
```

### PII Types

```python
PII_EMAIL = "email"
PII_API_KEY = "api_key"
PII_PHONE = "phone"
PII_CREDIT_CARD = "credit_card"
PII_SSN = "ssn"
PII_URL = "url"
```

---

## Configuration

### Environment Variables

```bash
# Shield configuration
SHIELD_LEVEL=5
SHIELD_THRESHOLD=0.7
SHIELD_CANARY=true
SHIELD_PII=true

# Pattern database
SHIELD_PATTERNS_DIR=promptshield/attack_db

# Logging
SHIELD_LOG_LEVEL=INFO
SHIELD_LOG_FILE=promptshield.log
```

---

## Examples

### Complete Request Flow

```python
from promptshield import Shield
from promptshield.methods import load_attack_patterns

# Initialize once
load_attack_patterns('promptshield/attack_db')
shield = Shield(level=5)

def handle_chat(user_message: str, user_id: str = None):
    """Complete protected chat flow."""
    
    # 1. INPUT PROTECTION
    input_check = shield.protect_input(
        user_input=user_message,
        system_context="You are a helpful AI assistant.",
        session_id=user_id
    )
    
    if input_check.blocked:
        return {
            "success": False,
            "error": f"Input blocked: {input_check.reason}",
            "threat_level": input_check.threat_level
        }
    
    # 2. CALL LLM
    try:
        llm_response = your_llm.generate(
            prompt=user_message,
            system=input_check.secured_context
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"LLM error: {str(e)}"
        }
    
    # 3. OUTPUT PROTECTION
    output_check = shield.protect_output(
        response=llm_response,
        metadata=input_check.metadata
    )
    
    if output_check.blocked:
        return {
            "success": False,
            "error": f"Output blocked: {output_check.reason}",
            "pii_found": output_check.pii_found
        }
    
    # 4. RETURN SAFE RESPONSE
    return {
        "success": True,
        "response": output_check.safe_response,
        "metadata": {
            "threat_level": input_check.threat_level,
            "pii_scanned": True
        }
    }
```

---

## Error Handling

### Common Errors

```python
# Missing patterns
# Error: Patterns not loaded
# Solution: Call load_attack_patterns() first

# Invalid level
# Error: ValueError: level must be 1-7
# Solution: Use valid shield level

# Missing metadata
# Error: KeyError: 'canary'
# Solution: Pass metadata from input_check to output check
```

---

## Performance

### Benchmark Results

| Operation | Latency | Throughput |
|-----------|---------|------------|
| `protect_input()` (L3) | 0.02ms | 60K req/s |
| `protect_input()` (L5) | 0.02ms | 60K req/s |
| `protect_output()` (L5) | 0.01ms | 60K req/s |
| Total (input+output) | 0.03ms | 60K req/s |

---

## Version History

### v2.0
- Framework-agnostic design
- Universal Shield interface
- L1-L7 security levels
- Comprehensive documentation

### v1.0
- Initial release
- L5 shield only
- Basic integrations

---

**Questions?** See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for usage examples.
