# PromptShield Examples

Simple examples showing how to integrate PromptShield with various LLM providers and frameworks.

## Basic Usage

```bash
python basic_usage.py
```

Shows the simplest integration - protect input, call LLM, protect output.

## OpenAI Integration

```bash
python openai_example.py
```

Protect OpenAI GPT models. Requires `OPENAI_API_KEY` environment variable.

## LangChain Integration

```bash
python langchain_example.py
```

Integrate with LangChain chains. Requires `OPENAI_API_KEY`.

## Anthropic Claude

```bash
python claude_example.py
```

Protect Claude models. Requires `ANTHROPIC_API_KEY`.

---

## Creating Your Own Integration

```python
from promptshield import Shield

shield = Shield(level=5)

# 1. Protect input
check = shield.protect_input(user_input, "system context")
if check["blocked"]:
    return "Blocked"

# 2. Call your LLM
response = your_llm(check["secured_context"])

# 3. Protect output
output = shield.protect_output(response, check["metadata"])
return output["safe_response"]
```

That's the pattern for ANY LLM provider!
