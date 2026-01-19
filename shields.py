# shields.py

from .methods import (
    sanitize_text,
    pattern_match,
    semantic_match,
    complexity_score,
    generate_canary,
    inject_canary,
    detect_canary,
    pii_scan,
)

# -----------------------------
# INPUT SHIELD — LEVEL 5 (MAX)
# -----------------------------

class InputShield_L5:
    """
    Full-strength input shield for user input.
    """

    def __init__(self, complexity_threshold=0.8):
        self.complexity_threshold = complexity_threshold

    def run(self, user_input: str, system_prompt: str):
        # Sanitize input
        text = sanitize_text(user_input)

        # 1. Pattern matching (Rapture DB)
        matched, score, rule = pattern_match(text)
        if matched:
            return {
                "block": True,
                "reason": "pattern_match",
                "rule": rule,
            }

        # 2. Semantic embedding match
        sem_match, similarity = semantic_match(text)
        if sem_match:
            return {
                "block": True,
                "reason": "semantic_match",
                "similarity": similarity,
            }

        # 3. Complexity analysis
        comp_score = complexity_score(text)
        if comp_score >= self.complexity_threshold:
            return {
                "block": True,
                "reason": "complexity",
                "score": comp_score,
            }

        # 4. Canary injection
        canary = generate_canary()
        secured_prompt = inject_canary(system_prompt, canary)

        return {
            "block": False,
            "secured_system_prompt": secured_prompt,
            "canary": canary,  # store per session
        }


# -----------------------------
# AGENT SHIELD — LEVEL 3
# -----------------------------

class AgentShield_L3:
    """
    Lightweight shield for agent-to-agent messages.
    """

    def __init__(self, complexity_threshold=0.7):
        self.complexity_threshold = complexity_threshold

    def run(self, agent_message: str):
        text = sanitize_text(agent_message)

        matched, score, rule = pattern_match(text)
        if matched:
            return {
                "block": True,
                "reason": "pattern_match",
                "rule": rule,
            }

        comp_score = complexity_score(text)
        if comp_score >= self.complexity_threshold:
            return {
                "block": True,
                "reason": "complexity",
                "score": comp_score,
            }

        return {
            "block": False,
            "message": text,
        }


# -----------------------------
# OUTPUT SHIELD — LEVEL 5
# -----------------------------

class OutputShield_L5:
    """
    Full-strength output shield.
    """

    def run(self, model_output: str, canary: str):
        # 1. Canary leak detection
        if detect_canary(model_output, canary):
            return {
                "block": True,
                "reason": "canary_leak",
            }

        # 2. PII detection
        pii_findings = pii_scan(model_output)
        if pii_findings:
            return {
                "block": True,
                "reason": "pii_detected",
                "details": pii_findings,
            }

        return {
            "block": False,
            "output": model_output,
        }
