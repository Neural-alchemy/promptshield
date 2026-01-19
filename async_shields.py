# promptshield/async_shields.py
"""
Async versions of PromptShield for high-throughput applications
"""

import asyncio
from typing import Dict
from .methods import (
    sanitize_text,
    pattern_match,
    complexity_score,
    generate_canary,
    inject_canary,
    detect_canary,
    pii_scan
)


class AsyncInputShield_L5:
    """Async version of InputShield_L5 for concurrent request handling"""
    
    def __init__(self, complexity_threshold=0.7):
        self.complexity_threshold = complexity_threshold
    
    async def run(self, user_input: str, system_prompt: str) -> Dict:
        """
        Async shield check - can handle many requests concurrently
        
        Returns:
            {
                "block": bool,
                "reason": str or None,
                "secured_system_prompt": str,
                "canary": str
            }
        """
        # All operations are CPU-bound and fast, so we can run them directly
        # For truly async I/O operations, we'd use asyncio.to_thread()
        
        # Layer 1: Sanitization
        cleaned = sanitize_text(user_input)
        
        # Layer 2: Pattern matching
        matched, score, rule = pattern_match(cleaned)
        if matched:
            return {
                "block": True,
                "reason": f"pattern_match (rule: {rule}, score: {score})",
                "secured_system_prompt": system_prompt,
                "canary": None
            }
        
        # Layer 3: Complexity analysis
        complexity = complexity_score(cleaned)
        if complexity > self.complexity_threshold:
            return {
                "block": True,
                "reason": f"high_complexity (score: {complexity:.2f})",
                "secured_system_prompt": system_prompt,
                "canary": None
            }
        
        # Layer 4: Inject canary token
        canary = generate_canary()
        secured_prompt = inject_canary(system_prompt, canary)
        
        return {
            "block": False,
            "reason": None,
            "secured_system_prompt": secured_prompt,
            "canary": canary
        }


class AsyncOutputShield_L5:
    """Async version of OutputShield_L5"""
    
    async def run(self, output: str, canary: str) -> Dict:
        """
        Async output validation
        
        Returns:
            {
                "block": bool,
                "reason": str or None,
                "output": str
            }
        """
        # Layer 1: Canary detection
        if canary and detect_canary(output, canary):
            return {
                "block": True,
                "reason": "canary_leak",
                "output": ""
            }
        
        # Layer 2: PII scanning
        findings = pii_scan(output)
        if findings:
            return {
                "block": True,
                "reason": f"pii_detected ({', '.join(findings.keys())})",
                "output": ""
            }
        
        return {
            "block": False,
            "reason": None,
            "output": output
        }


# Convenience function for batch processing
async def batch_shield_check(inputs: list, system_prompt: str) -> list:
    """
    Process multiple inputs concurrently
    
    Args:
        inputs: List of user input strings
        system_prompt: System prompt to protect
        
    Returns:
        List of shield results in same order as inputs
    """
    shield = AsyncInputShield_L5()
    
    tasks = [shield.run(inp, system_prompt) for inp in inputs]
    results = await asyncio.gather(*tasks)
    
    return results
