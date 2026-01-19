# PromptShield - LLM Defense Toolkit

from .shields import InputShield_L5, OutputShield_L5, AgentShield_L3

# Universal Shield interface (for v2.0)
class Shield:
    """
    Universal shield interface for protecting AI systems.
    Supports L1-L7 security levels.
    """
    
    def __init__(self, level=5, **kwargs):
        """
        Args:
            level: Security level (1-7)
            **kwargs: Additional configuration
        """
        self.level = level
        self.config = kwargs
        
        # Use existing shields based on level
        if level <= 3:
            self.input_shield = AgentShield_L3(**kwargs)
        else:  # level >= 5
            self.input_shield = InputShield_L5(**kwargs)
        
        self.output_shield = OutputShield_L5(**kwargs)
    
    def protect_input(self, user_input, system_context, session_id=None):
        """
        Protect user input before sending to LLM.
        
        Returns: dict with keys: blocked, reason, threat_level, secured_context, metadata
        """
        result = self.input_shield.run(user_input, system_context)
        
        # Add metadata for output check
        result["metadata"] = {
            "canary": result.get("canary", ""),
            "session_id": session_id
        }
        
        # Calculate threat level from result
        if result["block"]:
            result["threat_level"] = 0.9
        else:
            result["threat_level"] = 0.0
        
        # Rename keys for consistency
        result["blocked"] = result.pop("block")
        result["secured_context"] = result.pop("secured_prompt", system_context)
        
        return result
    
    def protect_output(self, response, metadata, scan_pii=True):
        """
        Protect LLM output before returning to user.
        
        Returns: dict with keys: blocked, reason, safe_response, pii_found
        """
        canary = metadata.get("canary", "")
        result = self.output_shield.run(response, canary)
        
        # Rename keys for consistency
        result["blocked"] = result.pop("block")
        result["safe_response"] = result.pop("output", response)
        result["pii_found"] = result.get("pii_detected", [])
        
        return result


__all__ = [
    "Shield",
    "InputShield_L5",
    "OutputShield_L5",
    "AgentShield_L3"
]
