"""
Configurable Shield Architecture

PyTorch-style composable security shields.
Replace fixed levels (L1/L3/L5/L7) with flexible component composition.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import time


@dataclass
class ShieldResult:
    """Standardized result from shield checks"""
    blocked: bool
    reason: Optional[str] = None
    threat_level: float = 0.0
    component: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ShieldComponent:
    """
    Base class for shield components.
    
    All components must implement check() method.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.metrics = {
            "total_checks": 0,
            "total_blocks": 0,
            "total_time_ms": 0.0
        }
    
    def check(self, text: str, **context) -> ShieldResult:
        """
        Check text for threats.
        
        Args:
            text: Text to check
            **context: Additional context (user_id, session_id, etc.)
        
        Returns:
            ShieldResult
        """
        raise NotImplementedError("Subclasses must implement check()")
    
    def _track_metrics(self, blocked: bool, time_ms: float):
        """Track component metrics"""
        self.metrics["total_checks"] += 1
        if blocked:
            self.metrics["total_blocks"] += 1
        self.metrics["total_time_ms"] += time_ms
    
    def get_stats(self) -> Dict:
        """Get component statistics"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self.metrics,
            "block_rate": (
                self.metrics["total_blocks"] / max(1, self.metrics["total_checks"])
            ),
            "avg_time_ms": (
                self.metrics["total_time_ms"] / max(1, self.metrics["total_checks"])
            )
        }


# Component registry for extensibility
_COMPONENT_REGISTRY = {}


def register_component(name: str):
    """
    Decorator to register custom components.
    
    Usage:
        @register_component("my_detector")
        class MyDetector(ShieldComponent):
            def check(self, text, **context):
                return ShieldResult(blocked=False)
    """
    def decorator(cls):
        _COMPONENT_REGISTRY[name] = cls
        return cls
    return decorator


def get_component(name: str, **kwargs) -> ShieldComponent:
    """Get component by name from registry"""
    if name not in _COMPONENT_REGISTRY:
        raise ValueError(f"Component '{name}' not registered. Available: {list(_COMPONENT_REGISTRY.keys())}")
    return _COMPONENT_REGISTRY[name](**kwargs)


class Shield:
    """
    Configurable security shield with PyTorch-style API.
    
    Compose security components declaratively instead of fixed levels.
    
    Examples:
        # Full customization
        shield = Shield(
            patterns=True,
            models=["xgboost"],
            canary=True,
            rate_limiting=True,
            pii_detection=True
        )
        
        # Use presets
        shield = Shield.fast()        # <1ms
        shield = Shield.balanced()    # ~1ms
        shield = Shield.secure()      # ~5ms
        
        # Customize preset
        shield = Shield.balanced(pii_detection=True)
    """
    
    def __init__(
        self,
        # Pattern matching
        patterns: bool = True,
        pattern_db: Optional[str] = None,
        
        # ML models
        models: Optional[List[str]] = None,
        model_threshold: float = 0.7,
        
        # Security features
        canary: bool = False,
        canary_mode: str = "crypto",  # "simple" | "crypto"
        
        rate_limiting: bool = False,
        rate_limit_base: int = 100,
        
        session_tracking: bool = False,
        session_history: int = 10,
        
        pii_detection: bool = False,
        pii_redaction: str = "smart",  # "smart" | "mask" | "partial"
        
        # Model security
        verify_models: bool = False,
        
        # Performance
        cache_predictions: bool = True,
        async_mode: bool = False,
        
        # Custom components
        custom_components: Optional[List[str]] = None,
        
        **kwargs
    ):
        """
        Initialize configurable shield.
        
        Args:
            patterns: Enable pattern matching
            pattern_db: Custom pattern database path
            models: List of ML models to use
            model_threshold: Confidence threshold for ML
            canary: Enable canary tokens
            canary_mode: "simple" or "crypto"
            rate_limiting: Enable adaptive rate limiting
            rate_limit_base: Base rate limit (req/min)
            session_tracking: Enable session anomaly detection
            session_history: Messages to track per session
            pii_detection: Enable PII detection
            pii_redaction: Redaction mode
            verify_models: Verify model signatures
            cache_predictions: Cache ML predictions
            async_mode: Enable async operations
            custom_components: List of custom component names
        """
        self.config = {
            "patterns": patterns,
            "pattern_db": pattern_db or "promptshield/attack_db",
            "models": models or [],
            "model_threshold": model_threshold,
            "canary": canary,
            "canary_mode": canary_mode,
            "rate_limiting": rate_limiting,
            "rate_limit_base": rate_limit_base,
            "session_tracking": session_tracking,
            "session_history": session_history,
            "pii_detection": pii_detection,
            "pii_redaction": pii_redaction,
            "verify_models": verify_models,
            "cache_predictions": cache_predictions,
            "async_mode": async_mode,
            **kwargs
        }
        
        # Build component pipeline
        self.components = []
        self._build_pipeline(custom_components or [])
        
        # Initialize subsystems
        self._init_subsystems()
    
    def _build_pipeline(self, custom_components: List[str]):
        """Build component pipeline based on config"""
        # Import components (lazy loading)
        from .pattern_manager import PatternManager
        
        # 1. Rate limiting (first layer of defense)
        if self.config["rate_limiting"]:
            from .rate_limiting import AdaptiveRateLimiter
            self.rate_limiter = AdaptiveRateLimiter(
                base_limit=self.config["rate_limit_base"]
            )
        
        # 2. Pattern matching (fast check)
        if self.config["patterns"]:
            self.pattern_manager = PatternManager(
                self.config["pattern_db"]
            )
        
        # 3. Session anomaly detection
        if self.config["session_tracking"]:
            from .session_anomaly import SessionAnomalyDetector
            self.session_detector = SessionAnomalyDetector(
                history_window=self.config["session_history"]
            )
        
        # 4. ML models (if specified)
        if self.config["models"]:
            self._load_ml_models()
        
        # 5. Canary generation
        if self.config["canary"]:
            if self.config["canary_mode"] == "crypto":
                from .security.canary_crypto import CryptoCanaryGenerator
                self.canary_generator = CryptoCanaryGenerator()
            else:
                from .methods import generate_canary
                self.canary_generator = None  # Use simple method
        
        # 6. PII detection
        if self.config["pii_detection"]:
            from .pii import ContextualPIIDetector
            self.pii_detector = ContextualPIIDetector()
        
        # 7. Custom components
        for comp_name in custom_components:
            comp = get_component(comp_name)
            self.components.append(comp)
    
    def _init_subsystems(self):
        """Initialize subsystems (caching, async, etc.)"""
        self.prediction_cache = {} if self.config["cache_predictions"] else None
    
    def protect_input(
        self,
        user_input: str,
        system_context: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **context
    ) -> Dict:
        """
        Protect user input.
        
        Args:
            user_input: User's input text
            system_context: System prompt/context
            user_id: User identifier (for rate limiting)
            session_id: Session identifier (for tracking)
            **context: Additional context
        
        Returns:
            Dictionary with protection result
        """
        start_time = time.time()
        
        # 1. Rate limiting check
        if self.config["rate_limiting"] and user_id:
            rate_result = self.rate_limiter.check_limit(user_id, threat_level=0.0)
            if not rate_result["allowed"]:
                return {
                    "blocked": True,
                    "reason": "rate_limit_exceeded",
                    "retry_after": rate_result["retry_after"],
                    "metadata": {"component": "rate_limiter"}
                }
        
        # 2. Pattern matching
        threat_level = 0.0
        if self.config["patterns"]:
            matched, score, rule = self.pattern_manager.match(user_input)
            threat_level = max(threat_level, score)
            
            if matched:
                return {
                    "blocked": True,
                    "reason": "pattern_match",
                    "rule": rule,
                    "threat_level": score,
                    "metadata": {"component": "pattern_matcher"}
                }
        
        # 3. ML model prediction
        if self.config["models"]:
            ml_threat = self._check_ml_models(user_input)
            threat_level = max(threat_level, ml_threat)
            
            if ml_threat >= self.config["model_threshold"]:
                return {
                    "blocked": True,
                    "reason": "ml_detection",
                    "threat_level": ml_threat,
                    "metadata": {"component": "ml_model"}
                }
        
        # 4. Session anomaly detection
        if self.config["session_tracking"] and session_id:
            shield_result = {"threat_level": threat_level, "blocked": False}
            session_result = self.session_detector.analyze(
                session_id, user_input, shield_result
            )
            
            if session_result["action"] == "block_session":
                return {
                    "blocked": True,
                    "reason": session_result["reason"],
                    "session_threat": session_result["session_threat"],
                    "metadata": {"component": "session_anomaly"}
                }
        
        # 5. Generate canary (if enabled)
        canary_data = None
        secured_context = system_context
        
        if self.config["canary"]:
            session_id = session_id or f"default_{int(time.time())}"
            
            if self.config["canary_mode"] == "crypto":
                canary_data = self.canary_generator.generate(system_context, session_id)
                secured_context = self.canary_generator.inject_into_prompt(
                    system_context, canary_data
                )
            else:
                from .methods import generate_canary, inject_canary
                canary_data = {"canary": generate_canary()}
                secured_context = inject_canary(system_context, canary_data["canary"])
        
        # Update rate limiter with final threat
        if self.config["rate_limiting"] and user_id:
            self.rate_limiter.check_limit(user_id, threat_level=threat_level)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "blocked": False,
            "secured_context": secured_context,
            "canary": canary_data,
            "threat_level": threat_level,
            "latency_ms": latency_ms,
            "metadata": {
                "components_executed": self._get_active_components()
            }
        }
    
    def protect_output(
        self,
        model_output: str,
        canary: Optional[Dict] = None,
        user_id: Optional[str] = None,
        user_input: Optional[str] = None,
        **context
    ) -> Dict:
        """
        Protect model output.
        
        Args:
            model_output: LLM output text
            canary: Canary data from protect_input
            user_id: User identifier
            user_input: Original user input (for PII context)
            **context: Additional context
        
        Returns:
            Dictionary with protection result
        """
        start_time = time.time()
        
        # 1. Canary leak detection
        if self.config["canary"] and canary:
            if self.config["canary_mode"] == "crypto":
                from .security.canary_crypto import verify_canary_leak
                is_leaked, reason = verify_canary_leak(model_output, canary)
                
                if is_leaked:
                    return {
                        "blocked": True,
                        "reason": f"canary_leak:{reason}",
                        "metadata": {"component": "canary_detector"}
                    }
            else:
                from .methods import detect_canary
                if detect_canary(model_output, canary.get("canary", "")):
                    return {
                        "blocked": True,
                        "reason": "canary_leak",
                        "metadata": {"component": "canary_detector"}
                    }
        
        # 2. PII detection
        if self.config["pii_detection"]:
            from .pii import PIIContext, smart_redact
            from .pii.contextual_detector import extract_user_pii
            
            # Build PII context
            user_pii = extract_user_pii(user_input) if user_input else []
            pii_context = PIIContext(
                user_id=user_id or "unknown",
                user_provided_pii=user_pii
            )
            
            result = self.pii_detector.scan_and_classify(model_output, pii_context)
            
            if result["action"] == "block":
                return {
                    "blocked": True,
                    "reason": "pii_leak_critical",
                    "summary": result["summary"],
                    "metadata": {"component": "pii_detector"}
                }
            
            elif result["action"] == "warn":
                # Redact and allow
                redacted_output = smart_redact(model_output, result["findings"])
                
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "blocked": False,
                    "output": redacted_output,
                    "redacted": True,
                    "pii_summary": result["summary"],
                    "latency_ms": latency_ms,
                    "metadata": {"component": "pii_detector"}
                }
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "blocked": False,
            "output": model_output,
            "latency_ms": latency_ms
        }
    
    def _check_ml_models(self, text: str) -> float:
        """Check text against ML models"""
        # TODO: Implement actual ML model loading and prediction
        # For now, return 0 (not implemented yet)
        return 0.0
    
    def _get_active_components(self) -> List[str]:
        """Get list of active components"""
        active = []
        if self.config["rate_limiting"]:
            active.append("rate_limiter")
        if self.config["patterns"]:
            active.append("pattern_matcher")
        if self.config["models"]:
            active.append("ml_models")
        if self.config["session_tracking"]:
            active.append("session_anomaly")
        if self.config["canary"]:
            active.append("canary")
        if self.config["pii_detection"]:
            active.append("pii_detector")
        return active
    
    def get_stats(self) -> Dict:
        """Get shield statistics"""
        stats = {
            "config": self.config,
            "active_components": self._get_active_components()
        }
        
        # Add component-specific stats
        if self.config["rate_limiting"]:
            stats["rate_limiter"] = self.rate_limiter.get_global_stats()
        
        if self.config ["patterns"]:
            stats["pattern_manager"] = self.pattern_manager.get_stats()
        
        if self.config["session_tracking"]:
            stats["session_detector"] = self.session_detector.get_global_stats()
        
        return stats
    
    # ============================================
    # Preset Factories (Convenience)
    # ============================================
    
    @classmethod
    def fast(cls, **overrides):
        """
        Fast preset: Pattern-only, <0.5ms
        
        Best for: High-throughput APIs, agent-to-agent
        """
        return cls(
            patterns=True,
            models=None,
            canary=False,
            rate_limiting=False,
            session_tracking=False,
            pii_detection=False,
            **overrides
        )
    
    @classmethod
    def balanced(cls, **overrides):
        """
        Balanced preset: Patterns + canary, ~1ms
        
        Best for: Most production use cases
        """
        return cls(
            patterns=True,
            models=None,  # Can add XGBoost later
            canary=True,
            canary_mode="crypto",
            rate_limiting=False,
            session_tracking=False,
            pii_detection=False,
            **overrides
        )
    
    @classmethod
    def secure(cls, **overrides):
        """
        Secure preset: Full protection, ~5ms
        
        Best for: High-value data, compliance requirements
        """
        return cls(
            patterns=True,
            models=None,  # Can add ML models
            canary=True,
            canary_mode="crypto",
            rate_limiting=True,
            session_tracking=True,
            pii_detection=True,
            pii_redaction="smart",
            **overrides
        )
    
    @classmethod
    def paranoid(cls, **overrides):
        """
        Paranoid preset: Everything enabled, ~10ms
        
        Best for: Maximum security, admin endpoints
        """
        return cls(
            patterns=True,
            models=["xgboost"],  # When available
            canary=True,
            canary_mode="crypto",
            rate_limiting=True,
            rate_limit_base=50,  # Stricter
            session_tracking=True,
            session_history=15,
            pii_detection=True,
            pii_redaction="mask",  # Most aggressive
            verify_models=True,
            **overrides
        )


# ============================================
# Backward Compatibility (Deprecated)
# ============================================

class InputShield_L5(Shield):
    """
    DEPRECATED: Use Shield.balanced() instead.
    
    Kept for backward compatibility.
    """
    def __init__(self):
        import warnings
        warnings.warn(
            "InputShield_L5 is deprecated. Use Shield.balanced() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(
            patterns=True,
            canary=True,
            canary_mode="simple"  # Old behavior
        )
    
    def run(self, user_input: str, system_prompt: str):
        """Legacy API compatibility"""
        return self.protect_input(user_input, system_prompt)


class OutputShield_L5(Shield):
    """
    DEPRECATED: Use Shield.balanced() instead.
    
    Kept for backward compatibility.
    """
    def __init__(self):
        import warnings
        warnings.warn(
            "OutputShield_L5 is deprecated. Use Shield.balanced() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(
            patterns=False,
            canary=True,
            pii_detection=True
        )
    
    def run(self, model_output: str, canary: str):
        """Legacy API compatibility"""
        canary_data = {"canary": canary}
        return self.protect_output(model_output, canary=canary_data)


class AgentShield_L3(Shield):
    """
    DEPRECATED: Use Shield.fast() instead.
    
    Kept for backward compatibility.
    """
    def __init__(self):
        import warnings
        warnings.warn(
            "AgentShield_L3 is deprecated. Use Shield.fast() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(
            patterns=True,
            canary=False,
            pii_detection=False
        )
    
    def run(self, message: str):
        """Legacy API compatibility"""
        result = self.protect_input(message, "")
        return {
            "block": result["blocked"],
            "reason": result.get("reason")
        }
