"""
OpenClay - The Secure Agentic Framework

Every boundary is a trust decision.
OpenClay makes those decisions explicit.
"""

__version__ = "0.3.0"

# ── Core Shield API ──────────────────────────────────────────────────
from .shields import Shield, AsyncShield

# ── Runtime (v0.2.0) ────────────────────────────────────────────────
from .runtime import ClayRuntime, ClayResult, SecureRuntime, WrappedAgent

# ── Tools (v0.2.0) ──────────────────────────────────────────────────
from .tools import ClayTool, ToolOutputBlocked, tool

# ── Tracing ──────────────────────────────────────────────────────────
from .tracing import Trace

# ── Policies ─────────────────────────────────────────────────────────
from .policies import Policy, StrictPolicy, ModeratePolicy, CustomPolicy

# ── Knights (v0.3.0) ────────────────────────────────────────────────
from .knights import Knight, Squad

# ── Memory (v0.3.0) ─────────────────────────────────────────────────
from .memory import ClayMemory, MemoryWriteBlocked, MemoryReadBlocked

__all__ = [
    # Shield
    "Shield",
    "AsyncShield",

    # Runtime (v0.2.0)
    "ClayRuntime",
    "ClayResult",
    "SecureRuntime",
    "WrappedAgent",

    # Tools (v0.2.0)
    "ClayTool",
    "ToolOutputBlocked",
    "tool",

    # Tracing
    "Trace",

    # Policies
    "Policy",
    "StrictPolicy",
    "ModeratePolicy",
    "CustomPolicy",

    # Knights (v0.3.0)
    "Knight",
    "Squad",

    # Memory (v0.3.0)
    "ClayMemory",
    "MemoryWriteBlocked",
    "MemoryReadBlocked",
]
