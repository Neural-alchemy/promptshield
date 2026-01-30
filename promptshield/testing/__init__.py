"""
PromptShield Testing Module

Automated adversarial testing and evasion detection.
"""

from .evasion_tester import PromptShieldEvasionTester, run_evasion_tests

__all__ = [
    "PromptShieldEvasionTester",
    "run_evasion_tests",
]
