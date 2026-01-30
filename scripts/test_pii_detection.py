"""
Test PII Detection

Demonstrates context-aware PII detection and redaction.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptshield.pii import ContextualPIIDetector, PIIContext, RedactionMode, redact_pii, smart_redact


def main():
    """Test PII detection"""
    print("=" * 60)
    print("PromptShield - Context-Aware PII Detection")
    print("=" * 60)
    print()
    
    detector = ContextualPIIDetector()
    
    # Test Case 1: User's own PII (should ALLOW)
    print("Test 1: User's Own PII")
    print("-" * 40)
    
    user_input = "My email is alice@company.com and my phone is 555-123-4567"
    llm_output = "I'll send the confirmation to alice@company.com"
    
    # Extract user PII
    from promptshield.pii.contextual_detector import extract_user_pii
    user_pii = extract_user_pii(user_input)
    
    context = PIIContext(
        user_id="alice",
        user_provided_pii=user_pii
    )
    
    result = detector.scan_and_classify(llm_output, context)
    
    print(f"User Input: {user_input}")
    print(f"LLM Output: {llm_output}")
    print(f"Action: {result['action'].upper()}")
    print(f"Summary: {result['summary']}")
    print()
    
    # Test Case 2: Third-party PII (should WARN + REDACT)
    print("Test 2: Third-Party PII Leak")
    print("-" * 40)
    
    llm_output2 = "Contact our support team at support@company.com or call 555-999-8888"
    result2 = detector.scan_and_classify(llm_output2, context)
    
    print(f"LLM Output: {llm_output2}")
    print(f"Action: {result2['action'].upper()}")
    print(f"Summary: {result2['summary']}")
    
    # Show redacted version
    redacted = smart_redact(llm_output2, result2['findings'])
    print(f"Redacted: {redacted}")
    print()
    
    # Test Case 3: API Key Leak (should BLOCK)
    print("Test 3: Critical Credential Leak")
    print("-" * 40)
    
    llm_output3 = "Your API key is sk-proj-1234567890abcdefghijklmnop"
    result3 = detector.scan_and_classify(llm_output3, context)
    
    print(f"LLM Output: {llm_output3}")
    print(f"Action: {result3['action'].upper()} ⚠️")
    print(f"Summary: {result3['summary']}")
    
    # Show different redaction modes
    print()
    print("Redaction Modes:")
    for mode in [RedactionMode.MASK, RedactionMode.PARTIAL, RedactionMode.HASH]:
        redacted = redact_pii(llm_output3, result3['findings'], mode)
        print(f"  {mode.value}: {redacted}")
    print()
    
    # Test Case 4: Multiple PII types
    print("Test 4: Multiple PII Types")
    print("-" * 40)
    
    llm_output4 = """
    Customer Information:
    - Email: john.doe@example.com
    - Phone: +1 (555) 867-5309
    - Card: 4532 1234 5678 9010
    - SSN: 123-45-6789
    """
    
    result4 = detector.scan_and_classify(llm_output4, context)
    
    print(f"Found {len(result4['findings'])} PII instances:")
    for finding in result4['findings']:
        print(f"  - {finding.pii_type.value}: {finding.value} [{finding.severity.value}]")
    
    print()
    print(f"Action: {result4['action'].upper()}")
    print()
    print("Smart Redacted Output:")
    print(smart_redact(llm_output4, result4['findings']))
    print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("  ✅ User's own PII: ALLOWED")
    print("  ⚠️  Third-party PII: REDACTED")
    print("  🔴 System credentials: BLOCKED")
    print("=" * 60)


if __name__ == "__main__":
    main()
