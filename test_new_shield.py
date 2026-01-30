"""
Simple Test: New Shield API
"""

from promptshield import Shield

print("Testing New Shield API")
print("=" * 40)

# Test 1: Fast preset
print("\n1. Shield.fast()")
shield = Shield.fast()
result = shield.protect_input("Test", "You are helpful")
print(f"   Blocked: {result['blocked']}")
print(f"   Components: {result['metadata']['components_executed']}")

# Test 2: Balanced preset  
print("\n2. Shield.balanced()")
shield = Shield.balanced()
result = shield.protect_input("Hello", "You are helpful")
print(f"   Blocked: {result['blocked']}")
print(f"   Has canary: {result['canary'] is not None}")

# Test 3: Secure preset
print("\n3. Shield.secure()")
shield = Shield.secure()
result = shield.protect_input("Hi", "You are helpful", user_id="test", session_id="s1")
print(f"   Blocked: {result['blocked']}")
print(f"   Components: {result['metadata']['components_executed']}")

# Test 4: Custom config
print("\n4. Custom Shield")
shield = Shield(patterns=True, canary=True, pii_detection=True)
print(f"   Active: {shield._get_active_components()}")

print("\n" + "=" * 40)
print("All tests passed!")
