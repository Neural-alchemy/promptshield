# PromptShield Best Practices

**Production deployment guide for PromptShield.**

---

## 🚀 Quick Checklist

Before deploying to production:

- [ ] Choose appropriate shield level (L3/L5/L7)
- [ ] Set complexity threshold for your use case
- [ ] Add logging for blocked attempts
- [ ] Set up monitoring/metrics
- [ ] Test with Rapture
- [ ] Document integration points
- [ ] Plan incident response

---

## 1. Shield Level Selection

### Production Recommendation

```python
# ✅ RECOMMENDED for most production systems
from promptshield import Shield

shield = Shield(
    level=5,                      # Full protection
    complexity_threshold=0.7,     # Balanced
    enable_canary=True,           # Detect leaks
    enable_pii_scanning=True      # Protect user data
)
```

### By Environment

```python
# Development
shield_dev = Shield(level=1)  # Fast, minimal blocking

# Staging
shield_staging = Shield(level=3)  # Test with patterns

# Production
shield_prod = Shield(level=5)  # Full protection

# High-Security Production
shield_secure = Shield(level=7)  # Maximum security
```

---

## 2. Initialization

### ✅ DO: Initialize Once at Startup

```python
# app.py
from promptshield import Shield
from promptshield.methods import load_attack_patterns

# Load patterns once
load_attack_patterns('promptshield/attack_db')

# Create shield instance (reuse across requests)
shield = Shield(level=5)

def handle_request(user_input):
    check = shield.protect_input(user_input, "context")
    # ...
```

### ❌ DON'T: Create New Shield Per Request

```python
# SLOW - creates new shield every time
def handle_request(user_input):
    shield = Shield(level=5)  # ❌ Don't do this!
    check = shield.protect_input(user_input, "context")
```

---

## 3. Error Handling

### Graceful Degradation

```python
def safe_ai_call(user_input):
    try:
        # Try to protect
        check = shield.protect_input(user_input, "context")
        
        if check.blocked:
            return {"error": "Request blocked", "safe": True}
        
        # Call AI
        response = ai_system.run(user_input)
        
        # Check output
        output = shield.protect_output(response, check.metadata)
        
        if output.blocked:
            return {"error": "Output blocked", "safe": True}
        
        return {"response": output.safe_response, "safe": True}
        
    except Exception as e:
        # Shield failed - decide on fail-open or fail-closed
        logger.error(f"Shield error: {e}")
        
        # Option 1: Fail-closed (safer)
        return {"error": "Security check failed", "safe": False}
        
        # Option 2: Fail-open (more available)
        # return ai_system.run(user_input)
```

### Logging All Failures

```python
import logging

logger = logging.getLogger("promptshield")

def protected_call(user_input):
    check = shield.protect_input(user_input, "context")
    
    if check.blocked:
        logger.warning(
            "Attack blocked",
            extra={
                "reason": check.reason,
                "threat_level": check.threat_level,
                "input_preview": user_input[:100],
                "timestamp": datetime.now().isoformat()
            }
        )
        return handle_blocked()
```

---

## 4. Monitoring & Metrics

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
blocked_total = Counter(
    'promptshield_blocked_total',
    'Total blocked requests',
    ['reason']
)

latency_seconds = Histogram(
    'promptshield_latency_seconds',
    'Shield latency',
    ['operation']
)

threat_level = Histogram(
    'promptshield_threat_level',
    'Threat level distribution'
)

def protected_endpoint(user_input):
    with latency_seconds.labels('input').time():
        check = shield.protect_input(user_input, "context")
    
    if check.blocked:
        blocked_total.labels(reason=check.reason).inc()
        return {"error": "Blocked"}
    
    # Record threat level
    threat_level.observe(check.threat_level)
    
    # ... rest of code
```

### Custom Metrics Dashboard

```python
import json
from datetime import datetime

class ShieldMetrics:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "by_reason": {},
            "by_hour": {}
        }
    
    def record_check(self, check_result):
        self.metrics["total_requests"] += 1
        
        if check_result.blocked:
            self.metrics["blocked_requests"] += 1
            
            reason = check_result.reason
            self.metrics["by_reason"][reason] = \
                self.metrics["by_reason"].get(reason, 0) + 1
            
            hour = datetime.now().hour
            self.metrics["by_hour"][hour] = \
                self.metrics["by_hour"].get(hour, 0) + 1
    
    def get_stats(self):
        total = self.metrics["total_requests"]
        blocked = self.metrics["blocked_requests"]
        
        return {
            "block_rate": blocked / total if total > 0 else 0,
            "total": total,
            "blocked": blocked,
            "by_reason": self.metrics["by_reason"],
            "by_hour": self.metrics["by_hour"]
        }

# Usage
metrics = ShieldMetrics()

def protected_call(user_input):
    check = shield.protect_input(user_input, "context")
    metrics.record_check(check)
    # ...
```

---

## 5. Logging Strategy

### Structured Logging

```python
import logging
import json

logger = logging.getLogger("promptshield")

def log_shield_event(event_type, data):
    logger.info(json.dumps({
        "event": event_type,
        "timestamp": datetime.now().isoformat(),
        **data
    }))

def protected_call(user_input, user_id=None):
    check = shield.protect_input(user_input, "context")
    
    if check.blocked:
        log_shield_event("attack_blocked", {
            "user_id": user_id,
            "reason": check.reason,
            "threat_level": check.threat_level,
            "input_hash": hash(user_input),
            "patterns_matched": check.get("patterns", [])
        })
```

### ELK Stack Integration

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(['localhost:9200'])

def log_to_elk(event):
    es.index(
        index="promptshield-events",
        document={
            "timestamp": datetime.now(),
            "event_type": event["type"],
            "user_id": event.get("user_id"),
            "blocked": event.get("blocked", False),
            "reason": event.get("reason"),
            "threat_level": event.get("threat_level", 0.0)
        }
    )
```

---

## 6. Testing with Rapture

### Regular Security Testing

```bash
# Test your protected system monthly
python rapture/main.py \
  --target your_protected_system.py \
  --attacks rapture/attacks \
  --output monthly_scan.json \
  --mode static

# Deep test quarterly
python rapture/main.py \
  --target your_protected_system.py \
  --attacks rapture/attacks \
  --output quarterly_deep_scan.json \
  --mode adaptive \
  --max-iterations 5
```

### CI/CD Integration

```yaml
# .github/workflows/security-test.yml
name: Security Test

on: [push, pull_request]

jobs:
  rapture-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run Rapture scan
        run: |
          python rapture/main.py \
            --target targets/your_app.py \
            --attacks rapture/attacks \
            --output scan_results.json
      
      - name: Check results
        run: |
          python check_scan_results.py scan_results.json
          # Fail if critical vulnerabilities found
```

---

## 7. Performance Optimization

### Async for High Throughput

```python
from promptshield.async_shields import AsyncInputShield_L5
import asyncio

shield = AsyncInputShield_L5()

async def handle_batch(inputs):
    tasks = [
        shield.run(inp, "context")
        for inp in inputs
    ]
    
    results = await asyncio.gather(*tasks)
    return results

# Process 1000 requests concurrently
results = asyncio.run(handle_batch(user_inputs))
```

### Caching for Repeated Inputs

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_input_check(user_input_hash, context_hash):
    return shield.protect_input(user_input, context)

def protected_call(user_input, context):
    # Hash inputs for caching
    input_hash = hash(user_input)
    context_hash = hash(context)
    
    check = cached_input_check(input_hash, context_hash)
    # ...
```

---

## 8. Incident Response

### Attack Detection Workflow

```
Attack Detected
    ↓
1. Log event with full context
    ↓
2. Alert security team (if high severity)
    ↓
3. Temporarily block user (if repeated attempts)
    ↓
4. Review and update patterns
    ↓
5. Post-incident analysis
```

### Automated Response

```python
from datetime import datetime, timedelta

class AttackTracker:
    def __init__(self):
        self.attempts = {}  # user_id: [timestamps]
    
    def record_attempt(self, user_id):
        now = datetime.now()
        
        if user_id not in self.attempts:
            self.attempts[user_id] = []
        
        # Clean old attempts (>1 hour)
        self.attempts[user_id] = [
            t for t in self.attempts[user_id]
            if now - t < timedelta(hours=1)
        ]
        
        self.attempts[user_id].append(now)
    
    def should_block_user(self, user_id):
        # Block if >5 attempts in 1 hour
        return len(self.attempts.get(user_id, [])) > 5

tracker = AttackTracker()

def protected_call(user_input, user_id):
    # Check if user is temporarily blocked
    if tracker.should_block_user(user_id):
        return {"error": "Too many suspicious requests. Please try later."}
    
    check = shield.protect_input(user_input, "context")
    
    if check.blocked:
        tracker.record_attempt(user_id)
        alert_security_team(user_id, check)
        return {"error": "Request blocked"}
    
    # ... normal flow
```

---

## 9. Configuration Management

### Environment-Based Config

```python
import os

class ShieldConfig:
    def __init__(self):
        self.level = int(os.getenv('SHIELD_LEVEL', 5))
        self.threshold = float(os.getenv('SHIELD_THRESHOLD', 0.7))
        self.enable_canary = os.getenv('SHIELD_CANARY', 'true').lower() == 'true'
        self.enable_pii = os.getenv('SHIELD_PII', 'true').lower() == 'true'

config = ShieldConfig()
shield = Shield(
    level=config.level,
    complexity_threshold=config.threshold,
    enable_canary=config.enable_canary,
    enable_pii_scanning=config.enable_pii
)
```

### YAML Configuration

```yaml
# config/promptshield.yml
shield:
  level: 5
  complexity_threshold: 0.7
  
  features:
    canary_tokens: true
    pii_scanning: true
    behavioral_analysis: false
  
  pii_types:
    - email
    - api_key
    - phone
  
  rate_limiting:
    enabled: true
    max_requests_per_minute: 60
```

```python
import yaml

with open('config/promptshield.yml') as f:
    config = yaml.safe_load(f)

shield = Shield(
    level=config['shield']['level'],
    complexity_threshold=config['shield']['complexity_threshold'],
    enable_canary=config['shield']['features']['canary_tokens'],
    enable_pii_scanning=config['shield']['features']['pii_scanning']
)
```

---

## 10. User Communication

### User-Friendly Error Messages

```python
def get_user_message(block_reason):
    messages = {
        "pattern_match": "Your message contains content that violates our usage policy. Please rephrase.",
        "high_complexity": "Your message is too complex. Please simplify and try again.",
        "canary_leak": "System error. Please contact support.",
        "pii_detected": "Your message contains personal information. Please remove and retry.",
        "rate_limit": "Too many requests. Please wait a moment and try again."
    }
    
    return messages.get(block_reason, "Request could not be processed.")

def protected_call(user_input):
    check = shield.protect_input(user_input, "context")
    
    if check.blocked:
        return {
            "error": get_user_message(check.reason),
            "code": "INPUT_REJECTED"
        }
```

---

## 11. Documentation for Your Team

### Integration Documentation Template

```markdown
# PromptShield Integration - [Your Service Name]

## Integration Points
- **Input Protection:** Before LLM API call in `service/ai_handler.py:45`
- **Output Protection:** After LLM response in `service/ai_handler.py:67`

## Shield Configuration
- **Level:** 5 (Full protection)
- **Threshold:** 0.7 (Balanced)
- **Features:** Canary tokens, PII scanning

## Monitoring
- **Metrics:** Prometheus at `/metrics`
- **Logs:** ELK stack, index: `promptshield-prod`
- **Dashboard:** Grafana at `https://grafana.company.com/promptshield`

## Incident Response
1. Check logs in ELK
2. Review blocked requests in Grafana
3. Contact security team if sustained attack
4. Update patterns in `promptshield/attack_db/`

## Testing
```bash
# Run monthly security scan
npm run security:scan
```
```

---

## 12. Maintenance

### Regular Pattern Updates

```bash
# Monthly: Sync with latest Rapture attack database
cd rapture
git pull
cp attacks/* ../promptshield/attack_db/

# Restart services to reload patterns
kubectl rollout restart deployment/your-app
```

### Monitoring False Positives

```python
def review_blocked_requests():
    """Review and categorize blocked requests."""
    
    logs = get_blocked_requests_last_week()
    
    false_positives = []
    true_positives = []
    
    for log in logs:
        print(f"Input: {log['input']}")
        print(f"Reason: {log['reason']}")
        
        response = input("False positive? (y/n): ")
        
        if response.lower() == 'y':
            false_positives.append(log)
        else:
            true_positives.append(log)
    
    # Adjust threshold if too many false positives
    fp_rate = len(false_positives) / len(logs)
    
    if fp_rate > 0.05:  # >5% false positives
        print(f"⚠️ High false positive rate: {fp_rate:.1%}")
        print("Consider increasing complexity_threshold")
```

---

## Summary: Production Checklist

### Before Launch:
- [x] Shield level selected and documented
- [x] Error handling implemented
- [x] Logging configured
- [x] Metrics/monitoring set up
- [x] Tested with Rapture
- [x] Incident response plan documented
- [x] Team trained on integration

### After Launch:
- [ ] Monitor block rate daily (first week)
- [ ] Review false positives weekly
- [ ] Security scan monthly (Rapture)
- [ ] Update patterns quarterly
- [ ] Team review quarterly

---

**Questions?** See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for code examples.
