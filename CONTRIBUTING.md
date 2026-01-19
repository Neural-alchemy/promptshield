# Contributing to PromptShield

Thank you for your interest in contributing to PromptShield! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs
- Use GitHub Issues
- Include: OS, Python version, PromptShield version
- Provide minimal reproducible example
- Describe expected vs actual behavior

### Suggesting Features
- Open GitHub Issue with "Feature Request" label
- Describe use case and benefits
- Provide examples if possible

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make changes**
   - Follow code style (PEP 8)
   - Add tests
   - Update documentation

4. **Run tests**
   ```bash
   python -m pytest tests/
   ```

5. **Submit Pull Request**
   - Clear description
   - Reference related issues
   - Include tests

## Development Setup

```bash
git clone https://github.com/neuralalchemy/promptshield
cd promptshield
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and small

## Testing

- Write tests for new features
- Maintain >90% code coverage
- Test edge cases

## Documentation

- Update docs for new features
- Add code examples
- Keep README concise

## Community Guidelines

- Be respectful and constructive
- Help others in issues/discussions
- Follow Code of Conduct

---

**Questions?** Open a GitHub Discussion or join our Discord.
