# Contributing to Voice Personality Analysis System

Thank you for your interest in contributing to our Voice Personality Analysis System! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **🐛 Bug Reports**: Report issues you encounter
- **✨ Feature Requests**: Suggest new features or improvements
- **📝 Documentation**: Improve or add documentation
- **🔧 Code Contributions**: Submit code improvements or new features
- **🧪 Testing**: Help test the system and report issues
- **🌍 Translations**: Help translate the system to other languages

### Before You Start

1. **Check Existing Issues**: Search existing issues to avoid duplicates
2. **Read Documentation**: Familiarize yourself with the project structure
3. **Set Up Development Environment**: Follow the installation guide in README.md

## 🚀 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- FFmpeg (for audio processing)
- Google Gemini API Key (for testing)

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/voice-personality-ai.git
cd voice-personality-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_topic_org.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

## 📝 Code Style Guidelines

### Python Code Style

We follow PEP 8 style guidelines:

```bash
# Format code with black
black .

# Check code style with flake8
flake8 .

# Type checking with mypy
mypy .
```

### Code Quality Standards

- **Documentation**: Add docstrings to all functions and classes
- **Type Hints**: Use type hints for function parameters and return values
- **Error Handling**: Implement proper error handling and logging
- **Testing**: Write tests for new features

### Example Code Style

```python
def analyze_voice_characteristics(audio_file: str) -> Dict[str, float]:
    """
    Analyze voice characteristics from an audio file.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Dictionary containing voice characteristics
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file format is not supported
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Implementation here
    return characteristics
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_voice_analysis.py

# Run tests with verbose output
pytest -v
```

### Writing Tests

Create tests in the `tests/` directory:

```python
# tests/test_voice_analysis.py
import pytest
from multimodal_analysis import analyze_voice_characteristics

def test_analyze_voice_characteristics():
    """Test voice characteristics analysis."""
    # Test implementation
    result = analyze_voice_characteristics("test_audio.wav")
    assert "pitch" in result
    assert "tempo" in result
    assert isinstance(result["pitch"], float)
```

## 📋 Pull Request Process

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/voice-personality-ai.git
cd voice-personality-ai
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Write your code following the style guidelines
- Add tests for new functionality
- Update documentation if needed
- Ensure all tests pass

### 4. Commit Changes

```bash
git add .
git commit -m "Add feature: brief description of changes"
```

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to related issues
- Screenshots for UI changes
- Test results

## 🐛 Reporting Issues

### Bug Report Template

When reporting bugs, please include:

```markdown
**Bug Description**
Brief description of the issue

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., Windows 10, macOS 12.0]
- Python Version: [e.g., 3.9.7]
- Package Versions: [output of `pip freeze`]

**Additional Information**
Any other relevant information
```

## ✨ Feature Request Template

```markdown
**Feature Description**
Brief description of the feature

**Use Case**
Why this feature would be useful

**Proposed Implementation**
How you think it could be implemented

**Alternatives Considered**
Other approaches you considered
```

## 📚 Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep documentation up to date

### Documentation Structure

- `README.md`: Main project documentation
- `CONTRIBUTING.md`: This file
- `docs/`: Additional documentation
- Inline code documentation

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Follow the project's coding standards

### Communication

- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for general questions
- Be patient and helpful with new contributors

## 🏆 Recognition

Contributors will be recognized in:
- Project README
- Release notes
- GitHub contributors page

## 📞 Getting Help

If you need help:
- Check existing documentation
- Search existing issues
- Ask questions in GitHub Discussions
- Contact maintainers directly

Thank you for contributing to the Voice Personality Analysis System! 🎉 