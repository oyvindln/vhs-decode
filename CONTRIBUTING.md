# Contributing to ld-decode

Thank you for your interest in contributing to ld-decode! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

This project is run by volunteers and we expect all contributors to be respectful and constructive. Please:

- Be welcoming and inclusive
- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

Before contributing, make sure you have:

1. A GitHub account
2. Git installed and configured
3. Development environment set up (see [BUILD.md](BUILD.md))
4. Basic understanding of C++17, Python, and CMake

### Setting Up Development Environment

See [BUILD.md](BUILD.md) for complete build instructions. Quick reference:

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/happycube/ld-decode.git
cd ld-decode

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install for development
pip install -e ".[dev]"

# Run tests
pytest --output-on-failure
```

### Using a Virtual Environment (Python Development)

For Python development, see the virtual environment section in [INSTALL.md](INSTALL.md#python-module-not-found).

Quick reference:

```bash
python3 -m venv ~/ld-decode-venv
source ~/ld-decode-venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## How to Contribute

### Reporting Bugs

Before creating a bug report:

1. Check the [existing issues](https://github.com/happycube/ld-decode/issues) to avoid duplicates
2. Gather information about the bug:
   - Operating system and version
   - ld-decode version (git commit hash)
   - Steps to reproduce
   - Expected vs. actual behavior
   - Relevant logs or error messages

Create a detailed bug report including:

```markdown
**Environment:**
- OS: Ubuntu 22.04
- ld-decode commit: abc123def
- Qt version: 6.x.x

**Steps to Reproduce:**
1. Run `ld-decode myfile.ldf`
2. Click on X
3. Observe Y

**Expected Behavior:**
Should do Z

**Actual Behavior:**
Does W instead

**Logs:**
```
[paste relevant logs]
```
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

1. Check if the feature has already been requested
2. Provide a clear and detailed explanation of the feature
3. Explain why this enhancement would be useful
4. Include examples of how the feature would be used

### Contributing Code

Areas where contributions are particularly welcome:

- **Bug fixes**: Fixing existing issues
- **Performance improvements**: Optimizing decoding algorithms
- **Documentation**: Improving docs, comments, and examples
- **Testing**: Adding tests for existing functionality
- **Platform support**: Improving Windows/macOS compatibility
- **New features**: After discussion with maintainers

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone --recurse-submodules https://github.com/YOUR_USERNAME/ld-decode.git
cd ld-decode

# Add upstream remote
git remote add upstream https://github.com/happycube/ld-decode.git

# If you already cloned without --recurse-submodules, initialize testdata:
git submodule update --init --recursive
```

### 2. Create a Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/description` - For new features
- `fix/description` - For bug fixes
- `docs/description` - For documentation changes
- `refactor/description` - For code refactoring

### 3. Make Changes

- Write clear, readable code
- Follow existing code style
- Add comments for complex logic
- Update documentation as needed
- Add tests for new functionality

### 4. Test Your Changes

```bash
# Run all tests
pytest --output-on-failure

# Run specific tests
pytest tests/test_chroma.py --output-on-failure

# Run with coverage
pytest --cov=lddecode

# Test manually with your changes
python -c "import lddecode; print('Import successful')"
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Short description of change

Longer explanation of what changed and why. Reference any
related issues.

Fixes #123
```

Commit message guidelines:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Reference issues and pull requests when relevant

### 6. Keep Your Branch Updated

```bash
# Fetch latest changes from upstream
git fetch upstream

# Rebase your branch on upstream/main
git rebase upstream/main

# If there are conflicts, resolve them and continue
git rebase --continue
```

### 7. Push to Your Fork

```bash
git push origin feature/my-new-feature
```

## Coding Standards

### C++ Code

- **Standard**: C++17
- **Style**: Follow existing code style in the project
- **Naming**:
  - Classes: `PascalCase`
  - Functions/methods: `camelCase`
  - Variables: `camelCase`
  - Constants: `UPPER_CASE`
  - Private members: `camelCase` (some older code uses `m_` prefix, but this is not required for new code)

- **Best Practices**:
  - Use `const` where appropriate
  - Prefer smart pointers over raw pointers
  - Use `nullptr` instead of `NULL`
  - Avoid `using namespace std;`
  - Include guards or `#pragma once`

Example:
```cpp
class VideoDecoder {
public:
    VideoDecoder(int width, int height);
    bool decodeFrame(const FrameData& data);
    
private:
    int width;          // Modern style
    int height;         // Modern style
    std::vector<uint8_t> buffer;
};
```

### Python Code

- **Standard**: Python 3.6+
- **Style**: Follow PEP 8
- **Formatting**: 4 spaces for indentation (no tabs)
- **Imports**: Group standard library, third-party, and local imports
- **Documentation**: Use docstrings for functions and classes

Example:
```python
def decode_video(input_file, output_file, system='pal'):
    """Decode video from TBC file.
    
    Args:
        input_file (str): Path to input TBC file
        output_file (str): Path to output file
        system (str): Video system ('pal' or 'ntsc')
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Implementation
    pass
```

### CMake

- Use lowercase for commands
- Indent with 4 spaces
- Use modern CMake practices (targets, not variables)

Example:
```cmake
add_executable(ld-analyse
    main.cpp
    mainwindow.cpp
)

target_link_libraries(ld-analyse
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        lddecode-library
)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_core.py

# Run with coverage report
pytest --cov=lddecode --cov-report=html

# Run in parallel
pytest -n auto
```

### Writing Tests

When adding new features:

1. Add pytest test files in `tests/` directory
2. Name test files `test_*.py`
3. Use descriptive test function names starting with `test_`
4. Ensure tests are deterministic
5. Tests should run quickly

Example test:
```python
import pytest
from lddecode.core import LDDecode

def test_basic_import():
    """Test that the module can be imported."""
    assert LDDecode is not None

def test_decode_frame():
    """Test basic frame decoding."""
    # Your test implementation
    pass
```
4. Use descriptive test names

Example test structure:
```bash
# In cmake_modules/LdDecodeTests.cmake
add_test(
    NAME my-new-test
    COMMAND ${CMAKE_BINARY_DIR}/tools/my-tool/my-tool --test-flag
)
```

### Test Data

- Use existing test data in `testdata/` when possible
- Keep test files small and focused
- Document any new test data requirements

## Submitting Changes

### Pull Request Process

1. **Ensure your code builds and tests pass**
   ```bash
   pip install -e ".[dev]"
   pytest
   ```

2. **Update documentation**
   - Update README.md if needed
   - Add/update comments in code
   - Update relevant .md files

3. **Push to your fork**
   ```bash
   git push origin feature/my-new-feature
   ```

4. **Create Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill in the PR template with:
     - Description of changes
     - Motivation and context
     - Related issues
     - Testing performed
     - Screenshots (if UI changes)

5. **Address Review Feedback**
   - Respond to reviewer comments
   - Make requested changes
   - Push updates to the same branch
   - Re-request review when ready

### Pull Request Guidelines

A good pull request:

- **Focused**: Addresses one issue or feature
- **Tested**: Includes tests and passes existing tests
- **Documented**: Includes updated documentation
- **Clean**: Well-formatted code with clear commit history
- **Descriptive**: Clear PR description explaining what and why

PR Description Template:
```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Related Issues
Fixes #123
Related to #456

## Changes Made
- Added X
- Modified Y
- Removed Z

## Testing
- [ ] All tests pass
- [ ] Tested manually with [describe test case]
- [ ] Added new tests for [feature]

## Screenshots (if applicable)
[Add screenshots here]

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

### Review Process

- Maintainers will review your PR
- You may be asked to make changes
- Once approved, a maintainer will merge your PR
- Your contribution will be credited in the commit history


## License

By contributing to ld-decode, you agree that your contributions will be licensed under the GPL-3.0 license.

## Questions?

If you have questions about contributing:

1. Check the [ld-decode documentation](https://happycube.github.io/ld-decode/)
2. Search [existing issues](https://github.com/happycube/ld-decode/issues)
3. Ask in the [Discord server](https://discord.gg/pVVrrxd)
4. Open a new issue with your question

Thank you for contributing to ld-decode! 🎉
