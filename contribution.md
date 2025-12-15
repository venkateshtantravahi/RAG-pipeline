# Contributing to RAG Pipeline

Thank you for your interest in contributing! I love input from the community, whether it's bug reports, feature requests, or code contributions.

This document outlines the standards and workflows I follow to keep the codebase high-quality and production-ready.

---

## Table of Contents

1.  [Getting Started](#getting-started)
2.  [Development Workflow](#development-workflow)
3.  [Coding Standards](#coding-standards)
4.  [Testing Guidelines](#testing-guidelines)
5.  [Pull Request Process](#pull-request-process)
6.  [Reporting Bugs](#reporting-bugs)

---

## Getting Started

### 1. Fork & Clone
Fork the repository to your GitHub account and clone it locally:
```bash
git clone [https://github.com/venkateshtantravahi/RAG-pipeline.git](https://github.com/venkateshtantravahi/RAG-pipeline.git)
cd RAG-pipeline
```

### 2. Environment Setup
We use conda and make to manage environments. Run the following to set up your local machine:
```bash
# Create the environment (Python 3.12)
make create_environment

# Activate the environment
conda activate rag-pipeline

# Install all dependencies
make requirements
```

## Development Workflow

### 1. Create a Branch
Never push directly to main. Always create a feature branch for your work or use the existing feature branches:
```bash
# Format: type/short-description
git checkout -b feat/add-reranker
# or
git checkout -b fix/api-timeout
```

### 2. Make Your Changes
Write your code. Keep changes atomic (focused on one task).

### 3. Verify Changes
Before committing, ensure your code builds and tests pass:
```bash
# Run the test suite
make test

# Run code formatting
make format
```

## Coding Standards
We enforce strict coding standards to maintain "Production Grade" quality.

Style Guide
- **Formatter:** We use `black` (or `ruff` via Makefile). Always run `make format` before pushing.

- **Type Hints:** All function signatures must have Python type hints.
```python
#  GOOD
def search(query: str, k: int = 5) -> List[Document]: ...

#  BAD
def search(query, k=5): ...
```
- **Docstrings:** All modules, classes, and public functions must have docstrings describing arguments, returns, and behavior.

### Warning Policy
Do not ignore warnings globally.

If you encounter a warning from an external library, use surgical suppression in `tests/conftest.py` rather than sweeping it under the rug in `pyproject.toml`.

## Testing Guidelines
Code without tests will not be merged.

- **Unit Tests:** Place them in `tests/`. Mirror the structure of `src/`.

- **Mocking:** Do not rely on external services (Ollama, ChromaDB) for unit tests. Use `unittest.mock` to mock heavy dependencies.

- **Performance:** Tests should run quickly. If a test takes >1s, verify you aren't loading a model unnecessarily.

Run the full test:
```bash
make test
```

## Pull Request Process
1. **Update Documentation:** If you change an API endpoint or logic, update `README.md` and docstrings.

2. **Pass CI:** Ensure all tests pass locally.

3. **Descriptive Title:** Use [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat: add streaming support` or `fix: broken pipe in ingestion`).

4. **Description:** Describe what you changed and why.

5. **Review:** Wait for a maintainer to review your code. Address any comments promptly.

## Reporting Bugs
If you find a bug, please open an Issue using the following template:

**Title:** [Bug] Short description of the error

**Description:**

- **OS:** (e.g., macOS Sequoia, Ubuntu 22.04)

- **Python Version:** (e.g., 3.12.1)

- **Steps to Reproduce**:

    1. Run `make setup`

    2. Run ...

- **Expected Behavior:** What should have happened?

- **Actual Behavior:** What actually happened? (Include stack traces/logs)