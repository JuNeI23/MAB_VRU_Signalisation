# Test Suite for VRU Signalization Project

This directory contains the test suite for the VRU Signalization project, including unit tests, integration tests, and test fixtures.

## Structure

```
tests/
├── conftest.py           # Pytest configuration and shared fixtures
├── requirements.txt      # Test dependencies
├── fixtures/            # Test data and fixtures
│   └── sample_trace.csv  # Sample SUMO trace data for testing
├── unit/               # Unit tests
│   ├── test_mab.py     # Tests for MAB algorithms
│   └── test_models.py  # Tests for simulation models
└── integration/        # Integration tests
    └── test_simulation.py  # End-to-end simulation tests
```

## Running Tests

1. Install test dependencies:
```bash
pip install -r tests/requirements.txt
```

2. Run all tests:
```bash
pytest
```

3. Run specific test files:
```bash
pytest tests/unit/test_mab.py  # Run MAB tests only
pytest tests/integration/      # Run all integration tests
```

4. Run tests with coverage report:
```bash
pytest --cov=simulation --cov=MAB --cov-report=html
```

## Test Categories

### Unit Tests
- `test_mab.py`: Tests for Multi-Armed Bandit implementations
- `test_models.py`: Tests for Node, User, and Infrastructure models

### Integration Tests
- `test_simulation.py`: End-to-end tests for the simulation pipeline

### Fixtures
- `conftest.py`: Shared test fixtures and configurations
- `fixtures/sample_trace.csv`: Sample SUMO trace data for testing

## Coverage Requirements

The test suite aims for at least 80% code coverage. Coverage reports can be found in:
- HTML: `tests/coverage_html/`
- Console: Run pytest with `--cov` flag
