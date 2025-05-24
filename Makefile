# Makefile for MAB VRU Signalisation project

.PHONY: help install test lint format type-check security clean docs build run run-parallel

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies and package in development mode"
	@echo "  test          - Run tests with coverage"
	@echo "  test-unit     - Run only unit tests"
	@echo "  test-integration - Run only integration tests"
	@echo "  lint          - Run linting checks (flake8)"
	@echo "  format        - Format code with black and isort"
	@echo "  format-check  - Check code formatting without changing files"
	@echo "  type-check    - Run mypy type checking"
	@echo "  security      - Run security checks with bandit"
	@echo "  quality       - Run all quality checks (lint, type-check, security)"
	@echo "  clean         - Clean temporary files and caches"
	@echo "  docs          - Generate documentation"
	@echo "  build         - Build package"
	@echo "  run           - Run simulation with default config"
	@echo "  run-parallel  - Run simulation with parallel execution"
	@echo "  env-example   - Show example environment variables"

# Development setup
install:
	pip install -e .
	pip install pytest pytest-cov black isort mypy flake8 bandit

# Testing
test:
	PYTHONPATH=src pytest --cov=src/mab_vru --cov-report=html --cov-report=term-missing

test-unit:
	PYTHONPATH=src pytest tests/unit/ -v

test-integration:
	PYTHONPATH=src pytest tests/integration/ -v -m "not slow"

test-slow:
	pytest -m slow

# Code quality
lint:
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

type-check:
	mypy src/

security:
	bandit -r src/ -f json -o security-report.json || bandit -r src/

quality: format-check lint type-check security
	@echo "All quality checks passed!"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf security-report.json

# Documentation
docs:
	@echo "Documentation generation would go here"
	@echo "Consider using Sphinx or MkDocs"

# Build
build: clean
	python -m build

# Run simulations
run:
	python -m src.mab_vru.main

run-parallel:
	MAB_ENABLE_PARALLEL=true python -m src.mab_vru.main

run-env:
	MAB_USE_ENV_CONFIG=true python -m src.mab_vru.main

# Environment variables example
env-example:
	@echo "Example environment variables:"
	@echo "export MAB_USE_ENV_CONFIG=true"
	@echo "export MAB_ENABLE_PARALLEL=true"
	@echo "export MAB_V2V_NETWORK_LOAD=0.15"
	@echo "export MAB_V2I_NETWORK_LOAD=0.08"
	@echo "export MAB_ALGORITHMS='ucb,epsilon-greedy'"
	@echo "export MAB_INFRA_POSITIONS='0,0;100,100;200,200'"
	@echo "export MAB_TRACE_FILE='sumoTraceCroisement.csv'"
	@echo "export MAB_CSV_OUTPUT='results/custom_results.csv'"

# Development workflow
dev-setup: install
	@echo "Development environment set up!"
	@echo "Run 'make test' to verify everything works"

# Continuous integration simulation
ci: format-check lint type-check test
	@echo "CI checks completed successfully!"

# Performance testing
perf-test:
	@echo "Running performance tests..."
	time make run
	@echo "Performance test completed!"
