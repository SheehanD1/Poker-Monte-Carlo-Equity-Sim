# ============================================================================
# Poker Monte Carlo Equity Simulator — Development Makefile
# ============================================================================
#
# Usage:
#   make install    Install the package with dev dependencies
#   make lint       Run ruff linter
#   make format     Auto-format code with ruff and black
#   make typecheck  Run mypy static type checker
#   make test       Run the test suite (fast tests only)
#   make test-all   Run the full test suite including slow/benchmark tests
#   make test-cov   Run tests with coverage report
#   make check      Run lint + typecheck + test (CI-style)
#   make clean      Remove build artifacts and caches
#   make help       Show this help message
#
# ============================================================================

.DEFAULT_GOAL := help
.PHONY: install lint format typecheck test test-all test-cov check clean help

# ----------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------

PYTHON      ?= python
SRC_DIRS    := src tests
PACKAGE     := poker_equity
COV_FAIL    := 80

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------

install: ## Install the package in dev mode with all dependencies
	$(PYTHON) -m pip install -e ".[dev]"

# ----------------------------------------------------------------------------
# Code quality
# ----------------------------------------------------------------------------

lint: ## Run ruff linter on source and test files
	$(PYTHON) -m ruff check $(SRC_DIRS)

format: ## Auto-format code with ruff (fix + format) and black
	$(PYTHON) -m ruff check --fix $(SRC_DIRS)
	$(PYTHON) -m ruff format $(SRC_DIRS)
	$(PYTHON) -m black $(SRC_DIRS)

typecheck: ## Run mypy static type checker
	$(PYTHON) -m mypy src/$(PACKAGE)

# ----------------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------------

test: ## Run fast tests (skip slow and benchmark tests)
	$(PYTHON) -m pytest -m "not slow and not benchmark"

test-all: ## Run the full test suite including slow and benchmark tests
	$(PYTHON) -m pytest

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COV_FAIL)

# ----------------------------------------------------------------------------
# Combined targets
# ----------------------------------------------------------------------------

check: lint typecheck test ## Run all checks (lint + typecheck + test) — used in CI
	@echo ""
	@echo "✅ All checks passed!"

# ----------------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------------

clean: ## Remove build artifacts, caches, and generated files
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "🧹 Cleaned!"

# ----------------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------------

help: ## Show this help message
	@echo ""
	@echo "  Poker Monte Carlo Equity Simulator"
	@echo "  ==================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
