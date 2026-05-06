#!/usr/bin/env zsh

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================"
echo " Running Unit Tests with 100% Coverage"
echo "========================================"

# --cov: specifies which modules to measure coverage for.
# --cov-report=term-missing: Prints the exact line numbers missing coverage to the console.
# --cov-fail-under=100: Fails the CI/Script (returns exit code 1) if coverage is < 100%.

# check every file for coverage
pytest --cov=quantum_experiment_structures --cov-report=term-missing --cov-fail-under=100 unittests

echo "========================================"
echo " Success! 100% Coverage Maintained."
echo "========================================"
