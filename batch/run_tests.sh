#!/bin/bash
# DEPRECATED: Use batch/run_test_suite.sbatch (sbatch) instead.
# Quick Test Suite Runner for Lorenz-63 DA Baselines
# Created: June 25, 2026

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

echo "====================================="
echo "Lorenz-63 DA Baselines - Test Runner"
echo "====================================="
echo ""

# Check if we're in the right directory
if [ ! -d "tests" ]; then
    echo "❌ Error: tests/ directory not found!"
    echo "Please run this script from the repository root."
    exit 1
fi

# Activate conda environment
echo "🔧 Activating conda environment 'fdv'..."
source activate fdv 2>/dev/null || conda activate fdv

# Option 1: Fast tests only (recommended for quick validation)
echo ""
echo "Option 1: Fast Tests (16 tests, ~25 seconds)"
echo "============================================="
echo "Run: pytest tests/test_metrics.py tests/test_lorenz63.py -v"
echo ""
read -p "Run fast tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pytest tests/test_metrics.py tests/test_lorenz63.py -v --tb=short
    echo ""
fi

# Option 2: Full test suite (slow, includes DA optimization)
echo ""
echo "Option 2: Full Test Suite (36 tests, ~5-10 minutes)"
echo "===================================================="
echo "Run: pytest tests/ -v"
echo ""
read -p "Run full test suite? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pytest tests/ -v --tb=short
    echo ""
fi

# Option 3: Coverage report
echo ""
echo "Option 3: Coverage Report"
echo "========================="
echo "Run: pytest tests/ --cov=data --cov=evaluation --cov-report=html"
echo ""
read -p "Generate coverage report? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pytest tests/test_metrics.py tests/test_lorenz63.py --cov=data --cov=evaluation --cov-report=html --cov-report=term
    echo ""
    echo "✅ Coverage report saved to htmlcov/index.html"
    echo ""
fi

echo ""
echo "====================================="
echo "Test suite complete!"
echo "====================================="
echo ""
echo "📚 For more information, see TEST_SUITE_SUMMARY.md"
echo ""
echo "Quick reference:"
echo "  - Fast tests:     pytest tests/test_metrics.py tests/test_lorenz63.py -v"
echo "  - Full suite:     pytest tests/ -v"
echo "  - Skip slow:      pytest tests/ -m 'not slow' -v"
echo "  - Only slow:      pytest tests/ -m 'slow' -v"
echo "  - With coverage:  pytest tests/ --cov=data --cov=evaluation"
echo ""
