#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=sw-tests
#SBATCH --output=sbatch_logs/sw_tests_%j.out
#SBATCH --error=sbatch_logs/sw_tests_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " SW Test Suite (CPU) | $(date)"
echo "============================================"
pytest tests/test_shallow_water.py -m "not slow" -v --tb=short
echo "============================================"
echo " Done: $(date)"
echo "============================================"
