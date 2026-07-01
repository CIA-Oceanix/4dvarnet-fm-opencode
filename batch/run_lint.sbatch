#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=4dvarnet-lint
#SBATCH --output=sbatch_logs/lint_%j.out
#SBATCH --error=sbatch_logs/lint_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " 4DVarNet-FM Lint | $(date)"
echo "============================================"
echo ""

echo "=== [1/2] ruff check . ==="
ruff check .
echo "  ruff: PASS"
echo ""

echo "=== [2/2] mypy . ==="
mypy .
echo "  mypy: PASS"
echo ""

echo "============================================"
echo " All lint checks passed at $(date)"
echo "============================================"
