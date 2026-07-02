#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=4dvarnet-evaluate-all
#SBATCH --output=sbatch_logs/evaluate_all_%j.out
#SBATCH --error=sbatch_logs/evaluate_all_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " Evaluate All: baselines + CFM on CS1-CS4 | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python evaluate_all.py --experiments E1 E2 E3 F1 F2 F3 G1 G2 G3

echo ""
echo "  Done at: $(date)"
echo "============================================"
