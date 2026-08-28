#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=l96-joint-da
#SBATCH --output=sbatch_logs/l96_joint_da_%j.out
#SBATCH --error=sbatch_logs/l96_joint_da_%j.err

set -euo pipefail

# Work from this script's repo root (works in any worktree, before/after merge).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p sbatch_logs

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " L96 Joint DA Benchmark (ETKF) | $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

# Joint-ETKF + vanilla ETKF on the cached S0/S1 test set (Obs30, 200 windows).
python eval_joint_comparison_l96.py \
  --device cuda \
  --obs-interval 100 \
  --obs-j 2 \
  --num-test-windows 200 \
  --etkf-inflation 1.6 \
  --batch-size 10

echo ""
echo " Done at $(date)"
echo "============================================"
