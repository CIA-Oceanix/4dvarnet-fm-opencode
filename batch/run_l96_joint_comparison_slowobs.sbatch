#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-joint-slowobs
#SBATCH --output=sbatch_logs/l96_joint_slowobs_%j.out
#SBATCH --error=sbatch_logs/l96_joint_slowobs_%j.err

set -euo pipefail

REPO_ROOT="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/sbatch_logs"

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " L96 Joint DA Benchmark (slow-only obs, state+param) | $(date)"
echo "  obs_j=0 (8 slow X observed)"
echo "  S1 reduced dynamics kept at J=2 (24D state space)"
echo "  eval subspace = 24D obsj2 group"
echo "  methods: EnKF/Joint-EnKF, ETKF/Joint-ETKF, Strong-4DVar/Joint-Strong-4DVar"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python eval_joint_comparison_l96.py \
  --device cuda \
  --obs-interval 100 \
  --obs-j 0 \
  --s1-j 2 \
  --eval-j 2 \
  --num-test-windows 200 \
  --batch-size 200 \
  --out-json experiments/l96_joint_comparison_slowobs.json

echo ""
echo " Done at $(date)"
echo "============================================"
