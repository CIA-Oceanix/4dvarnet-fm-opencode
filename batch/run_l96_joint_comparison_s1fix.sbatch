#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-joint-s1fix
#SBATCH --output=sbatch_logs/l96_joint_s1fix_%j.out
#SBATCH --error=sbatch_logs/l96_joint_s1fix_%j.err

set -euo pipefail

REPO_ROOT="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/sbatch_logs"

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " L96 Joint DA Benchmark (S1 corrupted-forcing fix) | $(date)"
echo "  Re-runs the canonical obsj2 joint comparison on master with"
echo "  the cfg_s1 case=2 fix (S1 DA now feeds forcing_corrupted)"
echo "  Parallel output: l96_joint_comparison_s1cfix.json"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python eval_joint_comparison_l96.py \
  --device cuda \
  --obs-interval 100 \
  --obs-j 2 \
  --s1-j 2 \
  --eval-j 2 \
  --num-test-windows 200 \
  --batch-size 200 \
  --out-json experiments/l96_joint_comparison_s1cfix.json

echo ""
echo " Done at $(date)"
echo "============================================"
