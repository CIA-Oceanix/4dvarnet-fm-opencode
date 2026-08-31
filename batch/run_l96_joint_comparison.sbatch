#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-joint-strong4dvar
#SBATCH --output=sbatch_logs/l96_joint_strong4dvar_%j.out
#SBATCH --error=sbatch_logs/l96_joint_strong4dvar_%j.err

set -euo pipefail

# Work from this repo root. Note: slurm relocates the sbatch script into
# /var/spool/slurm/slurmd/<jobid>/, so $0 / BASH_SOURCE / readlink all resolve to
# the spool copy, NOT this repo. Use a hardcoded absolute path (same convention as
# run_l96_joint_neural_eval.sbatch).
REPO_ROOT="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-joint-da"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/sbatch_logs"

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " L96 Joint DA Benchmark (Joint-Strong-4DVar) | $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

# Joint-Strong-4DVar on the cached S0/S1 test set (Obs30, 200 windows).
# Runs a batched, purely-gradient Adam solve over all 200 windows at once
# (batch-size 200 == the state-only Strong-4DVar benchmark's batch) and MERGES
# into experiments/l96_joint_comparison.json so the existing EnKF/ETKF rows
# (from the earlier benchmarks) are preserved in the file.
# (Vanilla Strong-4DVar is ported from the canonical DA cache, not re-run.)
python eval_joint_comparison_l96.py \
  --device cuda \
  --obs-interval 100 \
  --obs-j 2 \
  --num-test-windows 200 \
  --batch-size 200 \
  --methods Joint-Strong-4DVar

echo ""
echo " Done at $(date)"
echo "============================================"
