#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-strong4dvar
#SBATCH --output=sbatch_logs/l96_strong4dvar_%j.out
#SBATCH --error=sbatch_logs/l96_strong4dvar_%j.err

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
echo " L96 Joint DA Benchmark (vanilla Strong-4DVar) | $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

# Re-run only vanilla (state-only) Strong-4DVar at batch-size 200 (the fast
# state-only config) on the cached S0/S1 test set (Obs30, 200 windows), so its
# row appears in experiments/l96_joint_comparison.json in the comparator schema
# (the full 6-method JSON is assembled from this + master's vanilla ETKF/EnKF +
# the refreshed joint rows). No reconstruction npz saved for this run (vanilla
# Strong-4DVar trajectories already exist on master's state-only cache).
python eval_joint_comparison_l96.py \
  --device cuda \
  --obs-interval 100 \
  --obs-j 2 \
  --num-test-windows 200 \
  --batch-size 200 \
  --methods Strong-4DVar

echo ""
echo " Done at $(date)"
echo "============================================"
