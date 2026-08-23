#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-l3-mtau
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXP=L3_vanilla_cfm_s0s1

echo "============================================"
echo " L96 Neural Training (L3 multi-tau CFM) | $(date)"
echo "  Experiment: ${EXP} (clone of L2b with train_tau_0_only=false)"
echo "  Config inherits lorenz96_default (all params randomized +-20%; partial obs obs_j=2)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

# Remove any stale smoke-test results so the full run starts fresh.
rm -rf "experiments/${EXP}"

python train.py \
  --config-name experiment/${EXP} \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo " Done: ${EXP} at $(date)"
echo "============================================"
