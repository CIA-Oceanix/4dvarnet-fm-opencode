#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=s0s1-joint-smoke
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/s0s1_joint_smoke_%A_%a.out
#SBATCH --error=sbatch_logs/s0s1_joint_smoke_%A_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments

EXPERIMENTS=(
  S5_joint_cfm_s0s1
  S6_joint_cfm_s0s1_tau0
)
EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " 4DVarNet-FM S0/S1 Joint Smoke (1 epoch) | $(date)"
echo "============================================"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python train.py \
  --config-name experiment/${EXP} \
  training.stage1.epochs=1 \
  data.num_train_windows=10 \
  data.num_val_windows=2 \
  data.num_test_windows=5 \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo "============================================"
echo " Done: ${EXP} at $(date)"
echo "============================================"
