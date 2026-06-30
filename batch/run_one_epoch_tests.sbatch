#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=4dvarnet-1epoch
#SBATCH --array=0-5%2
#SBATCH --output=sbatch_logs/one_epoch_%j_%a.out
#SBATCH --error=sbatch_logs/one_epoch_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXPERIMENTS=(
  E1_direct_unet_default
  E2_direct_unet_small
  E3_direct_unet_rand
  F1_vanilla_cfm_default
  F2_vanilla_cfm_small
  F3_vanilla_cfm_rand
)
EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " 4DVarNet-FM One-Epoch Test | $(date)"
echo "============================================"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python train.py \
  --config-name experiment/${EXP} \
  training.stage1.epochs=1 \
  +data.num_train_windows=10 \
  +data.num_val_windows=2 \
  +data.num_test_windows=5 \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo "============================================"
echo " Done: ${EXP} at $(date)"
echo "============================================"
