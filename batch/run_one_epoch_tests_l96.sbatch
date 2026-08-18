#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=l96-1epoch
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/l96_one_epoch_%j_%a.out
#SBATCH --error=sbatch_logs/l96_one_epoch_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXPERIMENTS=(
  L1_direct_unet_s0s1
  L2_vanilla_cfm_s0s1
)
EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " L96 One-Epoch Test | $(date)"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

python train.py \
  --config-name experiment/${EXP} \
  training.stage1.epochs=1 \
  data.num_train_windows=10 \
  data.num_val_windows=2 \
  data.num_test_windows=5 \
  data.spinup_steps=3000 \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo "  Cleaning smoke-test artifacts for experiment ${EXP}..."
rm -rf "experiments/${EXP}"

echo ""
echo " Done: ${EXP} at $(date)"
echo "============================================"
