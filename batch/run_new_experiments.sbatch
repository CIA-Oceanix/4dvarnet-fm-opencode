#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=4dvarnet-new-exp
#SBATCH --array=0-5
#SBATCH --output=sbatch_logs/new_exp_%j_%a.out
#SBATCH --error=sbatch_logs/new_exp_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

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
echo " 4DVarNet-FM New Experiment | $(date)"
echo "============================================"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

python train.py \
  --config-name experiment/${EXP} \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo "============================================"
echo " Done: ${EXP} at $(date)"
echo "============================================"
