#!/bin/bash
# DEPRECATED: Use batch/run_new_experiments.sbatch instead (same experiments, updated config).
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --job-name=4dvarnet-vanilla
#SBATCH --output=sbatch_logs/vanilla_%A_%a.out
#SBATCH --error=sbatch_logs/vanilla_%A_%a.err

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
echo " 4DVarNet-FM Vanilla Experiments | $(date)"
echo "============================================"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP}"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv -i 0)"
echo ""

python train.py \
  --config-name "experiment/${EXP}" \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo "============================================"
echo " Done: ${EXP} at $(date)"
echo "============================================"
