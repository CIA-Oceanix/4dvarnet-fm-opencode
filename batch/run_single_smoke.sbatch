#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=sbatch_logs/smoke_%x_%j.out
#SBATCH --error=sbatch_logs/smoke_%x_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:${PATH}"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" sbatch_logs experiments

EXP="${1}"

echo "============================================"
echo " Smoke Test (1 epoch) | $(date)"
echo " Experiment: ${EXP}"
echo "============================================"

python train.py \
  --config-name "experiment/${EXP}" \
  training.stage1.epochs=1 \
  data.num_train_windows=10 \
  data.num_val_windows=2 \
  data.num_test_windows=5 \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo "Done: ${EXP} at $(date)"
