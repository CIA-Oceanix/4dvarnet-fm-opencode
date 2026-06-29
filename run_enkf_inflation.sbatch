#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=4dvarnet-enkf-inf
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/enkf_inflation_%j_%a.out
#SBATCH --error=sbatch_logs/enkf_inflation_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

INFLATION_VALUES=(1.05 1.1 1.15 1.2)
INFLATION=${INFLATION_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " EnKF Inflation Sweep | $(date)"
echo "============================================"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / Inflation: ${INFLATION}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

export HYDRA_FULL_ERROR=1

python eval_baselines.py \
  --config-name baselines/dws300 \
  baselines.enkf.inflation=$INFLATION \
  baselines.batch_size=128 \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo "  Done at: $(date)"
echo "============================================"
