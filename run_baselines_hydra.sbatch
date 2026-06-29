#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=4dvarnet-baselines-hydra
#SBATCH --output=sbatch_logs/baselines_hydra_%j.out
#SBATCH --error=sbatch_logs/baselines_hydra_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " 4DVarNet-FM Baselines (Hydra) | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
  echo "  DWS: ${DWS:-300}"
echo ""

export HYDRA_FULL_ERROR=1

python eval_baselines.py \
  --config-name baselines/dws${DWS:-300} \
  baselines.batch_size=128 \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo "  Done at: $(date)"
echo "============================================"
