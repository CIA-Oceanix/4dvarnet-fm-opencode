#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=4dvarnet-preflight
#SBATCH --output=sbatch_logs/preflight_%j.out
#SBATCH --error=sbatch_logs/preflight_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " 4DVarNet-FM Preflight GPU Test | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv -i 0)"
echo ""

# Delete any stale results so this test actually runs
rm -f experiments/E1_direct_unet_default/results.json

python train.py \
  --config-name experiment/E1_direct_unet_default \
  training.stage1.epochs=1 \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo "  GPU mem after: $(nvidia-smi --query-gpu=memory.used --format=csv -i 0)"
echo "  Done at: $(date)"
echo "============================================"
