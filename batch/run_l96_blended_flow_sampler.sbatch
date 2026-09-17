#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=l96-blended-flow
#SBATCH --output=sbatch_logs/blended_flow_%j.out
#SBATCH --error=sbatch_logs/blended_flow_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

# monai checkpoints require the monai-enabled env; the plain `fdv` env fails with a
# late ModuleNotFoundError inside train.py::model_factory.
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONPATH=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PYTHONUNBUFFERED=1

MEMBERS=${MEMBERS:-30}
WINDOWS=${WINDOWS:-200}
N_OUTER=${N_OUTER:-50}
OUTPUT=${OUTPUT:-experiments/l96_blended_flow_m${MEMBERS}_w${WINDOWS}.json}

echo "============================================"
echo " L96 blended-flow sampler | $(date)"
echo "  members=${MEMBERS} windows=${WINDOWS} n_outer=${N_OUTER}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python reports/l96/run_blended_flow_sampler.py \
  --prior-ckpt experiments/l96/SDA3_monai_cond_noisy_l96_norm/checkpoints/stage1_best.ckpt --cond \
  --mean-ckpt experiments/l96/L1b_monai_unet_s0s1_norm_obsdensity/checkpoints/stage1_best.ckpt \
  --windows "${WINDOWS}" --chunk 10 --members "${MEMBERS}" --n-outer "${N_OUTER}" \
  --lam0s 0.5 --ps 1 2 --tau0 0.3 \
  --output "${OUTPUT}"

echo "done: $(date)"
