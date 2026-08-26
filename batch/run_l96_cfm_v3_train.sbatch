#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-v3-cfm
#SBATCH --output=sbatch_logs/l96_v3_cfm_%j.out
#SBATCH --error=sbatch_logs/l96_v3_cfm_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXPERIMENT="V3_predict_state_cfm_s0s1"

echo "============================================"
echo " L96 V3 PredictStateCFM Training | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Config: predict_state_cfm, single-stage, 400 epochs"
echo "  obs_j=2 (24D subspace), all params randomized +-20%"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

# Remove any stale smoke-test results so the full run starts fresh.
rm -rf "experiments/${EXPERIMENT}"

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo "============================================"
