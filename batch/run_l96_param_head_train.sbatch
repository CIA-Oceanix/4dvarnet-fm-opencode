#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-param-head
#SBATCH --output=sbatch_logs/l96_param_head_%j.out
#SBATCH --error=sbatch_logs/l96_param_head_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-joint-neural

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXP="${EXP:-C1_stateparam_head_s1}"

echo "============================================"
echo " L96 Cascade param-head training (${EXP}) | $(date)"
echo "  Decoupled param estimation fed by frozen L1b DirectUNet state estimator"
echo "  Inputs: (obs, biased *_da params, corrupted forcing, L1b state estimate)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

# Remove stale smoke results so the full run starts fresh.
rm -rf "experiments/${EXP}"

# Optional epoch override for smoke runs.
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
EPOCHS_ARG=""
if [ -n "$EPOCHS_OVERRIDE" ]; then
  EPOCHS_ARG="++training.stage1.epochs=${EPOCHS_OVERRIDE}"
fi

python train.py \
  --config-name experiment/${EXP} \
  ++data.test_cache=experiments/l96_datasets_obsj2_int100_nwin200.pt \
  ${EPOCHS_ARG} \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo " Done: ${EXP} at $(date)"
echo "============================================"