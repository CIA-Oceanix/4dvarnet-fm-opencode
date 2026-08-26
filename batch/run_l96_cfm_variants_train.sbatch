#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-cfm-variants-train
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/l96_cfm_variants_train_%j_%a.out
#SBATCH --error=sbatch_logs/l96_cfm_variants_train_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXPERIMENT="V3_predict_state_cfm_l96"
    echo "============================================"
    echo " L96 CFM Variants: PredictStateCFM (V3) | $(date)"
    echo "  Task: 0 / Experiment: ${EXPERIMENT}"
    echo "  Config: 24D state, 400 epochs, single-stage CFM"
    echo "   - PredictStateCFM: v(x) = (μ(x) - x) / (1 - τ)"
    echo "   - ODE sampling with deterministic τ-schedule"
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
    echo "============================================"
    echo ""

    rm -rf "experiments/${EXPERIMENT}"
    python train.py \
      --config-name experiment/${EXPERIMENT} \
      hydra.run.dir=. \
      hydra.output_subdir=null
    ;;

  1)
    EXPERIMENT="V2_tweedie_cfm_l96"
    echo "============================================"
    echo " L96 CFM Variants: TweedieCFM (V2) | $(date)"
    echo "  Task: 1 / Experiment: ${EXPERIMENT}"
    echo "  Config: 24D state, 200-stage1 + 400-stage2 epochs, two-stage CFM"
    echo "   - Stage 1: MeanEstimator (trained on obs → mean(x1|x0,y))"
    echo "   - Stage 2: velocity_unet (trained on (x1 - μ)/K_inner - v)"
    echo "   - ODE sampling: integrate v(x, τ) across K_inner steps"
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
    echo "============================================"
    echo ""

    rm -rf "experiments/${EXPERIMENT}"
    python train.py \
      --config-name experiment/${EXPERIMENT} \
      hydra.run.dir=. \
      hydra.output_subdir=null
    ;;

  *)
    echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo "============================================"
