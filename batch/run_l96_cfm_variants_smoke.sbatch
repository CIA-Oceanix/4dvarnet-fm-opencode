#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-cfm-v2-v3-smoke
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/l96_cfm_variants_smoke_%j_%a.out
#SBATCH --error=sbatch_logs/l96_cfm_variants_smoke_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXPERIMENT="V3_predict_state_cfm_l96"
    echo "============================================"
    echo " L96 CFM Variants: PredictStateCFM (V3) SMOKE | $(date)"
    echo "  Task: 0 / Experiment: ${EXPERIMENT}"
    echo "  Config: 1 epoch smoke test (full: 400 epochs)"
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
    echo "============================================"
    echo

    rm -rf "experiments/${EXPERIMENT}"
    python train.py \
      --config-name experiment/${EXPERIMENT} \
      hydra.run.dir=. \
      hydra.output_subdir=null \
      training.stage1.epochs=1

    ;;

  1)
    EXPERIMENT="V2_tweedie_cfm_l96"
    echo "============================================"
    echo " L96 CFM Variants: TweedieCFM (V2) SMOKE | $(date)"
    echo "  Task: 1 / Experiment: ${EXPERIMENT}"
    echo "  Config: 1-epoch smoke test (full: 100+400 epochs)"
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
    echo "============================================"
    echo

    rm -rf "experiments/${EXPERIMENT}"
    python train.py \
      --config-name experiment/${EXPERIMENT} \
      hydra.run.dir=. \
      hydra.output_subdir=null \
      training.stage1.epochs=1 training.stage2.epochs=1

    ;;

  *)
    echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo ""
echo " SMOKE TEST PASSED: ${EXPERIMENT} at $(date)"
echo ""
