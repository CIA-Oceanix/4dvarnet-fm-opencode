#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=l96-cfm-variants-eval
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/l96_cfm_variants_eval_%j_%a.out
#SBATCH --error=sbatch_logs/l96_cfm_variants_eval_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXPERIMENT="V3_predict_state_cfm_l96"
    CHECKPOINT="checkpoints/stage1_best.ckpt"
    echo "============================================"
    echo " L96 CFM Variants: PredictStateCFM Eval | $(date)"
    echo "  Task: 0 / Experiment: ${EXPERIMENT}"
    echo "  Checkpoint: ${CHECKPOINT}"
    echo "  Output: experiments/${EXPERIMENT}/neural_eval.json"
    echo "============================================"
    echo ""

    python eval_neural_l96.py \
      --checkpoint "experiments/${EXPERIMENT}/${CHECKPOINT}" \
      --dataset "experiments/l96_datasets_obsj2_int100_nwin200.pt" \
      --num-windows 200 \
      --obs-interval 100 \
      --obs-j 2 \
      --device cuda \
      --output "experiments/${EXPERIMENT}/neural_eval.json"
    ;;

  1)
    EXPERIMENT="V2_tweedie_cfm_l96"
    # V2 is two-stage: stage2_best.ckpt contains the full model (mean_estimator + velocity_unet)
    CHECKPOINT="checkpoints/stage2_best.ckpt"
    echo "============================================"
    echo " L96 CFM Variants: TweedieCFM Eval | $(date)"
    echo "  Task: 1 / Experiment: ${EXPERIMENT}"
    echo "  Checkpoint: ${CHECKPOINT} (Stage 2, full model)"
    echo "  Output: experiments/${EXPERIMENT}/neural_eval.json"
    echo "============================================"
    echo ""

    python eval_neural_l96.py \
      --checkpoint "experiments/${EXPERIMENT}/${CHECKPOINT}" \
      --dataset "experiments/l96_datasets_obsj2_int100_nwin200.pt" \
      --num-windows 200 \
      --obs-interval 100 \
      --obs-j 2 \
      --device cuda \
      --output "experiments/${EXPERIMENT}/neural_eval.json"
    ;;

  *)
    echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo "============================================"
