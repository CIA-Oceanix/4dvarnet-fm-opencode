#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=l96-v2-ablation-eval
#SBATCH --array=0-2
#SBATCH --output=sbatch_logs/l96_v2_ablation_eval_%j_%a.out
#SBATCH --error=sbatch_logs/l96_v2_ablation_eval_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR"

DATASET="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

TASKS=(
  "V2_tweedie_cfm_l96_rerun"
  "V2_tweedie_cfm_l96_s0p2"
  "V2_tweedie_cfm_l96_kinner1"
)
EXP=${TASKS[$SLURM_ARRAY_TASK_ID]}
EXP_DIR="experiments/${EXP}"
ENS_DIR="${EXP_DIR}/ens30_no10"
mkdir -p "${ENS_DIR}"

echo "============================================"
echo " L96 V2 TweedieCFM Ablation Eval | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

# Pass 1: single-sample eval (N=1) -> experiment root.
python eval_neural_l96.py \
  --checkpoint "${EXP_DIR}/checkpoints/stage2_best.ckpt" \
  --config "config/experiment/${EXP}.yaml" \
  --dataset "${DATASET}" \
  --num-windows 200 \
  --obs-interval 100 \
  --obs-j 2 \
  --device cuda \
  --output "${EXP_DIR}/neural_eval.json"

# Pass 2: ens30 x10 ensemble eval -> ens30_no10/ (apples-to-apples with L3/L9).
python eval_neural_l96.py \
  --checkpoint "${EXP_DIR}/checkpoints/stage2_best.ckpt" \
  --config "config/experiment/${EXP}.yaml" \
  --dataset "${DATASET}" \
  --num-windows 200 \
  --obs-interval 100 \
  --obs-j 2 \
  --device cuda \
  --n-members 30 \
  --n-outer 10 \
  --seed 0 \
  --output "${ENS_DIR}/neural_eval.json"

echo ""
echo " Done: ${EXP} at $(date)"
echo ""
