#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-v2-ablation-smoke
#SBATCH --array=0-2
#SBATCH --output=sbatch_logs/l96_v2_ablation_smoke_%j_%a.out
#SBATCH --error=sbatch_logs/l96_v2_ablation_smoke_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR"

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

TASKS=(
  "V2_tweedie_cfm_l96_rerun"
  "V2_tweedie_cfm_l96_s0p2"
  "V2_tweedie_cfm_l96_kinner1"
)
EXP=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " L96 V2 TweedieCFM Ablation SMOKE: ${EXP} | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID} / 1-epoch smoke (tiny train/val windows)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

rm -rf "experiments/${EXP}"
python train.py \
  --config-name experiment/${EXP} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ++data.test_cache=${TEST_CACHE} \
  ++data.num_train_windows=5 \
  ++data.num_val_windows=2 \
  training.stage1.epochs=1 \
  training.stage2.epochs=1

echo ""
echo " Done: ${EXP} smoke at $(date)"
echo ""
