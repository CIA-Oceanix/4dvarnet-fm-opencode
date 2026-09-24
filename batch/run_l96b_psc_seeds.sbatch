#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-3
#SBATCH --job-name=l96b-psc
#SBATCH --output=sbatch_logs/l96b_psc_seed%a_%A.out
#SBATCH --error=sbatch_logs/l96b_psc_seed%a_%A.err
# PredictStateCFM-M under the benchmark default, seed = array index; then the
# benchmark flow eval (ens30 x 20 early-fine steps) on the regular grid and on the canonical
# random-observing-system test set.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
SEED=$SLURM_ARRAY_TASK_ID
NAME="L96B_predictstatecfm_monaiM_seed${SEED}"
echo "===== ${NAME} | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
python train.py --config-name experiment/L96B_predictstatecfm_monaiM hydra.run.dir=. hydra.output_subdir=null training.seed=${SEED}
CKPT="experiments/${NAME}/checkpoints/stage1_best.ckpt"
OUT="experiments/${NAME}/ens30_no20"; mkdir -p "$OUT"
python eval_neural_l96.py --checkpoint "$CKPT" --dataset $R/experiments/l96_datasets_obsj2_int100_nwin200.pt \
  --num-windows 200 --n-members 30 --n-outer 20 --step-power 0.5 --batch-size 25 --output "$OUT/neural_eval.json"
OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens30_no20"; mkdir -p "$OUT"
python eval_neural_l96.py --checkpoint "$CKPT" --dataset $R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt \
  --num-windows 200 --n-members 30 --n-outer 20 --step-power 0.5 --batch-size 25 --output "$OUT/neural_eval.json"
echo "Done at $(date)"
