#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --job-name=l96b-flows-ep1200
#SBATCH --output=sbatch_logs/l96b_flows_ep1200_%A_%a.out
#SBATCH --error=sbatch_logs/l96b_flows_ep1200_%A_%a.err
# VanillaCFM-M / PredictStateCFM-M, 1000 windows x 1200 epochs, seed 1, reusing
# the cached 1000-window train/val set; then the P1 flow eval (ens30 x 10
# steps) on both test sets, scores only.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
FAMS=(vanillacfm predictstatecfm)
F=${FAMS[$SLURM_ARRAY_TASK_ID]}; NAME=L96B_${F}_monaiM_ep1200_seed1
python train.py --config-name experiment/L96B_${F}_monaiM_ep1200 hydra.run.dir=. hydra.output_subdir=null \
  training.seed=1 ++data.train_cache=$R/experiments/l96_trainval_cache/l96b_trainval_n1000_v100.pt
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="experiments/${NAME}/ens30_no10";
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens30_no10"; fi
  mkdir -p "$OUT"
  python eval_neural_l96.py --step-power 1 --checkpoint "experiments/${NAME}/checkpoints/stage1_best.ckpt" --dataset "$DS" \
    --num-windows 200 --n-members 30 --n-outer 10 --batch-size 25 --members-file scores --output "$OUT/neural_eval.json"
done
echo "Done: ${NAME} at $(date)"
