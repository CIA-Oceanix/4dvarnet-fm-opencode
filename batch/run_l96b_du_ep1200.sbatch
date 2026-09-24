#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=l96b-du-ep1200
#SBATCH --output=sbatch_logs/l96b_du_ep1200_%j.out
#SBATCH --error=sbatch_logs/l96b_du_ep1200_%j.err
# DirectUNet-M, 1000 windows x 1200 epochs (step-count control for the
# 3000-window run), seed 1, train/val windows cached for reuse; then the P1
# deterministic eval on both test sets.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
NAME=L96B_directunet_monaiM_ep1200_seed1
python train.py --config-name experiment/L96B_directunet_monaiM_ep1200 hydra.run.dir=. hydra.output_subdir=null \
  training.seed=1 ++data.train_cache=$R/experiments/l96_trainval_cache/l96b_trainval_n1000_v100.pt
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="experiments/${NAME}/ens1_no1";
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens1_no1"; fi
  mkdir -p "$OUT"
  python eval_neural_l96.py --step-power 0.5 --checkpoint "experiments/${NAME}/checkpoints/stage1_best.ckpt" --dataset "$DS" \
    --num-windows 200 --n-members 1 --n-outer 1 --batch-size 25 --output "$OUT/neural_eval.json"
done
echo "Done at $(date)"
