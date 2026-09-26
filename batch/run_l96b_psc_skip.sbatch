#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --job-name=l96b-psc-skip
#SBATCH --output=sbatch_logs/l96b_psc_skip_%a_%A.out
#SBATCH --error=sbatch_logs/l96b_psc_skip_%a_%A.err
# PredictStateCFM-M, 1200 epochs, benchmark-default data (cached train/val windows),
# with the linear skip connection D = tau * x_tau + (1 - tau) * net; task 0 trains
# MSE(D, x1) (skip_loss state), task 1 the unweighted net loss (skip_loss net). Seed 1,
# then the benchmark flow eval (30 members x 20 early-fine steps) on the regular grid
# and the canonical random test set -- directly comparable with
# L96B_predictstatecfm_monaiM_ep1200_seed1.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-psc-skip
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
SEED=1
LOSSES=(state net); EXP=L96B_predictstatecfm_monaiM_ep1200_skip_${LOSSES[$SLURM_ARRAY_TASK_ID]}loss
NAME="${EXP}_seed${SEED}"
echo "===== ${NAME} | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
[ -f experiments/${NAME}/checkpoints/stage1_best.ckpt ] || \
  python train.py --config-name experiment/${EXP} hydra.run.dir=. hydra.output_subdir=null \
    training.seed=${SEED} ++data.train_cache=$R/experiments/l96_trainval_cache/l96b_trainval_n1000_v100.pt
CKPT="experiments/${NAME}/checkpoints/stage1_best.ckpt"
for TS in regular canonical; do
  if [ $TS = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="experiments/${NAME}/ens30_no20"
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens30_no20"; fi
  [ -f "$OUT/neural_eval.json" ] && continue; mkdir -p "$OUT"
  python eval_neural_l96.py --checkpoint "$CKPT" --dataset $DS --num-windows 200 --n-members 30 --n-outer 20 \
    --step-power 0.5 --batch-size 25 --members-file scores --output "$OUT/neural_eval.json"
done
echo "Done at $(date)"
