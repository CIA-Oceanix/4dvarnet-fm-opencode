#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-5%4
#SBATCH --job-name=l96b-ep1200-seeds
#SBATCH --output=sbatch_logs/l96b_ep1200_seeds_%A_%a.out
#SBATCH --error=sbatch_logs/l96b_ep1200_seeds_%A_%a.err
# Seeds 2-3 of the 1200-epoch arms (DirectUNet-M, PredictStateCFM-M,
# VanillaCFM-M; 1000 windows from the shared train_cache), then the P1 eval on
# both test sets (deterministic / ens30 x 10 steps, scores only).
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
FAMS=(directunet predictstatecfm vanillacfm); MEM=(1 30 30); OUTER=(1 10 10)
i=$SLURM_ARRAY_TASK_ID; j=$((i / 2)); SEED=$((i % 2 + 2))
F=${FAMS[$j]}; NAME=L96B_${F}_monaiM_ep1200_seed${SEED}; SUB=ens${MEM[$j]}_no${OUTER[$j]}
[ -f experiments/$NAME/checkpoints/stage1_best.ckpt ] || \
  python train.py --config-name experiment/L96B_${F}_monaiM_ep1200 hydra.run.dir=. hydra.output_subdir=null \
    training.seed=${SEED} ++data.train_cache=$R/experiments/l96_trainval_cache/l96b_trainval_n1000_v100.pt
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="experiments/${NAME}/${SUB}";
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/${SUB}"; fi
  [ -f "$OUT/neural_eval.json" ] && continue; mkdir -p "$OUT"
  python eval_neural_l96.py --checkpoint "experiments/${NAME}/checkpoints/stage1_best.ckpt" --dataset "$DS" \
    --num-windows 200 --n-members ${MEM[$j]} --n-outer ${OUTER[$j]} --batch-size 25 --members-file scores --output "$OUT/neural_eval.json"
done
echo "Done: ${NAME} at $(date)"
