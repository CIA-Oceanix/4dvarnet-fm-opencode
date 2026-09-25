#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%3
#SBATCH --job-name=l96-hybrid-test
#SBATCH --output=sbatch_logs/l96_hybrid_test_%A_%a.out
#SBATCH --error=sbatch_logs/l96_hybrid_test_%A_%a.err
# DirectUNet-M (benchmark default, seeds 1-3) -> SDA hybrid at the
# validation-selected setting (tau0 0.1, guidance weight 2), SDA2-M and SDA1-M
# priors, on both test sets; scores only.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
F=$R/4dvarnet-fm-fdv-tau-aware/experiments
B=$R/4dvarnet-fm-bench-default/experiments
SDAS=(A3_sda2_monaiM_l96 B4_sda1_monaiM_l96)
i=$SLURM_ARRAY_TASK_ID; E=${SDAS[$((i / 3))]}; S=$((i % 3 + 1))
DU=$B/L96B_directunet_monaiM_seed$S; NAME=hybrid_DU${S}_${E}
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT=experiments/eval_regular/$NAME/tau0.1_gw2;
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT=experiments/eval_rlayout_n10-100_k4-16_w200/$NAME/tau0.1_gw2; fi
  [ -f $OUT/hybrid_eval.json ] && continue; mkdir -p $OUT
  python eval_sda_directunet_hybrid_l96.py \
    --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
    --sda-checkpoint $F/$E/checkpoints/stage1_best.ckpt --sda-config $F/$E/resolved_config.yaml \
    --dataset $DS --num-windows 200 --batch-size 25 --n-members 30 --n-outer 10 \
    --tau0 0.1 --guidance-weight 2 --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --members-file scores --output $OUT/hybrid_eval.json
done
echo "Done $NAME at $(date)"
