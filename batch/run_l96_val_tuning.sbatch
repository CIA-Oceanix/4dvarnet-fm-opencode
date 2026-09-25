#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-13%4
#SBATCH --job-name=l96-val-tune
#SBATCH --output=sbatch_logs/l96_val_tuning_%A_%a.out
#SBATCH --error=sbatch_logs/l96_val_tuning_%A_%a.err
# Evaluation-time settings tuned on the 50 VALIDATION windows (disjoint from
# test), regular and random-layout versions; scores only.
#   tasks 0-5  : SDA guidance weight {10,15,20,25,30,40}, SDA1-M / SDA2-M / SDA3-M
#   tasks 6-9  : DirectUNet-M(benchmark) -> SDA hybrid, tau0 {0.3,0.5,0.7} x gw {2,5,10,20}, SDA1-M / SDA2-M
#   tasks 10-13: flow calibration, VanillaCFM-M / PredictStateCFM-M (benchmark, seed 1):
#                N_outer {10,20,50} at the trained sigma_prior, sigma_prior {0.75,1.0} at N_outer 10
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
F=$R/4dvarnet-fm-fdv-tau-aware/experiments
B=$R/4dvarnet-fm-bench-default/experiments
H=$R/4dvarnet-fm-sda-nanfix/experiments
NORM=experiments/l96_norm_stats_obsj2.pt
i=$SLURM_ARRAY_TASK_ID
REGIMES=(regular rlayout)
declare -A VAL=([regular]=$R/experiments/l96_valset_regular_w50.pt [rlayout]=$R/experiments/l96_valset_rlayout_n10-100_k4-16_w50.pt)
run() { [ -f "$1/$2" ] && return 0; mkdir -p "$1"; shift 2; "$@"; }
if [ $i -le 5 ]; then
  SDAS=(B4_sda1_monaiM_l96 A3_sda2_monaiM_l96 A3_sda3_monaiM_l96)
  EXP=${SDAS[$((i / 2))]}; RG=${REGIMES[$((i % 2))]}
  for GW in 10 15 20 25 30 40; do
    OUT=experiments/eval_val/$RG/sda_gw/$EXP/gw$GW
    run "$OUT" sda_eval.json python eval_sda_l96.py --checkpoint $F/$EXP/checkpoints/stage1_best.ckpt \
      --config $F/$EXP/resolved_config.yaml --dataset ${VAL[$RG]} --num-windows 50 --batch-size 25 \
      --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight $GW --normalize-stats $NORM \
      --members-file scores --output $OUT/sda_eval.json
  done
elif [ $i -le 9 ]; then
  SDAS=(B4_sda1_monaiM_l96 A3_sda2_monaiM_l96)
  j=$((i - 6)); EXP=${SDAS[$((j / 2))]}; RG=${REGIMES[$((j % 2))]}
  DU=$B/L96B_directunet_monaiM_seed1
  for T0 in 0.3 0.5 0.7; do for GW in 2 5 10 20; do
    OUT=experiments/eval_val/$RG/hybrid/$EXP/tau${T0}_gw$GW
    run "$OUT" hybrid_eval.json python eval_sda_directunet_hybrid_l96.py \
      --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
      --sda-checkpoint $F/$EXP/checkpoints/stage1_best.ckpt --sda-config $F/$EXP/resolved_config.yaml \
      --dataset ${VAL[$RG]} --num-windows 50 --batch-size 25 --n-members 30 --n-outer 10 \
      --tau0 $T0 --guidance-weight $GW --normalize-stats $NORM --members-file scores --output $OUT/hybrid_eval.json
  done; done
else
  FLOWS=($B/L96B_vanillacfm_monaiM_seed1 $H/L96B_predictstatecfm_monaiM_seed1)
  j=$((i - 10)); EXP=${FLOWS[$((j / 2))]}; RG=${REGIMES[$((j % 2))]}
  for SETTING in "10 none" "20 none" "50 none" "10 0.75" "10 1.0"; do
    read -r NO SIG <<< "$SETTING"
    EXTRA=(); [ "$SIG" != none ] && EXTRA=(--sigma-prior $SIG)
    OUT=experiments/eval_val/$RG/calib/$(basename $EXP)/no${NO}_sig${SIG}
    run "$OUT" neural_eval.json python eval_neural_l96.py --step-power 1 --checkpoint $EXP/checkpoints/stage1_best.ckpt \
      --dataset ${VAL[$RG]} --num-windows 50 --n-members 30 --n-outer $NO --batch-size 25 \
      "${EXTRA[@]}" --members-file scores --output $OUT/neural_eval.json
  done
fi
echo "Done task $i at $(date)"
