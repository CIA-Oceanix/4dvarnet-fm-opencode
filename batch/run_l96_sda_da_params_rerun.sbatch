#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-17%3
#SBATCH --job-name=l96-sda-daparams
#SBATCH --output=sbatch_logs/l96_sda_daparams_%A_%a.out
#SBATCH --error=sbatch_logs/l96_sda_daparams_%A_%a.err
# Re-run every params-conditioned SDA evaluation (SDA2 / SDA3, alone and in the
# DirectUNet hybrid) after the eval collate fix: S1 windows now condition on the
# biased DA params (*_da) the DA baselines and training use, not the true ones.
# Same commands/seeds as the original runs, so S0 must reproduce exactly; the
# old outputs were moved to <dir>_trueparams.
#   0-2  : SDA2-M seeds 1-3, gw 25, both test sets (scores)
#   3-5  : SDA3-fix-M seeds 1-3, gw 25, both test sets (scores)
#   6-8  : hybrid DU-M(400, seeds 1-3) -> SDA2-M, tau0 0.1 gw 2, both test sets
#   9-11 : hybrid DU-M(1200, seeds 1-3) -> SDA2-M, same
#   12-14: NEW hybrid DU-M(1200, seeds 1-3) -> SDA3-fix-M (seed 1), same
#   15-16: P1 SDA2-M / SDA3-M at gw 20, regular + canonical (full members: P1 and
#          benchmark-default reports read them)
#   17   : validation re-check: SDA gw sweep (SDA2-M, SDA3-fix-M) and the hybrid grids
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
F=$R/4dvarnet-fm-fdv-tau-aware/experiments
B=$R/4dvarnet-fm-bench-default/experiments
NORM=experiments/l96_norm_stats_obsj2.pt
REG=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt
CAN=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt
declare -A VAL=([regular]=$R/experiments/l96_valset_regular_w50.pt [rlayout]=$R/experiments/l96_valset_rlayout_n10-100_k4-16_w50.pt)
SDA2=$F/A3_sda2_monaiM_l96
SDA3F=experiments/A3_sda3fix_monaiM_l96_seed1
run() { [ -f "$1/$2" ] && return 0; mkdir -p "$1"; shift 2; "$@"; }
sda_eval() {  # exp_dir gw dataset out num_windows batch members_file
  run "$4" sda_eval.json python eval_sda_l96.py --checkpoint "$1/checkpoints/stage1_best.ckpt" --config "$1/resolved_config.yaml" \
    --dataset "$3" --num-windows "$5" --batch-size "$6" --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight "$2" \
    --normalize-stats $NORM --members-file "$7" --output "$4/sda_eval.json"
}
hyb() {  # du_dir sda_dir dataset out tau0 gw nwin
  run "$4" hybrid_eval.json python eval_sda_directunet_hybrid_l96.py \
    --mean-checkpoint $1/checkpoints/stage1_best.ckpt --mean-config $1/resolved_config.yaml \
    --sda-checkpoint $2/checkpoints/stage1_best.ckpt --sda-config $2/resolved_config.yaml \
    --dataset $3 --num-windows $7 --batch-size 25 --n-members 30 --n-outer 10 --tau0 $5 --guidance-weight $6 \
    --normalize-stats $NORM --members-file scores --output $4/hybrid_eval.json
}
hyb_tests() {  # du_dir sda_dir name
  hyb $1 $2 $REG experiments/eval_regular/$3/tau0.1_gw2 0.1 2 200
  hyb $1 $2 $CAN experiments/eval_rlayout_n10-100_k4-16_w200/$3/tau0.1_gw2 0.1 2 200
}
i=$SLURM_ARRAY_TASK_ID
if [ $i -le 5 ]; then
  S=$((i % 3 + 1))
  if [ $i -le 2 ]; then NAME=A3_sda2_monaiM_l96_seed$S; DIR=experiments/$NAME; [ $S = 1 ] && DIR=$SDA2
  else NAME=A3_sda3fix_monaiM_l96_seed$S; DIR=experiments/$NAME; fi
  sda_eval $DIR 25 $REG experiments/eval_regular/$NAME/ens30_gw25 200 16 scores
  sda_eval $DIR 25 $CAN experiments/eval_rlayout_n10-100_k4-16_w200/$NAME/ens30_gw25 200 16 scores
elif [ $i -le 8 ]; then
  S=$((i - 5)); hyb_tests $B/L96B_directunet_monaiM_seed$S $SDA2 hybrid_DU${S}_A3_sda2_monaiM_l96
elif [ $i -le 11 ]; then
  S=$((i - 8)); hyb_tests experiments/L96B_directunet_monaiM_ep1200_seed$S $SDA2 hybrid_DU1200s${S}_A3_sda2_monaiM_l96
elif [ $i -le 14 ]; then
  S=$((i - 11)); hyb_tests experiments/L96B_directunet_monaiM_ep1200_seed$S $SDA3F hybrid_DU1200s${S}_A3_sda3fix_monaiM_l96
elif [ $i -le 16 ]; then
  export FDV_KEEP_MEMBERS=1  # the P1 and benchmark-default reports read members from /Odyssey
  EXP=$( [ $i = 15 ] && echo A3_sda2_monaiM_l96 || echo A3_sda3_monaiM_l96 )
  sda_eval $F/$EXP 20 $REG $F/$EXP/ens30_gw20 200 16 full
  sda_eval $F/$EXP 20 $CAN experiments/eval_rlayout_n10-100_k4-16_w200/$EXP/ens30_gw20 200 16 full
else
  DU4=$B/L96B_directunet_monaiM_seed1; DU12=experiments/L96B_directunet_monaiM_ep1200_seed1
  for RG in regular rlayout; do
    for GW in 10 15 20 25 30 40; do
      sda_eval $SDA2 $GW ${VAL[$RG]} experiments/eval_val/$RG/sda_gw/A3_sda2_monaiM_l96/gw$GW 50 25 scores
      sda_eval $SDA3F $GW ${VAL[$RG]} experiments/eval_val/$RG/sda_gw/A3_sda3fix_monaiM_l96_seed1/gw$GW 50 25 scores
    done
    for TG in 0.1:0.5 0.1:1 0.1:2 0.2:0.5 0.2:1 0.2:2 0.3:0.5 0.3:1 0.3:2 0.3:5 0.3:10 0.3:20 \
              0.5:2 0.5:5 0.5:10 0.5:20 0.7:2 0.7:5 0.7:10 0.7:20; do
      T0=${TG%:*}; GW=${TG#*:}
      hyb $DU4 $SDA2 ${VAL[$RG]} experiments/eval_val/$RG/hybrid/A3_sda2_monaiM_l96/tau${T0}_gw$GW $T0 $GW 50
    done
    for T0 in 0.05 0.1 0.2; do for GW in 1 2 5; do
      hyb $DU12 $SDA2 ${VAL[$RG]} experiments/eval_val/$RG/hybrid_du1200/A3_sda2_monaiM_l96/tau${T0}_gw$GW $T0 $GW 50
      hyb $DU12 $SDA3F ${VAL[$RG]} experiments/eval_val/$RG/hybrid_du1200/A3_sda3fix_monaiM_l96_seed1/tau${T0}_gw$GW $T0 $GW 50
    done; done
  done
fi
echo "Done task $i at $(date)"
