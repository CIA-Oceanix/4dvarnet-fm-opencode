#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%3
#SBATCH --job-name=l96-sda-r2
#SBATCH --output=sbatch_logs/l96_sda_round2_%A_%a.out
#SBATCH --error=sbatch_logs/l96_sda_round2_%A_%a.err
# SDA round 2, scores only:
#   0-2  : SDA3-fix (noisy DA bias actually active) seeds 1-3: train, then
#          gw 20 (P1 protocol) and gw 25 (validation-tuned) on both test sets
#   3-8  : SDA1-M / SDA2-M seeds 1-3 at gw 25 on both test sets
#   9    : DirectUNet-M -> SDA hybrid, extended validation grid tau0 {0.1,0.2,0.3}
#          x gw {0.5,1,2}, plus DirectUNet-M alone on the validation sets
#   10-11: SDA1-M / SDA2-M (seed 1) at gw 25 on the 24 factorial cells
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
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
run() { [ -f "$1/$2" ] && return 0; mkdir -p "$1"; shift 2; "$@"; }
sda_eval() {  # exp_dir name gw dataset out num_windows
  run "$5" sda_eval.json python eval_sda_l96.py --checkpoint "$1/checkpoints/stage1_best.ckpt" --config "$1/resolved_config.yaml" \
    --dataset "$4" --num-windows "$6" --batch-size 16 --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight "$3" \
    --normalize-stats $NORM --members-file scores --output "$5/sda_eval.json"
}
both_tests() {  # exp_dir name gw
  sda_eval "$1" "$2" "$3" $REG experiments/eval_regular/$2/ens30_gw$3 200
  sda_eval "$1" "$2" "$3" $CAN experiments/eval_rlayout_n10-100_k4-16_w200/$2/ens30_gw$3 200
}
i=$SLURM_ARRAY_TASK_ID
if [ $i -le 2 ]; then
  S=$((i + 1)); NAME=A3_sda3fix_monaiM_l96_seed$S
  [ -f experiments/$NAME/checkpoints/stage1_best.ckpt ] || \
    python train.py --config-name experiment/A3_sda3fix_monaiM_l96 hydra.run.dir=. hydra.output_subdir=null ++training.seed=$S
  for GW in 20 25; do both_tests experiments/$NAME $NAME $GW; done
elif [ $i -le 8 ]; then
  j=$((i - 3)); EXPS=(B4_sda1_monaiM_l96 A3_sda2_monaiM_l96); E=${EXPS[$((j / 3))]}; S=$((j % 3 + 1))
  if [ $S = 1 ]; then DIR=$F/$E; NAME=${E}_seed1; else NAME=${E}_seed$S; DIR=experiments/$NAME; fi
  both_tests $DIR $NAME 25
elif [ $i = 9 ]; then
  DU=$B/L96B_directunet_monaiM_seed1
  for RG in regular rlayout; do
    run experiments/eval_val/$RG/du_alone/L96B_directunet_monaiM_seed1 neural_eval.json python eval_neural_l96.py \
      --checkpoint $DU/checkpoints/stage1_best.ckpt --dataset ${VAL[$RG]} --num-windows 50 --n-members 1 --n-outer 1 \
      --batch-size 25 --output experiments/eval_val/$RG/du_alone/L96B_directunet_monaiM_seed1/neural_eval.json
    for E in B4_sda1_monaiM_l96 A3_sda2_monaiM_l96; do for T0 in 0.1 0.2 0.3; do for GW in 0.5 1 2; do
      OUT=experiments/eval_val/$RG/hybrid/$E/tau${T0}_gw$GW
      run $OUT hybrid_eval.json python eval_sda_directunet_hybrid_l96.py \
        --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
        --sda-checkpoint $F/$E/checkpoints/stage1_best.ckpt --sda-config $F/$E/resolved_config.yaml \
        --dataset ${VAL[$RG]} --num-windows 50 --batch-size 25 --n-members 30 --n-outer 10 \
        --tau0 $T0 --guidance-weight $GW --normalize-stats $NORM --members-file scores --output $OUT/hybrid_eval.json
    done; done; done
  done
else
  EXPS=(B4_sda1_monaiM_l96 A3_sda2_monaiM_l96); E=${EXPS[$((i - 10))]}
  for n in 10 20 30 50 75 100; do for k in 4 8 12 16; do
    CELL=rlayout_n${n}-${n}_k${k}-${k}_w20_d3
    sda_eval $F/$E ${E}_seed1 25 $R/experiments/l96_testsets_factorial/l96_testset_$CELL.pt \
      experiments/eval_factorial/$CELL/${E}_seed1/ens30_gw25 60
  done; done
fi
echo "Done task $i at $(date)"
