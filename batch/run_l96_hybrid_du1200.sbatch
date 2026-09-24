#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%3
#SBATCH --job-name=l96-hybrid-du1200
#SBATCH --output=sbatch_logs/l96_hybrid_du1200_%A_%a.out
#SBATCH --error=sbatch_logs/l96_hybrid_du1200_%A_%a.err
# Hybrid with the 1200-epoch DirectUNet-M as the mean, SDA2-M prior.
#   tasks 0-2: DU seeds 1-3 at the validation-selected tau0 0.1 / gw 2, both test sets
#   task 3   : validation re-check with DU seed 1: tau0 {0.05,0.1,0.2} x gw {1,2,5}, plus DU alone
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
SDA=$R/4dvarnet-fm-fdv-tau-aware/experiments/A3_sda2_monaiM_l96
NORM=experiments/l96_norm_stats_obsj2.pt
hyb() {  # du_dir dataset out tau0 gw nwin
  [ -f "$3/hybrid_eval.json" ] && return 0; mkdir -p "$3"
  python eval_sda_directunet_hybrid_l96.py --mean-checkpoint $1/checkpoints/stage1_best.ckpt --mean-config $1/resolved_config.yaml \
    --sda-checkpoint $SDA/checkpoints/stage1_best.ckpt --sda-config $SDA/resolved_config.yaml \
    --dataset $2 --num-windows $6 --batch-size 25 --n-members 30 --n-outer 10 --tau0 $4 --guidance-weight $5 \
    --normalize-stats $NORM --members-file scores --output $3/hybrid_eval.json
}
i=$SLURM_ARRAY_TASK_ID
if [ $i -le 2 ]; then
  S=$((i + 1)); DU=experiments/L96B_directunet_monaiM_ep1200_seed$S; NAME=hybrid_DU1200s${S}_A3_sda2_monaiM_l96
  hyb $DU $R/experiments/l96_datasets_obsj2_int100_nwin200.pt experiments/eval_regular/$NAME/tau0.1_gw2 0.1 2 200
  hyb $DU $R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt experiments/eval_rlayout_n10-100_k4-16_w200/$NAME/tau0.1_gw2 0.1 2 200
else
  DU=experiments/L96B_directunet_monaiM_ep1200_seed1
  for RG in regular rlayout; do
    if [ $RG = regular ]; then V=$R/experiments/l96_valset_regular_w50.pt; else V=$R/experiments/l96_valset_rlayout_n10-100_k4-16_w50.pt; fi
    O=experiments/eval_val/$RG/du_alone/L96B_directunet_monaiM_ep1200_seed1
    [ -f $O/neural_eval.json ] || { mkdir -p $O; python eval_neural_l96.py --checkpoint $DU/checkpoints/stage1_best.ckpt --dataset $V \
      --num-windows 50 --n-members 1 --n-outer 1 --batch-size 25 --output $O/neural_eval.json; }
    for T0 in 0.05 0.1 0.2; do for GW in 1 2 5; do
      hyb $DU $V experiments/eval_val/$RG/hybrid_du1200/A3_sda2_monaiM_l96/tau${T0}_gw$GW $T0 $GW 50
    done; done
  done
fi
echo "Done task $i at $(date)"
