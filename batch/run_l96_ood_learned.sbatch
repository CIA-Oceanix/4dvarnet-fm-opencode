#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%3
#SBATCH --job-name=l96-ood-learned
#SBATCH --output=sbatch_logs/l96_ood_learned_%A_%a.out
#SBATCH --error=sbatch_logs/l96_ood_learned_%A_%a.err
# Learned schemes on the out-of-range probes, on exactly the obs the DA probes
# assimilated (scripts/make_l96_canonical_testset.py from those layouts):
# benchmark-default DirectUNet-M / PredictStateCFM-M / VanillaCFM-M (seed 1)
# and SDA1-M (gw 25). Scores only.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
B=$R/4dvarnet-fm-bench-default/experiments
H=$R/4dvarnet-fm-sda-nanfix/experiments
F=$R/4dvarnet-fm-fdv-tau-aware/experiments
TAGS=(rlayout_n6-6_k16-16_w20_d3 rlayout_n30-30_k0-0_w20_d3 rlayout_n30-30_k2-2_w20_d3 rlayout_n300-300_k16-16_w20_d3
      rlayout_n1000-1000_k16-16_w20_d3 rlayout_n30-30_k16-16_w20_d3_r0.25 rlayout_n30-30_k16-16_w20_d3_r1)
i=$SLURM_ARRAY_TASK_ID
for T in "${TAGS[@]}"; do
  DS=$R/experiments/l96_testsets_ood/l96_testset_$T.pt
  case $i in
    0|1|2)
      MODELS=("1 1 $B/L96B_directunet_monaiM_seed1" "30 10 $H/L96B_predictstatecfm_monaiM_seed1" "30 10 $B/L96B_vanillacfm_monaiM_seed1")
      read -r MEM OUTER EXP <<< "${MODELS[$i]}"; NAME=$(basename $EXP)
      OUT=experiments/eval_ood/$T/$NAME/ens${MEM}_no${OUTER}
      [ -f $OUT/neural_eval.json ] && continue; mkdir -p $OUT
      python eval_neural_l96.py --checkpoint $EXP/checkpoints/stage1_best.ckpt --dataset $DS --num-windows 60 \
        --n-members $MEM --n-outer $OUTER --batch-size 20 --members-file scores --output $OUT/neural_eval.json ;;
    3)
      OUT=experiments/eval_ood/$T/B4_sda1_monaiM_l96_seed1/ens30_gw25
      [ -f $OUT/sda_eval.json ] && continue; mkdir -p $OUT
      python eval_sda_l96.py --checkpoint $F/B4_sda1_monaiM_l96/checkpoints/stage1_best.ckpt \
        --config $F/B4_sda1_monaiM_l96/resolved_config.yaml --dataset $DS --num-windows 60 --batch-size 16 \
        --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight 25 --normalize-stats experiments/l96_norm_stats_obsj2.pt \
        --members-file scores --output $OUT/sda_eval.json ;;
  esac
done
echo "Done task $i at $(date)"
