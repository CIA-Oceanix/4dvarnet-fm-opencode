#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --job-name=l96-cfm-ens3
#SBATCH --output=sbatch_logs/l96_cfm_ens3_%A_%a.out
#SBATCH --error=sbatch_logs/l96_cfm_ens3_%A_%a.err
# Equal-weight velocity ensembles of the 1200-epoch flows (models/cfm_blend.MeanVelocityFlow),
# benchmark sampler, 200 test windows, regular + canonical random set:
#   task 0: PredictStateCFM-M seeds 1-3; task 1: VanillaCFM-M seeds 1-3; task 2: all six.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
c() { echo experiments/L96B_$1_monaiM_ep1200_seed$2/checkpoints/stage1_best.ckpt; }
case $SLURM_ARRAY_TASK_ID in
  0) FIRST=$(c predictstatecfm 1); REST="$(c predictstatecfm 2) $(c predictstatecfm 3)"; NAME=ens3_psc123 ;;
  1) FIRST=$(c vanillacfm 1); REST="$(c vanillacfm 2) $(c vanillacfm 3)"; NAME=ens3_vc123 ;;
  2) FIRST=$(c predictstatecfm 1); REST="$(c predictstatecfm 2) $(c predictstatecfm 3) $(c vanillacfm 1) $(c vanillacfm 2) $(c vanillacfm 3)"; NAME=ens6_psc123_vc123 ;;
esac
for TS in regular canonical; do
  if [ $TS = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT=experiments/eval_regular/$NAME/ens30_no20
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT=experiments/eval_rlayout_n10-100_k4-16_w200/$NAME/ens30_no20; fi
  [ -f $OUT/neural_eval.json ] && continue; mkdir -p $OUT
  python eval_neural_l96.py --checkpoint $FIRST --blend-with $REST --dataset $DS --num-windows 200 \
    --n-members 30 --n-outer 20 --step-power 0.5 --batch-size 25 --members-file scores --output $OUT/neural_eval.json
done
echo "Done $NAME at $(date)"
