#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --job-name=l96-ens3-rw
#SBATCH --output=sbatch_logs/l96_ens3_rw_%A_%a.out
#SBATCH --error=sbatch_logs/l96_ens3_rw_%A_%a.err
# PredictStateCFM-M seeds 1-3 (1200 epochs) as a velocity ensemble: equal weights vs 10 random
# tau-varying simplex weight schedules (models/cfm_blend.random_weight_schedule, seeds 0-9), on
# the 50 validation windows (task 0: regular, task 1: random layout). Selection only; the best
# random schedule is then scored on the test sets.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
c() { echo experiments/L96B_predictstatecfm_monaiM_ep1200_seed$1/checkpoints/stage1_best.ckpt; }
RGS=(regular rlayout); RG=${RGS[$SLURM_ARRAY_TASK_ID]}
if [ $RG = regular ]; then V=$R/experiments/l96_valset_regular_w50.pt; else V=$R/experiments/l96_valset_rlayout_n10-100_k4-16_w50.pt; fi
for W in mean random:0 random:1 random:2 random:3 random:4 random:5 random:6 random:7 random:8 random:9; do
  OUT=experiments/eval_val/$RG/cfm_ens3_psc/${W/:/_}
  [ -f $OUT/neural_eval.json ] && continue; mkdir -p $OUT
  python eval_neural_l96.py --checkpoint $(c 1) --blend-with $(c 2) $(c 3) --blend-weights $W --dataset $V --num-windows 50 \
    --n-members 30 --n-outer 20 --step-power 0.5 --batch-size 25 --members-file scores --output $OUT/neural_eval.json
done
echo "Done $RG at $(date)"
