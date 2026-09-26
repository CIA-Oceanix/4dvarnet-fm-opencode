#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --job-name=l96-cfm-blend
#SBATCH --output=sbatch_logs/l96_cfm_blend_%A_%a.out
#SBATCH --error=sbatch_logs/l96_cfm_blend_%A_%a.err
# Sampling-time tau-varying blend of VanillaCFM-M and PredictStateCFM-M (1200 epochs, seed 1):
# v = lam(tau) v_PSC + (1 - lam(tau)) v_VC (models/cfm_blend.py), benchmark sampler (30 x 20
# early-fine steps), on the 50 validation windows (task 0: regular, task 1: random layout).
# Plus plain VanillaCFM on the same windows: const:0 must reproduce it exactly.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
VC=experiments/L96B_vanillacfm_monaiM_ep1200_seed1; PSC=experiments/L96B_predictstatecfm_monaiM_ep1200_seed1
RGS=(regular rlayout); RG=${RGS[$SLURM_ARRAY_TASK_ID]}
if [ $RG = regular ]; then V=$R/experiments/l96_valset_regular_w50.pt; else V=$R/experiments/l96_valset_rlayout_n10-100_k4-16_w50.pt; fi
ev() {  # out extra-args...
  [ -f "$1/neural_eval.json" ] && return 0; mkdir -p "$1"; local out=$1; shift
  python eval_neural_l96.py --checkpoint $VC/checkpoints/stage1_best.ckpt --dataset $V --num-windows 50 \
    --n-members 30 --n-outer 20 --step-power 0.5 --batch-size 25 --members-file scores --output "$out/neural_eval.json" "$@"
}
ev experiments/eval_val/$RG/cfm_blend/plain_vc
for S in const:0 const:0.25 const:0.5 const:0.75 const:1 switch:0.2 switch:0.4 switch:0.6 power:0.5 power:1 power:2 rpower:1; do
  ev experiments/eval_val/$RG/cfm_blend/${S/:/_} --blend-with $PSC/checkpoints/stage1_best.ckpt --blend-schedule $S
done
echo "Done $RG at $(date)"
