#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%3
#SBATCH --job-name=l96-cfm-blend-test
#SBATCH --output=sbatch_logs/l96_cfm_blend_test_%A_%a.out
#SBATCH --error=sbatch_logs/l96_cfm_blend_test_%A_%a.err
# Blend of two 1200-epoch flows at sampling time (models/cfm_blend.py), benchmark sampler.
#   task 0  : model-averaging controls on the 50 validation windows (regular + random layout):
#             PredictStateCFM s1 + s2 and VanillaCFM s1 + s2 at a constant 50/50 weight
#   tasks 1-9: the 200 test windows (regular + canonical random set), seed pairs s = 1..3:
#             1-3 VanillaCFM s + PredictStateCFM s at SCHED (validation-selected, env var),
#             4-6 PredictStateCFM s + s' const 0.5, 7-9 VanillaCFM s + s' const 0.5 (s' = s mod 3 + 1)
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
VC=experiments/L96B_vanillacfm_monaiM_ep1200_seed; PSC=experiments/L96B_predictstatecfm_monaiM_ep1200_seed
ev() {  # first second schedule dataset nwin out
  [ -f "$6/neural_eval.json" ] && return 0; mkdir -p "$6"
  python eval_neural_l96.py --checkpoint $1/checkpoints/stage1_best.ckpt --blend-with $2/checkpoints/stage1_best.ckpt \
    --blend-schedule $3 --dataset $4 --num-windows $5 --n-members 30 --n-outer 20 --step-power 0.5 --batch-size 25 \
    --members-file scores --output "$6/neural_eval.json"
}
i=$SLURM_ARRAY_TASK_ID
if [ $i = 0 ]; then
  for RG in regular rlayout; do
    if [ $RG = regular ]; then V=$R/experiments/l96_valset_regular_w50.pt; else V=$R/experiments/l96_valset_rlayout_n10-100_k4-16_w50.pt; fi
    ev ${PSC}1 ${PSC}2 const:0.5 $V 50 experiments/eval_val/$RG/cfm_blend/psc1+psc2_const_0.5
    ev ${VC}1 ${VC}2 const:0.5 $V 50 experiments/eval_val/$RG/cfm_blend/vc1+vc2_const_0.5
  done
else
  j=$((i - 1)); S=$((j % 3 + 1)); S2=$((S % 3 + 1)); K=$((j / 3))
  case $K in
    0) A=${VC}$S; B=${PSC}$S; SC=${SCHED:?set SCHED}; NAME=blend_vc${S}+psc${S}_${SC/:/_} ;;
    1) A=${PSC}$S; B=${PSC}$S2; SC=const:0.5; NAME=blend_psc${S}+psc${S2}_const_0.5 ;;
    2) A=${VC}$S; B=${VC}$S2; SC=const:0.5; NAME=blend_vc${S}+vc${S2}_const_0.5 ;;
  esac
  ev $A $B $SC $R/experiments/l96_datasets_obsj2_int100_nwin200.pt 200 experiments/eval_regular/$NAME/ens30_no20
  ev $A $B $SC $R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt 200 experiments/eval_rlayout_n10-100_k4-16_w200/$NAME/ens30_no20
fi
echo "Done task $i at $(date)"
