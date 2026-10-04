#!/bin/bash
set -u
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-p1-structure
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python
S=/tmp/claude-7017/-Odyssey-private-rfablet-Python-4dvarnet-fm-opencode/4c265ce5-5c02-4224-adff-3f557544d889/scratchpad
E=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments
DS=$E/l96_datasets_obsj2_int100_nwin200.pt
NS=$E/l96_norm_stats_obsj2.pt
export PYTHONPATH=$PWD
run() { name=$1; shift; o=$S/pooled/$name; mkdir -p $o; echo "=== $name $(date +%T)";
  $PY reports/l96/probe_pooled_calibration.py "$@" --output $o/eval.json 2>&1 | grep -E "POOLED|Error|error" ; rm -f $o/estimates_*.npz; }
for fam in predictstatecfm vanillacfm; do
  r=$E/l96/L96B_${fam}_monaiM_ep1200_seed1
  run ${fam}1200s1 eval_neural_l96 --checkpoint $r/checkpoints/stage1_best.ckpt --config $r/resolved_config.yaml \
    --dataset $DS --normalize-stats $NS --n-members 30 --n-outer 20 --step-power 0.5 --seed 0 --batch-size 50
done
for n in B4_sda1_monaiM_l96 A3_sda2_monaiM_l96 A3_sda3fix_monaiM_l96_seed1; do
  r=$E/l96/$n
  run $n eval_sda_l96 --checkpoint $r/checkpoints/stage1_best.ckpt --config $r/resolved_config.yaml \
    --dataset $DS --normalize-stats $NS --n-members 30 --n-outer 10 --guidance-weight 25 --r-var 0.5 --seed 0 --batch-size 50
done
m=$E/l96/L96B_directunet_monaiM_ep1200_seed1; r=$E/l96/A3_sda3fix_monaiM_l96_seed1
run hybrid_DU1200s1_sda3fix eval_sda_directunet_hybrid_l96 --mean-checkpoint $m/checkpoints/stage1_best.ckpt --mean-config $m/resolved_config.yaml \
  --sda-checkpoint $r/checkpoints/stage1_best.ckpt --sda-config $r/resolved_config.yaml --dataset $DS --normalize-stats $NS \
  --n-members 30 --n-outer 10 --tau0 0.1 --guidance-weight 2 --r-var 0.5 --seed 0 --batch-size 16
echo "=== done $(date +%T)"
