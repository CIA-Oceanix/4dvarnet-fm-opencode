#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-13
#SBATCH --job-name=l96-canon-eval
#SBATCH --output=sbatch_logs/l96_canonical_rlayout_eval_%A_%a.out
#SBATCH --error=sbatch_logs/l96_canonical_rlayout_eval_%A_%a.err
# Learned schemes on the canonical random-observing-system test set
# (scripts/make_l96_canonical_testset.py: the P1 200 windows with the exact
# obs the DA baselines assimilated in eval_da_random_layout_l96.py, n 10-100,
# k 4-16). Protocol per family: deterministic ens1; flows ens30 x 20 early-fine steps (p = 0.5);
# SDA guided ens30, 10 steps, gw 20, r_var 0.5. Verify afterwards with
# scripts/check_l96_testset_consistency.py.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
TESTSET=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt
B=$R/4dvarnet-fm-bench-default/experiments
F=$R/4dvarnet-fm-fdv-tau-aware/experiments
RUNS=(
  "neural 1 1 $B/L96B_directunet_monaiM_seed1" "neural 1 1 $B/L96B_directunet_monaiM_seed2" "neural 1 1 $B/L96B_directunet_monaiM_seed3"
  "neural 30 20 $B/L96B_vanillacfm_monaiM_seed1" "neural 30 20 $B/L96B_vanillacfm_monaiM_seed2" "neural 30 20 $B/L96B_vanillacfm_monaiM_seed3"
  "neural 1 1 $F/P1_directunet_monaiM_noaug_l96" "neural 30 20 $F/A1_vanillacfm_monaiM_l96" "neural 30 20 $F/A2_predictstatecfm_monaiM_l96"
  "sda 30 10 $F/B4_sda1_monaiSplus_l96" "sda 30 10 $F/B4_sda1_monaiM_l96" "sda 30 10 $F/B4_sda1_monaiL_l96"
  "sda 30 10 $F/A3_sda2_monaiM_l96" "sda 30 10 $F/A3_sda3_monaiM_l96"
)
read -r KIND MEM OUTER EXP <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
NAME=$(basename "$EXP")
echo "===== ${NAME} (${KIND}) | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
if [ "$KIND" = sda ]; then
  OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens30_gw20"; mkdir -p "$OUT"
  python eval_sda_l96.py --checkpoint "$EXP/checkpoints/stage1_best.ckpt" --config "$EXP/resolved_config.yaml" \
    --dataset "$TESTSET" --num-windows 200 --batch-size 16 --n-members 30 --n-outer 10 --r-var 0.5 \
    --guidance-weight 20 --normalize-stats experiments/l96_norm_stats_obsj2.pt --output "$OUT/sda_eval.json"
else
  OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens${MEM}_no${OUTER}"; mkdir -p "$OUT"
  python eval_neural_l96.py --checkpoint "$EXP/checkpoints/stage1_best.ckpt" --dataset "$TESTSET" \
    --num-windows 200 --n-members "$MEM" --n-outer "$OUTER" --batch-size 25 --output "$OUT/neural_eval.json"
fi
echo "Done at $(date)"
