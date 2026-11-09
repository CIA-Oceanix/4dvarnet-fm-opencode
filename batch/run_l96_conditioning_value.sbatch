#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%3
#SBATCH --job-name=l96-cond-value
#SBATCH --output=sbatch_logs/l96_cond_value_%A_%a.out
#SBATCH --error=sbatch_logs/l96_cond_value_%A_%a.err
# Value of the window parameters and forcing for DA and for the parameter-conditioned SDA priors
# (L96, regular observing system). The caches come from scripts/make_l96_conditioning_variants.py:
# their test_s1 holds the S0 windows with nominal parameters and/or zero forcing given to the schemes.
#   tasks 0-2: ETKF, EnKF, ETKS and Strong-4DVar on the full 40-variable model, S0 inflations
#   tasks 3-8: SDA2 and SDA3 (seed 1, benchmark protocol: 30 members, 10 guided steps, gw 25)
# Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
SHARED=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments
OUT="$SHARED/l96_conditioning/results"
mkdir -p "$MPLCONFIGDIR" sbatch_logs "$OUT"
VARS=(no_theta no_forcing neither)
T=$SLURM_ARRAY_TASK_ID
RANDOMIZE='{"F": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "c1": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "eps": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}}'
echo "task $T | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
if (( T < 3 )); then
  V=${VARS[$T]}
  python -u evaluate_all_l96.py --data-cache "$SHARED/l96_conditioning/l96_conditioning_${V}.pt" --cases s1 --s1-j 4 \
    --da-window-steps 500 --num-test-windows 200 --t-max 3.0 --obs-j 2 --obs-interval 100 --suffix "_cond_${V}" \
    --randomize "$RANDOMIZE" --da-fast-weights --skip-weak \
    --etkf-inflation "s0=1.15,s1=1.15" --enkf-inflation "s0=1.2,s1=1.2" --etks --etks-inflation "s0=1.15,s1=1.15"
else
  V=${VARS[$(( (T - 3) % 3 ))]}
  NAME=$([ $(( (T - 3) / 3 )) = 0 ] && echo A3_sda2_monaiM_ep1200_l96_seed1 || echo A3_sda3fix_monaiM_ep1200_l96_seed1)
  python -u eval_sda_l96.py --checkpoint "$SHARED/l96/$NAME/checkpoints/stage1_best.ckpt" \
    --config "$SHARED/l96/$NAME/resolved_config.yaml" --dataset "$SHARED/l96_conditioning/l96_conditioning_${V}.pt" \
    --cases s1 --num-windows 200 --batch-size 16 --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight 25 \
    --normalize-stats "$SHARED/l96_norm_stats_obsj2.pt" --members-file scores --output "$OUT/${NAME}_${V}.json"
fi
echo "Done at $(date)"
