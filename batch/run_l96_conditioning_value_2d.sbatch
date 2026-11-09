#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%3
#SBATCH --job-name=l96-cond-2d
#SBATCH --output=sbatch_logs/l96_cond_2d_%A_%a.out
#SBATCH --error=sbatch_logs/l96_cond_2d_%A_%a.err
# Value of the window parameters and forcing for the 2D (time x sector) SDA2 / SDA3-fix priors (seed 1),
# on the conditioning test sets of scripts/make_l96_conditioning_variants.py (test_s1 = the S0 windows
# with nominal parameters and/or zero forcing given to the schemes). Benchmark protocol: 30 members,
# 10 guided steps, gw 25; the ensemble variance is stored (--members-file variance). One output
# directory per prior and variant (run_l96_conditioning_value.sbatch wrote every estimates_s1.npz to
# the same path).
#   tasks 0-2: SDA2-2D-M   x (no_theta, no_forcing, neither)
#   tasks 3-5: SDA3-fix-2D-M x (no_theta, no_forcing, neither)
# Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
SHARED=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments
VARS=(no_theta no_forcing neither)
T=$SLURM_ARRAY_TASK_ID
V=${VARS[$((T % 3))]}
NAME=$([ $((T / 3)) = 0 ] && echo L96B_sda2_monai2dM_ep1200_seed1 || echo L96B_sda3fix_monai2dM_ep1200_seed1)
RUN="$SHARED/l96/$NAME"
[ -f "$RUN/checkpoints/stage1_best.ckpt" ] || RUN=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-2d-sda23/experiments/$NAME
OUT="$SHARED/l96_conditioning/results_2d/$NAME/$V"
mkdir -p "$MPLCONFIGDIR" "$OUT"
# The 2D conditioned prior (monai_sda_prior_cond_2d) is defined on feature/l96-2d-sda23; run that code at a
# fixed commit, extracted to node-local scratch (CODE_COMMIT, default below).
CODE_COMMIT=${CODE_COMMIT:-0a9ceca}
CODE="$NODE_TMP/code"; mkdir -p "$CODE"
git -C /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode archive "$CODE_COMMIT" | tar -x -C "$CODE"
cd "$CODE"
echo "task $T: $NAME $V ($RUN) | code $CODE_COMMIT | $(hostname) | $(date)"
python -u eval_sda_l96.py --checkpoint "$RUN/checkpoints/stage1_best.ckpt" --config "$RUN/resolved_config.yaml" \
  --dataset "$SHARED/l96_conditioning/l96_conditioning_${V}.pt" --cases s1 --num-windows 200 --batch-size 16 \
  --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight 25 --normalize-stats "$SHARED/l96_norm_stats_obsj2.pt" \
  --members-file variance --output "$OUT/sda_eval.json"
echo "Done at $(date)"
