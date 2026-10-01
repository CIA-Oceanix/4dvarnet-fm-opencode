#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-6%4
#SBATCH --exclude=sl-mee-br-[212-213]
#SBATCH --job-name=l96-sda1200-fact
#SBATCH --output=sbatch_logs/l96_sda1200_fact_%A_%a.out
#SBATCH --error=sbatch_logs/l96_sda1200_fact_%A_%a.err
# SDA1-M at 1200 epochs (seed 1, gw 25) on the #243 factorial cells (tasks 0-5, one n_obs
# per task, k {4,8,12,16}) and the out-of-range probes (task 6), on exactly the obs the DA
# runs assimilated; same protocol as run_l96_factorial_eval.sbatch / run_l96_ood_learned.sbatch
# (60 windows, scores only). Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
NAME=B4_sda1_monaiM_ep1200_l96_seed1; SDA=experiments/$NAME
i=$SLURM_ARRAY_TASK_ID
NS=(10 20 30 50 75 100)
if [ $i -le 5 ]; then
  CELLS=(); for k in 4 8 12 16; do CELLS+=("factorial rlayout_n${NS[$i]}-${NS[$i]}_k${k}-${k}_w20_d3"); done
else
  CELLS=(); for T in rlayout_n6-6_k16-16_w20_d3 rlayout_n30-30_k0-0_w20_d3 rlayout_n30-30_k2-2_w20_d3 rlayout_n300-300_k16-16_w20_d3 \
                     rlayout_n1000-1000_k16-16_w20_d3 rlayout_n30-30_k16-16_w20_d3_r0.25 rlayout_n30-30_k16-16_w20_d3_r1; do CELLS+=("ood $T"); done
fi
echo "task $i | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
for C in "${CELLS[@]}"; do
  read -r KIND T <<< "$C"
  OUT=experiments/eval_$KIND/$T/$NAME/ens30_gw25
  [ -f $OUT/sda_eval.json ] && continue; mkdir -p $OUT
  python eval_sda_l96.py --checkpoint $SDA/checkpoints/stage1_best.ckpt --config $SDA/resolved_config.yaml \
    --dataset $R/experiments/l96_testsets_$KIND/l96_testset_$T.pt --num-windows 60 --batch-size 16 \
    --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight 25 --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --members-file scores --output $OUT/sda_eval.json
done
echo "Done task $i at $(date)"
