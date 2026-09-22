#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=sl-mee-br-207
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-sda1_guided_eval
#SBATCH --output=sbatch_logs/l96_sda1_guided_eval_%j.out
#SBATCH --error=sbatch_logs/l96_sda1_guided_eval_%j.err
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
DATASET="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"
# SDA1 is an UNCONDITIONAL prior: observations enter only as a guidance cost at
# inference, so it MUST be scored with eval_sda_l96.py (--guidance-weight 40,
# --r-var 0.5, matching the benchmarked SDA rows). eval_neural_l96.py samples the
# prior UNGUIDED and returns climatological spread (~RMSE 1.64), not a DA result.
for EXP in B4_sda1_monaiSplus_l96 B4_sda1_monaiM_l96 B4_sda1_monaiL_l96; do
  OUT="experiments/${EXP}/ens30_no10"
  mkdir -p "$OUT"
  echo "=== ${EXP} | guided ens30 | $(date) ==="
  python eval_sda_l96.py \
    --checkpoint "experiments/${EXP}/checkpoints/stage1_best.ckpt" \
    --config "experiments/${EXP}/resolved_config.yaml" \
    --dataset "$DATASET" --num-windows 200 --batch-size 16 \
    --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight 40 \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --output "${OUT}/sda_eval_ens30.json" || echo "FAILED: ${EXP}"
done
echo "Done: $(date)"
