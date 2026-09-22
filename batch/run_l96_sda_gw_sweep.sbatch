#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=sl-mee-br-205
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-sda_gw_sweep
#SBATCH --output=sbatch_logs/l96_sda_gw_sweep_%j.out
#SBATCH --error=sbatch_logs/l96_sda_gw_sweep_%j.err
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
DATASET="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"
# guidance-weight sweep on the SDA1-M checkpoint. 40 is the value inherited from
# the benchmarked SDA rows and has never been tuned; since conditioning turned out
# worth only ~2% (SDA1 0.5444 / SDA2 0.5342 / SDA3 0.5380), guidance strength is
# the more likely binding constraint on all six SDA numbers. Eval only, no retrain.
EXP=B4_sda1_monaiM_l96
for GW in 10 20 40 80; do
  OUT="experiments/${EXP}/gw_sweep/gw${GW}"
  mkdir -p "$OUT"
  echo "=== ${EXP} | guidance_weight=${GW} | $(date) ==="
  python eval_sda_l96.py \
    --checkpoint "experiments/${EXP}/checkpoints/stage1_best.ckpt" \
    --config "experiments/${EXP}/resolved_config.yaml" \
    --dataset "$DATASET" --num-windows 200 --batch-size 16 \
    --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight ${GW} \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --output "${OUT}/sda_eval.json" || echo "FAILED: gw=${GW}"
done
echo "Done: $(date)"
