#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-obs-density-gen
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# Fast-Y observation-density generalization sweep for the 4 best-of-subcategory
# L96 monai-backbone schemes (DirectUNet-L(cos), CFM-M(flat), SDA3,
# DirectUNet+SDA3 -- see l96_consolidated_benchmark.md), reusing their
# existing checkpoints with NO retraining. Defaults match those checkpoints'
# canonical benchmark hyperparameters exactly (n_members/n_outer/guidance
# weights/tau0), so keep_k=16 in the report should reproduce the consolidated
# benchmark numbers as a sanity check. --n-repeats 3 (independent mask-redraw
# + sampling-noise seeds per (method, case, keep_k) cell) is deliberately
# small (evaluation/obs_density.py::PLAN note: each run already averages
# over ~30 obs-times x 200 windows of independent random draws).
echo "============================================"
echo " L96 obs-density generalization sweep | $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python eval_obs_density_l96.py \
  --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
  --normalize-stats experiments/l96_norm_stats_obsj2.pt \
  --num-windows 200 \
  --batch-size 16 \
  --keep-k 16 8 4 0 \
  --n-repeats 3 \
  --output experiments/l96_obs_density_generalization/results.json

echo ""
echo " Sweep done at $(date) -- building report..."
python reports/l96/generate_l96_obs_density_generalization_report.py \
  --json experiments/l96_obs_density_generalization/results.json \
  --output reports/l96/outputs/l96_obs_density_generalization.md

echo ""
echo "All done at $(date)"
echo "============================================"
