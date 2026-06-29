#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=4dvarnet-verify
#SBATCH --output=sbatch_logs/verify_%j.out
#SBATCH --error=sbatch_logs/verify_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "=== [1/3] Clearing stale T_max=5.0 cache ==="
rm -f experiments/datasets.pt
echo "  Done."

echo ""
echo "=== [2/3] Unified baseline evaluation (DWS=50, EnKF inf=1.2) ==="
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

export HYDRA_FULL_ERROR=1

python eval_baselines.py \
  --config-name baselines/dws50 \
  baselines.enkf.inflation=1.2 \
  baselines.batch_size=128 \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo "=== [3/3] Generating synthesis PDF ==="

python reports/generate_baseline_report.py \
  --json experiments/baselines_dws50_inf1.2.json \
  --trajs experiments/baselines_trajectories_dws50_inf1.2.npz \
  --output reports/outputs/synthesis_dws50_inf12.pdf

echo ""
echo "  Done at: $(date)"
echo "============================================"
