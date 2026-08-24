#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=l96-esfix
#SBATCH --array=0-7
#SBATCH --output=sbatch_logs/l96_esfix_%j_%a.out
#SBATCH --error=sbatch_logs/l96_esfix_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

# Re-run order prioritized: canonical s0c int100 first, then the rest of the
# fw/int100/int200 benchmark lineage; dws50 sweeps last (smallest value).
CACHES=(
  "l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.json"
  "l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int200_fw.json"
  "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100.json"
  "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int200.json"
  "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100_fw.json"
  "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int200_fw.json"
  "l96_baselines_dws50_inf2.0_etkf_inf2.0_obsj2_int200.json"
  "l96_baselines_dws50_inf2.0_etkf_inf2.0_obsj2_int200_fw.json"
)
CACHE=${CACHES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " L96 ES-fix re-run | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID}: ${CACHE}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python rerun_l96_esfix.py --cache "${CACHE}" --device cuda

echo ""
echo " Done: ${CACHE} at $(date)"
