#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodelist=sl-mee-br-203
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=l96-sda-directunet-sweep
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# S0-only (tau0, guidance_weight) grid sweep for DirectUNet-M-warm-started
# SDA1/SDA2/SDA3 (see sweep_sda_directunet_hybrid_l96.py), mirroring the
# FDV1+SDA hybrids' sweep methodology. Idle node, doesn't conflict with the
# still-running CFM L-tier training (job 52654) or the FDV job (52672).
echo "============================================"
echo " DirectUNet-M + SDA hybrid sweep | $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python sweep_sda_directunet_hybrid_l96.py

echo ""
echo "Sweep complete at $(date)"
