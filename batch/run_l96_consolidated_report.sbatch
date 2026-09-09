#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=140G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-consolidated-report
#SBATCH --output=sbatch_logs/l96_report_%j.out
#SBATCH --error=sbatch_logs/l96_report_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# Regenerates l96_consolidated_benchmark.md with the new monai-backbone
# schemes, per-trajectory mean+/-std detail section, and the 7-scheme
# reconstruction figures (now incl. FDV1+SDA1/SDA2). 140G requested -- this
# loads ~11 members_*.npz files (~1.6GB each, W=200 x T=3000 x D=24 x M=30)
# simultaneously plus DA trajectories/truth; the interactive shell's 16G
# cgroup limit OOM-killed it at fewer rows already.
python reports/l96/generate_l96_consolidated_report.py

echo "Report regenerated at $(date)"
