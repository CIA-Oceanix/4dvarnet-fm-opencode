#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=sl-mee-br-206
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-p1_du_l
#SBATCH --output=sbatch_logs/l96_p1_du_l_train_%j.out
#SBATCH --error=sbatch_logs/l96_p1_du_l_train_%j.err
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
EXPERIMENT="P1_directunet_monaiL_noaug_l96"
echo "===== ${EXPERIMENT} | $(date) ====="
python train.py --config-name experiment/${EXPERIMENT} hydra.run.dir=. hydra.output_subdir=null
echo "Done: ${EXPERIMENT} at $(date)"
