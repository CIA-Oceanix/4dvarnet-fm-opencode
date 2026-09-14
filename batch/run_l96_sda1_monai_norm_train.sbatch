#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-sda1-monai-norm
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXP=SDA1_monai_prior_l96_norm

echo "============================================"
echo " L96 Neural Training (MonaiUnconditionalPriorCFM / SDA1, M tier, data.normalize=true) | $(date)"
echo "  Experiment: ${EXP} -- step 2 of the L96 normalization benchmark refresh"
echo "  Training only -- eval_sda_l96.py does not yet support --normalize-stats"
echo "  (needed: obs/state normalization AND rescaling the guidance cost's"
echo "  r_var to normalized units) -- eval to be run as its own follow-up"
echo "  once that's fixed."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""


python train.py \
  --config-name experiment/${EXP} \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo " Training done: ${EXP} at $(date)"
echo "============================================"

echo ""
echo " Verifying checkpoint/config persistence..."
ls -la "experiments/${EXP}/resolved_config.yaml" "experiments/${EXP}/checkpoints/"
