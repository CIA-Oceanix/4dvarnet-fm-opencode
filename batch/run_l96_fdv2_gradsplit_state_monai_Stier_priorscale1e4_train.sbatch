#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-gradsplit-Stier-ps1e4
#SBATCH --output=sbatch_logs/l96_fdv2_gradsplit_state_monai_Stier_priorscale1e4_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_gradsplit_state_monai_Stier_priorscale1e4_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_gradsplit_state_monai_l96_Stier_priorscale1e4"
echo "============================================"
echo " L96 DIAGNOSTIC: gradsplit+state, S-tier, g_prior scaled by 1e-4 (~obs+state-equivalent) | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Same as FDV2_gradsplit_state_monai_l96_Stier (jobs 53440/53447, both"
echo "  plateaued/oscillated ~3.5-4.7 through epoch 10) but"
echo "  gradsplit_prior_scale=1e-4 -- multiplies g_prior AFTER normalization/"
echo "  soft-clipping, so the fed tensor becomes cat([g_obs, ~0, x]),"
echo "  functionally close to obs+state's own cat([x, obs_clean])."
echo "  Isolates whether the persistent plateau traces back to g_prior's own"
echo "  contribution specifically."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ++data.test_cache=${TEST_CACHE} \
  +fresh=true

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
