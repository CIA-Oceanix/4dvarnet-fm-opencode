#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1-monai-obsdensity
#SBATCH --output=sbatch_logs/l96_fdv1_monai_obsdensity_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1_monai_obsdensity_train_%j.err

set -euo pipefail

# Deliberately run from THIS worktree, not 4dvarnet-fm-fdv-monai -- that
# worktree has an active training job (FDV2_grad_state_monai_l96, job
# 52926) and per this session's standing rule it is never touched, even
# just to `cd` into for an unrelated launch.
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

# Reuse the canonical 200-window S0/S1 test splits instead of regenerating
# them (slow per-window path). Absolute path to the master worktree cache
# (shared filesystem; every worktree sees it).
TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV1_unrolled_monai_unet_l96_obsdensity"
echo "============================================"
echo " L96 FDV1(monai): unrolled 4DVarNet-style solver, fast-Y obs-density-augmented training | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  As FDV1_unrolled_monai_unet_l96 but data.obs_density_augment=true"
echo "  (40% full density, else keep_k~U{0..15} of the 16 fast-Y channels,"
echo "  redrawn every batch) -- same recipe that fixed DirectUNet-M's"
echo "  reduced-density generalization gap (PR #182/#188). Purpose: test"
echo "  whether warm-starting SDA3 from this augmented FDV1 mean beats the"
echo "  current best full-density scheme (FDV1+SDA3-monai, non-augmented,"
echo "  RMSE 0.379 S0) and/or improves worst-case reduced-density RMSE the"
echo "  way DirectUNet(aug)+SDA3 did over plain DirectUNet+SDA3."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ++data.test_cache=${TEST_CACHE}

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
