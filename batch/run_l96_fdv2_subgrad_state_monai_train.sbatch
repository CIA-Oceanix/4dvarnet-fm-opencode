#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-subgrad-monai
#SBATCH --output=sbatch_logs/l96_fdv2_subgrad_state_monai_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_subgrad_state_monai_train_%j.err

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

EXPERIMENT="FDV2_subgrad_state_monai_l96"
echo "============================================"
echo " L96 FDV2(monai): subgrad+state (two-residual proxy gradient), MonaiUNet1D backbone | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  This job was previously launched as 53077 under the normalization fix"
echo "  alone (subgrad+state's g_obs/g_prior no longer run through"
echo "  _normalize_channels -- diluted 10x by the sparse 1%-density obs mask,"
echo "  confirmed as the cause of the original run's fast-Y collapse, variance"
echo "  ratio ~0.6 vs ~0.9+ healthy) and was killed pending the priorS+tbptt"
echo "  sibling's outcome. THIS retrain adds a second, independent fix:"
echo "  FourDVarNetSolver.compute_loss's auxiliary prior term included"
echo "  _masked_obs_cost as well as prior_cost (the full var_cost formula,"
echo "  outer-loop-side) even though subgrad+state never trains prior_weight"
echo "  itself -- that redundant obs_cost term is now removed, aux term is"
echo "  pure prior_cost(x_final)+prior_cost(states). The priorS+tbptt"
echo "  sibling (job 53078, completed under the OLD loss) evaluated at fast-Y"
echo "  variance ratio 0.53 -- still collapsed, and confounded by this loss"
echo "  bug, so not clean evidence the normalization fix alone works. This"
echo "  equal-capacity run isolates that question. +fresh=true: the old"
echo "  (differently-trained, now-stale) checkpoint is archived, not resumed."
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
