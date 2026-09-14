#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-subgrad-priorS-tbptt
#SBATCH --output=sbatch_logs/l96_fdv2_subgrad_state_monai_priorS_tbptt_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_subgrad_state_monai_priorS_tbptt_train_%j.err

set -euo pipefail

# Deliberately run from THIS worktree, not 4dvarnet-fm-fdv-monai -- that
# worktree has an active training job and per this session's standing rule
# it is never touched, even just to `cd` into for an unrelated launch.
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_subgrad_state_monai_l96_priorS_tbptt"
echo "============================================"
echo " L96 FDV2(monai) subgrad+state: solver-M + prior-S+ architecture, 2x5 truncated-BPTT | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Two changes vs FDV2_subgrad_state_monai_l96 (job 53011), tested together:"
echo "  1. prior_hidden_channels=[32,64,128] (S+ tier) while the solver unet"
echo "     stays M-tier [64,128,256] -- matches ronan_devs' own convention"
echo "     (solver 2x the prior's width), unlike this codebase's prior"
echo "     default of equal capacity."
echo "  2. tbptt_n_blocks=2, tbptt_block_size=5: the N_outer=10 unroll is"
echo "     detached at iteration 5, training loss = mean of MSE at iteration"
echo "     4 and iteration 9 (deep supervision + smaller backward graph)."
echo "     val_loss is UNCHANGED -- always the final-iteration answer only,"
echo "     never the block average (see FourDVarNetSolver.compute_loss)."
echo "  SECOND retrain under the normalization fix: subgrad+state's g_obs/g_prior"
echo "  are no longer run through _normalize_channels (same root cause/fix"
echo "  as the equal-capacity job's retrain -- see that sbatch script)."
echo "  This job previously ran as 53078 and completed 400 epochs cleanly,"
echo "  but evaluated at fast-Y variance ratio 0.53 / correlation 0.72 --"
echo "  still in the pre-fix collapsed regime. That run was confounded by a"
echo "  second bug since fixed: compute_loss's auxiliary prior term included"
echo "  _masked_obs_cost (the full var_cost formula) instead of a pure"
echo "  prior_cost, which changes the effective training objective for every"
echo "  aux_var_cost_weight>0 config including this one, independent of"
echo "  prior_weight (fixed at 1.0 here, never trained for subgrad+state)."
echo "  THIS retrain has both fixes applied. +fresh=true: the old (now-stale)"
echo "  checkpoint is archived, not resumed."
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
