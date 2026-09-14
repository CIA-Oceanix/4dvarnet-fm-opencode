#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-grad-softclip
#SBATCH --output=sbatch_logs/l96_fdv2_grad_state_monai_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_grad_state_monai_train_%j.err

set -euo pipefail

# Deliberately run from THIS worktree, not 4dvarnet-fm-fdv-monai -- job 52926
# (the original grad+state run) lives there and per this session's standing
# rule it is never touched. This is a fresh, independent training in this
# worktree's own experiments/, no checkpoint here to conflict with.
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_grad_state_monai_l96"
echo "============================================"
echo " L96 FDV2(monai) grad+state: tanh soft-clip + corrected aux prior loss | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  THIRD retrain of this config. History:"
echo "  1. job 52926 (original recipe): RMSE 0.804, fast-Y variance ratio"
echo "     ~0.61, prior_weight collapsed to ~0.0 during training."
echo "  2. job 53104 (soft-clip fix only): _normalize_channels' post-norm"
echo "     bound on the grad term became a smooth tanh soft-clip"
echo "     (clip_range*tanh(t/clip_range), grad_clip_range=5.0) instead of a"
echo "     hard torch.clamp. Did NOT fix the collapse: prior_weight again"
echo "     collapsed 0.97->0.0 by epoch ~120, train_loss degrading past that"
echo "     point (1.23->1.54). Killed once the root cause below was found."
echo "  3. THIS run: FourDVarNetSolver.compute_loss's auxiliary prior term"
echo "     was prior_weight*prior_cost(z)+obs_cost(z) for z in {x_final,"
echo "     states} -- the full per-iteration var_cost formula, outer-loop-"
echo "     side. A trainable prior_weight could shrink this term to 0 for"
echo "     free (the obs_cost half doesn't depend on it), with no actual"
echo "     improvement in prior quality required -- exactly what job 53104"
echo "     did. Fixed to a pure prior_cost(x_final)+prior_cost(states) term,"
echo "     no prior_weight multiplier, no obs_cost. prior_weight keeps its"
echo "     only legitimate role inside the per-iteration solver update."
echo "  +fresh=true: job 53104's killed, now-stale checkpoint is archived,"
echo "  not resumed."
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
