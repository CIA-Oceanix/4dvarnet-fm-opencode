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
echo " L96 FDV2(monai) grad+state: tanh soft-clip on the grad term, grad_clip_range=5.0 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  RETRAIN vs the original job 52926 recipe (RMSE 0.804, fast-Y variance"
echo "  ratio ~0.61 -- measured directly on that checkpoint, along with"
echo "  prior_weight having collapsed to ~0.0 during training). Two changes:"
echo "  1. _normalize_channels' post-normalization bound on the combined"
echo "     autograd grad tensor is now a smooth tanh soft-clip"
echo "     (clip_range*tanh(t/clip_range)), not a hard torch.clamp -- a"
echo "     saturated element keeps a small gradient pulling it back in,"
echo "     instead of the exactly-zero gradient a hard clamp gives there."
echo "     Normalization ITSELF is kept (not removed, unlike subgrad+state's"
echo "     own residual channels) per prior published results (Fablet et"
echo "     al., JAMES) on its importance for this class of solver."
echo "  2. grad_clip_range=5.0 (separate field from clip_range=50.0, which"
echo "     still only bounds the raw state branch via an unchanged hard"
echo "     clamp): with a hard clamp, 50 was a rarely-triggered safety"
echo "     ceiling; with a smooth soft-clip, the nonlinearity should engage"
echo "     near the term's real operating range (RMS-normalized to ~1)."
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
