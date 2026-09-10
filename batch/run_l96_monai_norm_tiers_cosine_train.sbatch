#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-monai-norm-tiers-cosine
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# S+/M/L MonaiDirectUNet tier sweep, re-run with training.stage1.use_cosine_scheduler=true
# (CosineAnnealingLR, matching FDV2's PR #168 scheduler policy) for consistency with the
# flat-LR sweep (S+ 0.532/0.531, M 0.501/0.502, L 0.765-0.891/0.767-0.895). Single sbatch,
# one GPU, tiers run strictly sequentially so as not to load more than one training job onto
# the server at a time.
TIERS="L1b_monai_unet_s0s1_norm_splus_cosine L1b_monai_unet_s0s1_norm_cosine L1b_monai_unet_s0s1_norm_l_cosine"

for EXP in $TIERS; do
  echo "============================================"
  echo " L96 Neural Training (MonaiDirectUNet, cosine scheduler) | $(date)"
  echo "  Experiment: ${EXP}"
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

  echo ""
  echo " Running full neural_eval.json evaluation (matches L1b's report schema)..."
  echo " (--batch-size 16, not the 200 default: the L tier's wider bottleneck OOMs"
  echo "  a single-batch attention pass at batch=200 -- see L1b_monai_unet_s0s1_norm_l_cosine.yaml)"
  python eval_monai_l96.py \
    --checkpoint experiments/${EXP}/checkpoints/stage1.pt \
    --config experiments/${EXP}/resolved_config.yaml \
    --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
    --num-windows 200 \
    --batch-size 16 \
    --output experiments/${EXP}/neural_eval.json

  echo ""
  echo " Done: ${EXP} at $(date)"
  echo "============================================"
  echo ""
done

echo "All three cosine-scheduler tiers complete at $(date)"
