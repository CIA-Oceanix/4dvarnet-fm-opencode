#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=l96-monai-cfm-tiers-cosine
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# S+/M/L MonaiVanillaCFM (tau=0) tier sweep, mirroring the DirectUNet S+/M/L cosine
# sweep (job 52644) for consistency -- training.stage1.use_cosine_scheduler=true
# (CosineAnnealingLR, matching FDV2's PR #168 scheduler policy). Only the M tier has
# a flat-LR baseline so far (RMSE 0.482/0.480); S+/L are new capacity points, run here
# for the first time. Single sbatch, one GPU, tiers run strictly sequentially so as
# not to load more than one training job onto the server at a time.
#
# NOTE: uses eval_neural_l96.py (not eval_monai_l96.py, which only knows how to
# construct MonaiDirectUNet directly) -- eval_neural_l96.py's load_model() dispatches
# through train.py's model_factory, which already knows monai_vanilla_cfm, and reads
# the proper Lightning checkpoint (stage1_best.ckpt, not the raw stage1.pt dump).
TIERS="L2b_monai_vanilla_cfm_s0s1_norm_splus_cosine L2b_monai_vanilla_cfm_s0s1_norm_cosine L2b_monai_vanilla_cfm_s0s1_norm_l_cosine"

for EXP in $TIERS; do
  echo "============================================"
  echo " L96 Neural Training (MonaiVanillaCFM, tau=0, cosine scheduler) | $(date)"
  echo "  Experiment: ${EXP}"
  echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
  echo "============================================"
  echo ""

  rm -rf "experiments/${EXP}"

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
  echo " Running full neural_eval.json evaluation..."
  echo " (--batch-size 16, not the 200 default: the L tier's wider bottleneck OOMs"
  echo "  a single-batch attention pass at batch=200, per the DirectUNet-L precedent)"
  python eval_neural_l96.py \
    --checkpoint experiments/${EXP}/checkpoints/stage1_best.ckpt \
    --config experiments/${EXP}/resolved_config.yaml \
    --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
    --num-windows 200 \
    --batch-size 16 \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --output experiments/${EXP}/neural_eval.json

  echo ""
  echo " Done: ${EXP} at $(date)"
  echo "============================================"
  echo ""
done

echo "All three CFM cosine-scheduler tiers complete at $(date)"
