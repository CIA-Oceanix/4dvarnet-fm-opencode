#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-monai-norm-l-eval
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXP=L1b_monai_unet_s0s1_norm_l

echo "============================================"
echo " Eval-only rerun (smaller batch size to avoid the attention-block OOM"
echo " seen at --batch-size 200 for the L tier's wider [128,256,512]"
echo " bottleneck: 26.8GiB attention matrix vs 44.4GiB GPU capacity)."
echo " Training checkpoint already exists -- not retraining."
echo "============================================"

python eval_monai_l96.py \
  --checkpoint experiments/${EXP}/checkpoints/stage1.pt \
  --config experiments/${EXP}/resolved_config.yaml \
  --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
  --num-windows 200 \
  --batch-size 20 \
  --output experiments/${EXP}/neural_eval.json

echo ""
echo " Done: ${EXP} eval at $(date)"
echo "============================================"
