#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-l2b-cfm-norm
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXP=L2b_vanilla_cfm_s0s1_norm

echo "============================================"
echo " L96 Neural Training (VanillaCFM tau=0, plain UNet1D, data.normalize=true) | $(date)"
echo "  Experiment: ${EXP} -- step 2 of the L96 normalization benchmark refresh"
echo "  (plain-UNet1D comparison point alongside L2b_monai_vanilla_cfm_s0s1_norm)"
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
echo " Running full neural_eval.json evaluation..."
python eval_neural_l96.py \
  --checkpoint experiments/${EXP}/checkpoints/stage1_best.ckpt \
  --config experiments/${EXP}/resolved_config.yaml \
  --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
  --num-windows 200 \
  --normalize-stats experiments/l96_norm_stats_obsj2.pt \
  --output experiments/${EXP}/neural_eval.json

echo ""
echo " Done: ${EXP} at $(date)"
echo "============================================"
