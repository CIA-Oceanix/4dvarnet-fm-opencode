#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-joint-unet
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/l96_joint_unet_train_%j_%a.out
#SBATCH --error=sbatch_logs/l96_joint_unet_train_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-joint-neural

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# L10 = JointCFMCoupled (UNet param-flow, multi-tau coupled ODE)
# L12 = JointDirectUNet with a UNet param head
EXPERIMENTS=(
  L10_joint_cfm_coupled_multitau
  L12_joint_direct_unet_unethead
)
EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo " L96 Joint Neural UNet-head Training | $(date)"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP}"
echo "  L10: coupled JointCFM ODE (both x_tau,theta_tau condition both velocity fields)"
echo "  L12: JointDirectUNet with UNet param head"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

rm -rf "experiments/${EXP}"

python train.py \
  --config-name experiment/${EXP} \
  ++data.test_cache=experiments/l96_datasets_obsj2_int100_nwin200.pt \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo " Done: ${EXP} at $(date)"
echo "============================================"
