#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-joint-unet-eval
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/l96_joint_unet_eval_%j_%a.out
#SBATCH --error=sbatch_logs/l96_joint_unet_eval_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-joint-neural

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

DATASET="experiments/l96_datasets_obsj2_int100_nwin200.pt"

# experiment|config|extra_flags
# L10 = JointCFMCoupled multi-tau (no tau0 flag); L12 = JointDirectUNet UNet head.
TASKS=(
  "L10_joint_cfm_coupled_multitau|config/experiment/L10_joint_cfm_coupled_multitau.yaml|"
  "L12_joint_direct_unet_unethead|config/experiment/L12_joint_direct_unet_unethead.yaml|"
)
SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}
EXP=${SPEC%%|*}
SPEC=${SPEC#*|}
CFG=${SPEC%%|*}
FLAGS=${SPEC#*|}

echo "============================================"
echo " L96 Joint Neural UNet-head Eval (DA-parity) | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP} / flags: ${FLAGS:-none}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python eval_joint_neural_l96.py \
  --checkpoint "experiments/${EXP}/checkpoints/stage2_best.ckpt" \
  --config "${CFG}" \
  --dataset "${DATASET}" \
  --num-windows 200 \
  --obs-interval 100 \
  --obs-j 2 \
  --device cuda \
  ${FLAGS} \
  --output "experiments/${EXP}/joint_neural_eval.json"

echo ""
echo " Done: ${EXP} at $(date)"