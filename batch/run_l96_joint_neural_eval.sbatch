#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-joint-eval
#SBATCH --array=0-2
#SBATCH --output=sbatch_logs/l96_joint_eval_%j_%a.out
#SBATCH --error=sbatch_logs/l96_joint_eval_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

DATASET="experiments/l96_datasets_obsj2_int100_nwin200.pt"

# experiment|extra_flags  (tau0 flag required for tau=0-trained joint CFM ckpts:
# Lightning hyper_parameters do not record train_tau_0_only)
TASKS=(
  "L7_joint_cfm_s0s1|--train-tau0-only"
  "L8_joint_direct_unet_s0s1|"
  "L9_joint_cfm_s0s1_multitau|"
)
SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}
EXP=${SPEC%%|*}
FLAGS=${SPEC#*|}

echo "============================================"
echo " L96 Joint Neural Eval (DA-parity) | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP} / flags: ${FLAGS:-none}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python eval_joint_neural_l96.py \
  --checkpoint "experiments/${EXP}/checkpoints/stage1_best.ckpt" \
  --dataset "${DATASET}" \
  --num-windows 200 \
  --obs-interval 100 \
  --obs-j 2 \
  --device cuda \
  ${FLAGS} \
  --output "experiments/${EXP}/joint_neural_eval.json"

echo ""
echo " Done: ${EXP} at $(date)"
