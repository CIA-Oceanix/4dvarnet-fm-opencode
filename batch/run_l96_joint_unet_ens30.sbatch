#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-joint-l10-ens30
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/l96_joint_unet_ens30_%j_%a.out
#SBATCH --error=sbatch_logs/l96_joint_unet_ens30_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-joint-neural

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

DATASET="experiments/l96_datasets_obsj2_int100_nwin200.pt"

# L10 JointCFMCoupled (multi-tau coupled ODE): N=30-member ensemble at k=1 and k=10,
# mirroring the L9 CFM ens30 framework (--ens-then-head: avg 30 member states, then
# estimate params once from the ensemble-mean state).
TASKS=(
  "L10_joint_cfm_coupled_multitau|config/experiment/L10_joint_cfm_coupled_multitau.yaml|1"
  "L10_joint_cfm_coupled_multitau|config/experiment/L10_joint_cfm_coupled_multitau.yaml|10"
)

SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}
IFS='|' read -r EXP CFG NO <<< "$SPEC"

echo "============================================"
echo " L96 Joint Neural Eval (ens30) — L10 | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP} | steps: ${NO} / ens-then-head"
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
  --n-members 30 \
  --n-outer ${NO} \
  --ens-then-head \
  --output "experiments/${EXP}/joint_neural_eval_ens30_m30_k${NO}.json"

echo ""
echo " Done: ${EXP} ens30 (M=30, k=${NO}) at $(date)"