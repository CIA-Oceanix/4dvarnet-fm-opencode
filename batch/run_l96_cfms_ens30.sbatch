#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=l96-ens30
#SBATCH --array=0-3
#SBATCH --output=sbatch_logs/l96_ens30_%j_%a.out
#SBATCH --error=sbatch_logs/l96_ens30_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

DATASET="experiments/l96_datasets_obsj2_int100_nwin200.pt"

# exp|tag|extra_flags  (S0 only; tau0 flag required for tau=0-trained CFM ckpts)
TASKS=(
  "L3_vanilla_cfm_s0s1|no1|--n-members 30 --n-outer 1|"
  "L3_vanilla_cfm_s0s1|no10|--n-members 30 --n-outer 10|"
  "L2b_vanilla_cfm_s0s1|no1|--n-members 30 --n-outer 1|--train-tau0-only"
  "L2b_vanilla_cfm_s0s1|no10|--n-members 30 --n-outer 10|--train-tau0-only"
)
SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}
EXP=${SPEC%%|*}
REST=${SPEC#*|}
TAG=${REST%%|*}
REST=${REST#*|}
ARGS=${REST%%|*}
FLAGS=${REST#*|}

OUTDIR="experiments/${EXP}/ens30_${TAG}"
mkdir -p "${OUTDIR}"

echo "============================================"
echo " L96 CFM Ensemble Eval N=30 (S0) | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID}: ${EXP} / ${TAG} / args: ${ARGS} / flags: ${FLAGS:-none}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python eval_neural_l96.py \
  --checkpoint "experiments/${EXP}/checkpoints/stage1_best.ckpt" \
  --dataset "${DATASET}" \
  --num-windows 200 \
  --obs-interval 100 \
  --obs-j 2 \
  --cases s0 \
  --device cuda \
  ${ARGS} \
  ${FLAGS} \
  --output "${OUTDIR}/neural_eval.json"

echo ""
echo " Done: ${EXP}/${TAG} at $(date)"
