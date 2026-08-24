#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=l96-ens30-s1
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/l96_ens30_s1_%j_%a.out
#SBATCH --error=sbatch_logs/l96_ens30_s1_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

DATASET="experiments/l96_datasets_obsj2_int100_nwin200.pt"

# exp|tag|n_outer  (L3 multi-tau only; S1 only; 2 tasks)
TASKS=(
  "L3_vanilla_cfm_s0s1|no1|1"
  "L3_vanilla_cfm_s0s1|no10|10"
)
SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}
EXP=${SPEC%%|*}
REST=${SPEC#*|}
TAG=${REST%%|*}
NOUTER=${REST#*|}

OUTDIR="experiments/${EXP}/ens30_s1_${TAG}"
mkdir -p "${OUTDIR}"

echo "============================================"
echo " L96 CFM Ensemble Eval N=30 (S1) | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID}: ${EXP} / n_outer=${NOUTER}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python eval_neural_l96.py \
  --checkpoint "experiments/${EXP}/checkpoints/stage1_best.ckpt" \
  --dataset "${DATASET}" \
  --num-windows 200 \
  --obs-interval 100 \
  --obs-j 2 \
  --cases s1 \
  --device cuda \
  --n-members 30 \
  --n-outer ${NOUTER} \
  --seed 0 \
  --output "${OUTDIR}/neural_eval.json"

echo ""
echo " Done: ${EXP}/ens30_s1_${TAG} at $(date)"
