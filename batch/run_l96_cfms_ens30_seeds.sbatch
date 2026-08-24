#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=l96-ens30-seeds
#SBATCH --array=0-9
#SBATCH --output=sbatch_logs/l96_ens30_seeds_%j_%a.out
#SBATCH --error=sbatch_logs/l96_ens30_seeds_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

DATASET="experiments/l96_datasets_obsj2_int100_nwin200.pt"

# exp|seed|n_outer  (L3 multi-tau only; S0 only; 5 seeds x 2 schemes = 10 tasks)
TASKS=(
  "L3_vanilla_cfm_s0s1|1|1"
  "L3_vanilla_cfm_s0s1|1|10"
  "L3_vanilla_cfm_s0s1|2|1"
  "L3_vanilla_cfm_s0s1|2|10"
  "L3_vanilla_cfm_s0s1|3|1"
  "L3_vanilla_cfm_s0s1|3|10"
  "L3_vanilla_cfm_s0s1|4|1"
  "L3_vanilla_cfm_s0s1|4|10"
  "L3_vanilla_cfm_s0s1|5|1"
  "L3_vanilla_cfm_s0s1|5|10"
)
SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}
EXP=${SPEC%%|*}
REST=${SPEC#*|}
SEED=${REST%%|*}
NOUTER=${REST#*|}
TAG="seed${SEED}_no${NOUTER}"

OUTDIR="experiments/${EXP}/ens30_${TAG}"
mkdir -p "${OUTDIR}"

echo "============================================"
echo " L96 CFM Ensemble Eval N=30 (S0, 5-seed study) | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID}: ${EXP} / seed=${SEED} / n_outer=${NOUTER}"
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
  --n-members 30 \
  --n-outer ${NOUTER} \
  --seed ${SEED} \
  --output "${OUTDIR}/neural_eval.json"

echo ""
echo " Done: ${EXP}/${TAG} at $(date)"
