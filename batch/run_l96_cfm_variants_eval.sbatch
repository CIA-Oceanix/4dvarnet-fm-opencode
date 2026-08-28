#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-cfm-v2-v3-eval
#SBATCH --array=0-1
#SBATCH --output=sbatch_logs/l96_cfm_variants_eval_%j_%a.out
#SBATCH --error=sbatch_logs/l96_cfm_variants_eval_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR"

# Canonical 200-window S0/S1 test set (shared with DA baselines + L1b-L9.
# Must match the --data.test_cache used at training time).
DATASET="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

# experiment|checkpoint|config   (V2 is two-stage -> stage2; V3 single-stage -> stage1)
TASKS=(
  "V3_predict_state_cfm_l96|checkpoints/stage1_best.ckpt|experiment/V3_predict_state_cfm_l96"
  "V2_tweedie_cfm_l96|checkpoints/stage2_best.ckpt|experiment/V2_tweedie_cfm_l96"
)
SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}
EXP=${SPEC%%|*}
REST=${SPEC#*|}
CKPT_FILE=${REST%%|*}
EXP_CONFIG=${REST#*|}

echo "============================================"
echo " L96 CFM Variants Eval (DA-parity) | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP}"
echo "  Checkpoint: ${CKPT_FILE} / Config: ${EXP_CONFIG}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python eval_neural_l96.py \
  --checkpoint "experiments/${EXP}/${CKPT_FILE}" \
  --config "config/${EXP_CONFIG}.yaml" \
  --dataset "${DATASET}" \
  --num-windows 200 \
  --obs-interval 100 \
  --obs-j 2 \
  --device cuda \
  --output "experiments/${EXP}/neural_eval.json"

echo ""
echo " Done: ${EXP} at $(date)"
echo ""
