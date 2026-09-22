#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=sl-mee-br-207
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --job-name=l96-p1_du_eval
#SBATCH --output=sbatch_logs/l96_p1_du_eval_%j.out
#SBATCH --error=sbatch_logs/l96_p1_du_eval_%j.err
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
DATASET="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"
for EXP in P1_directunet_monaiSplus_noaug_l96 P1_directunet_monaiM_noaug_l96; do
  OUT="experiments/${EXP}/ens1_no1"; mkdir -p "$OUT"
  echo "=== ${EXP} | deterministic | $(date) ==="
  python eval_neural_l96.py --checkpoint "experiments/${EXP}/checkpoints/stage1_best.ckpt" \
    --dataset "$DATASET" --num-windows 200 --n-members 1 --n-outer 1 --batch-size 25 \
    --output "${OUT}/neural_eval.json" || echo "FAILED: ${EXP}"
done
echo "Done: $(date)"
