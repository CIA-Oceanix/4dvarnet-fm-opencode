#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=sl-mee-br-207
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-w2evalL
#SBATCH --output=sbatch_logs/l96_wave2_evalL_%j.out
#SBATCH --error=sbatch_logs/l96_wave2_evalL_%j.err
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
DATASET="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"
for EXP in A2_predictstatecfm_monaiL_l96 A1_vanillacfm_monaiL_l96; do
  OUT="experiments/${EXP}/ens30_no10"; mkdir -p "$OUT"
  echo "=== ${EXP} | ens30 | $(date) ==="
  python eval_neural_l96.py --checkpoint "experiments/${EXP}/checkpoints/stage1_best.ckpt" \
    --dataset "$DATASET" --num-windows 200 --n-members 30 --n-outer 10 --batch-size 25 \
    --output "${OUT}/neural_eval.json" || echo "FAILED: ${EXP}"
done
echo "Done: $(date)"
