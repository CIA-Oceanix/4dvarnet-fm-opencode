#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%3
#SBATCH --job-name=l96-factorial
#SBATCH --output=sbatch_logs/l96_factorial_eval_%A_%a.out
#SBATCH --error=sbatch_logs/l96_factorial_eval_%A_%a.err
# Benchmark-default learned models on every cell of the #243 factorial grid
# (n_obs {10,20,30,50,75,100} x k {4,8,12,16}, 20 windows x 3 draws, the exact
# obs the DA baselines assimilated). One model per task, looping over cells;
# per-window scores only (no members files). SDA rows run separately with
# the validation-tuned guidance weight.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
B=$R/4dvarnet-fm-bench-default/experiments
H=$R/4dvarnet-fm-sda-nanfix/experiments
MODELS=(
  "1 1 $B/L96B_directunet_monaiM_seed1" "1 1 $B/L96B_directunet_monaiM_seed2" "1 1 $B/L96B_directunet_monaiM_seed3"
  "30 10 $H/L96B_predictstatecfm_monaiM_seed1" "30 10 $H/L96B_predictstatecfm_monaiM_seed2" "30 10 $H/L96B_predictstatecfm_monaiM_seed3"
  "30 10 $B/L96B_vanillacfm_monaiM_seed1" "30 10 $B/L96B_vanillacfm_monaiM_seed2" "30 10 $B/L96B_vanillacfm_monaiM_seed3"
)
read -r MEM OUTER EXP <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"
NAME=$(basename "$EXP")
for n in 10 20 30 50 75 100; do for k in 4 8 12 16; do
  CELL="rlayout_n${n}-${n}_k${k}-${k}_w20_d3"
  OUT="experiments/eval_factorial/${CELL}/${NAME}/ens${MEM}_no${OUTER}"
  [ -f "$OUT/neural_eval.json" ] && continue
  mkdir -p "$OUT"
  echo "=== ${NAME} ${CELL} | $(date) ==="
  python eval_neural_l96.py --step-power 1 --checkpoint "$EXP/checkpoints/stage1_best.ckpt" \
    --dataset "$R/experiments/l96_testsets_factorial/l96_testset_${CELL}.pt" --num-windows 60 \
    --n-members "$MEM" --n-outer "$OUTER" --batch-size 20 --members-file scores --output "$OUT/neural_eval.json"
done; done
echo "Done: ${NAME} at $(date)"
