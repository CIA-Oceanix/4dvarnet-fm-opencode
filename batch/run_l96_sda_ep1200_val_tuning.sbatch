#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --exclude=sl-mee-br-207
#SBATCH --job-name=l96-sda1200-val
#SBATCH --output=sbatch_logs/l96_sda1200_val_%A_%a.out
#SBATCH --error=sbatch_logs/l96_sda1200_val_%A_%a.err
# Validation re-tune for the 1200-epoch SDA priors (seed 1), same protocol as the
# 400-epoch tuning (l96_benchmark_extended.md section 6), 50 validation windows
# per case, regular grid and random layout:
#   tasks 0-5: guidance weight {10,15,20,25,30,40} for SDA1 / SDA2 / SDA3-fix
#   tasks 6-9: DirectUNet-M(1200, seed 1) -> SDA2 / SDA3-fix hybrid, tau0 {0.1,0.2,0.3} x gw {1,2,5}
#              (+ DirectUNet alone)
# Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
NORM=experiments/l96_norm_stats_obsj2.pt
DU=$R/experiments/l96/L96B_directunet_monaiM_ep1200_seed1
PRIORS=(B4_sda1_monaiM_ep1200_l96 A3_sda2_monaiM_ep1200_l96 A3_sda3fix_monaiM_ep1200_l96)
LAYOUTS=(regular rlayout)
valset() { if [ "$1" = regular ]; then echo $R/experiments/l96_valset_regular_w50.pt; else echo $R/experiments/l96_valset_rlayout_n10-100_k4-16_w50.pt; fi; }
i=$SLURM_ARRAY_TASK_ID
echo "task $i | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
if [ $i -le 5 ]; then
  P=${PRIORS[$((i / 2))]}; RG=${LAYOUTS[$((i % 2))]}; V=$(valset $RG); SDA=experiments/${P}_seed1
  for GW in 10 15 20 25 30 40; do
    O=experiments/eval_val/$RG/gw_sweep/${P}_seed1/gw$GW
    [ -f $O/sda_eval.json ] && continue; mkdir -p $O
    python eval_sda_l96.py --checkpoint $SDA/checkpoints/stage1_best.ckpt --config $SDA/resolved_config.yaml \
      --dataset $V --num-windows 50 --batch-size 16 --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight $GW \
      --normalize-stats $NORM --members-file scores --output $O/sda_eval.json
  done
else
  j=$((i - 6)); P=${PRIORS[$((j / 2 + 1))]}; RG=${LAYOUTS[$((j % 2))]}; V=$(valset $RG); SDA=experiments/${P}_seed1
  O=experiments/eval_val/$RG/du_alone/L96B_directunet_monaiM_ep1200_seed1
  [ -f $O/neural_eval.json ] || { mkdir -p $O; python eval_neural_l96.py --step-power 0.5 --checkpoint $DU/checkpoints/stage1_best.ckpt \
    --dataset $V --num-windows 50 --n-members 1 --n-outer 1 --batch-size 25 --members-file scores --output $O/neural_eval.json; }
  for T0 in 0.1 0.2 0.3; do for GW in 1 2 5; do
    O=experiments/eval_val/$RG/hybrid_du1200/${P}_seed1/tau${T0}_gw$GW
    [ -f $O/hybrid_eval.json ] && continue; mkdir -p $O
    python eval_sda_directunet_hybrid_l96.py --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
      --sda-checkpoint $SDA/checkpoints/stage1_best.ckpt --sda-config $SDA/resolved_config.yaml \
      --dataset $V --num-windows 50 --batch-size 25 --n-members 30 --n-outer 10 --tau0 $T0 --guidance-weight $GW \
      --normalize-stats $NORM --members-file scores --output $O/hybrid_eval.json
  done; done
fi
echo "Done task $i at $(date)"
