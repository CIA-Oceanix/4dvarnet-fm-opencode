#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%2
#SBATCH --job-name=l96-obs-learned
#SBATCH --output=sbatch_logs/l96_obs_learned_%A_%a.out
#SBATCH --error=sbatch_logs/l96_obs_learned_%A_%a.err
# P1 observing-system figure: the 2D benchmark learned schemes (1200 epochs, seed 1) on the 16 cells,
# on the exact obs the DA cells assimilated (experiments/l96_testsets_{factorial,ood}); estimates + scores only.
#   0 DirectUNet-2D-M   1 VanillaCFM-2D-M (30 members, 20 early-fine steps)
#   2,3 SDA2-2D-M (gw 25, 10 steps; cells 0-7 / 8-15)
#   4,5 DirectUNet-2D-M -> SDA3-fix-2D-M (tau0 0.1, gw 1, 10 steps; cells 0-7 / 8-15)
# The 2026-11-09 run of tasks 2-5 was submitted from feature/l96-2d-sda23 (before #384 merged its SDA2 / SDA3-fix
# 2D model types) with OUTROOT=<this worktree>/experiments; OUTROOT redirects the outputs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig" FDV_MEMBERS_TMP="$NODE_TMP/members"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
L96=$R/experiments/l96
S23=${S23:-$R/4dvarnet-fm-l96-2d-sda23/experiments}
NORM=$R/experiments/l96_norm_stats_obsj2.pt
DU=$L96/L96B_directunet2d_monaiM_seed1
CFM=$L96/L96B_vanillacfm_monai2dM_ep1200_seed1
SDA2=$S23/L96B_sda2_monai2dM_ep1200_seed1
SDA3=$S23/L96B_sda3fix_monai2dM_ep1200_seed1
CELLS=()
for n in 6 10 20 30 50 75 100 300 1000; do CELLS+=("rlayout_n${n}-${n}_k16-16_w20_d3"); done
for k in 0 2 4 8 12; do CELLS+=("rlayout_n30-30_k${k}-${k}_w20_d3"); done
CELLS+=("rlayout_n30-30_k16-16_w20_d3_r0.25" "rlayout_n30-30_k16-16_w20_d3_r1")
i=$SLURM_ARRAY_TASK_ID
case $i in 0|1) SEL=$(seq 0 15) ;; 2|4) SEL=$(seq 0 7) ;; 3|5) SEL=$(seq 8 15) ;; esac
for c in $SEL; do
  CELL=${CELLS[$c]}
  DS=$R/experiments/l96_testsets_factorial/l96_testset_${CELL}.pt
  [ -f "$DS" ] || DS=$R/experiments/l96_testsets_ood/l96_testset_${CELL}.pt
  RV=0.5; case $CELL in *_r0.25) RV=0.25 ;; *_r1) RV=1.0 ;; esac
  E=${OUTROOT:-experiments}/eval_obs_system/$CELL
  echo "=== task $i $CELL (R $RV) | $(date +%T) ==="
  case $i in
    0) OUT=$E/L96B_directunet2d_monaiM_seed1/ens1_no1; [ -f $OUT/neural_eval.json ] && continue; mkdir -p $OUT
       python eval_neural_l96.py --step-power 0.5 --checkpoint $DU/checkpoints/stage1_best.ckpt --dataset $DS \
         --num-windows 60 --n-members 1 --n-outer 1 --batch-size 20 --members-file scores --output $OUT/neural_eval.json ;;
    1) OUT=$E/L96B_vanillacfm_monai2dM_ep1200_seed1/ens30_no20; [ -f $OUT/neural_eval.json ] && continue; mkdir -p $OUT
       python eval_neural_l96.py --step-power 0.5 --checkpoint $CFM/checkpoints/stage1_best.ckpt --dataset $DS \
         --num-windows 60 --n-members 30 --n-outer 20 --batch-size 20 --members-file scores --output $OUT/neural_eval.json ;;
    2|3) OUT=$E/L96B_sda2_monai2dM_ep1200_seed1/ens30_gw25; [ -f $OUT/sda_eval.json ] && continue; mkdir -p $OUT
       python eval_sda_l96.py --checkpoint $SDA2/checkpoints/stage1_best.ckpt --config $SDA2/resolved_config.yaml \
         --dataset $DS --num-windows 60 --batch-size 16 --n-members 30 --n-outer 10 --r-var $RV --guidance-weight 25 \
         --normalize-stats $NORM --members-file scores --output $OUT/sda_eval.json ;;
    4|5) OUT=$E/hybrid_DU2dMs1_L96B_sda3fix_monai2dM_ep1200_seed1/tau0.1_gw1; [ -f $OUT/hybrid_eval.json ] && continue; mkdir -p $OUT
       python eval_sda_directunet_hybrid_l96.py \
         --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
         --sda-checkpoint $SDA3/checkpoints/stage1_best.ckpt --sda-config $SDA3/resolved_config.yaml \
         --dataset $DS --num-windows 60 --batch-size 20 --n-members 30 --n-outer 10 --tau0 0.1 --guidance-weight 1 \
         --r-var $RV --normalize-stats $NORM --members-file scores --output $OUT/hybrid_eval.json ;;
  esac
done
echo "Done task $i at $(date)"
