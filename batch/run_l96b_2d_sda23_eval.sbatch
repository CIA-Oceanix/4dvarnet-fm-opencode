#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-25%4
#SBATCH --job-name=l96b-2d-sda23-eval
#SBATCH --output=sbatch_logs/l96b_2d_sda23_eval_%A_%a.out
# Evaluations that bring the 2D SDA2 / SDA3-fix priors and the 2D seeds 2-3 into the P1 tables
# (same samplers and sampling seed as the benchmark rows, batch/run_l96b_sda23_2d.sbatch):
#   0-11  guidance-weight validation of DirectUNet-2D-M (seeds 1-3) -> SDA3-fix-2D-M (seed 1), tau0 0.1,
#         gw {1, 2}, 100 validation windows, regular and random layouts, S0 + S1 (scores only);
#   12-23 regular test set with the ensemble variance stored (--members-file variance): VanillaCFM-2D seeds 2-3,
#         SDA1-2D seeds 2-3, SDA2-2D and SDA3-fix-2D seeds 1-3 (gw 25), DirectUNet-2D seeds 2-3 -> SDA1-2D (gw 1);
#   24-25 the same two hybrids on the canonical random test set (scores only);
#   26-28 DirectUNet-2D seeds 1-3 -> SDA3-fix-2D at the validation-selected gw (GW=...), regular, with variance.
# Variance outputs go to sibling *_var dirs under experiments/l96/ (as batch/run_l96b_2d_variance_eval.sbatch).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig" FDV_MEMBERS_TMP="$NODE_TMP/members"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
SRC=${SRC:-$R/4dvarnet-fm-l96-2d-sda23/experiments}
L96=$R/experiments/l96
NORM=$R/experiments/l96_norm_stats_obsj2.pt
REG=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt
CAN=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt
VAL=$R/experiments/l96_hybrid_gw_revalidation/val_du2d_sda3fix2d
i=${SLURM_ARRAY_TASK_ID:-0}
echo "===== task $i | commit $(git rev-parse --short HEAD) | $(hostname) | $(date) ====="

sda() {  # ckpt_dir dataset n_windows out mode
  mkdir -p "$4"
  python eval_sda_l96.py --checkpoint $1/checkpoints/stage1_best.ckpt --config $1/resolved_config.yaml \
    --dataset "$2" --num-windows $3 --batch-size 16 --n-members 30 --n-outer 10 --r-var 0.5 \
    --guidance-weight 25 --normalize-stats $NORM --members-file $5 --output "$4/sda_eval.json"
}
hybrid() {  # du_seed sda_dir dataset n_windows gw out mode
  local DU=$L96/L96B_directunet2d_monaiM_seed$1; mkdir -p "$6"
  python eval_sda_directunet_hybrid_l96.py \
    --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
    --sda-checkpoint $2/checkpoints/stage1_best.ckpt --sda-config $2/resolved_config.yaml \
    --dataset "$3" --num-windows $4 --batch-size 25 --n-members 30 --n-outer 10 --tau0 0.1 \
    --guidance-weight $5 --normalize-stats $NORM --members-file $7 --output "$6/hybrid_eval.json"
}

SDA3=$SRC/L96B_sda3fix_monai2dM_ep1200_seed1
SDA1=$L96/L96B_sda1_monai2dM_ep1200_seed1
if [ $i -lt 12 ]; then
  S=$((i % 3 + 1)); GW=$(( (i / 3) % 2 + 1 )); LAY=$([ $i -lt 6 ] && echo regular || echo random)
  V=$R/experiments/l96_valset_$([ $LAY = regular ] && echo regular || echo rlayout_n10-100_k4-16)_w100.pt
  hybrid $S $SDA3 $V 100 $GW $VAL/$LAY/du2d$S/tau0.1_gw$GW scores
elif [ $i -lt 24 ]; then
  j=$((i - 12))
  case $j in
    0|1) N=L96B_vanillacfm_monai2dM_ep1200_seed$((j + 2)); OUT=$L96/$N/ens30_no20_var; mkdir -p "$OUT"
         python eval_neural_l96.py --step-power 0.5 --checkpoint $SRC/$N/checkpoints/stage1_best.ckpt --dataset $REG \
           --num-windows 200 --n-members 30 --n-outer 20 --batch-size 25 --members-file variance \
           --output "$OUT/neural_eval.json" ;;
    2|3) N=L96B_sda1_monai2dM_ep1200_seed$((j)); sda $SRC/$N $REG 200 $L96/eval_regular/$N/ens30_gw25_var variance ;;
    4|5|6) N=L96B_sda2_monai2dM_ep1200_seed$((j - 3)); sda $SRC/$N $REG 200 $L96/eval_regular/$N/ens30_gw25_var variance ;;
    7|8|9) N=L96B_sda3fix_monai2dM_ep1200_seed$((j - 6)); sda $SRC/$N $REG 200 $L96/eval_regular/$N/ens30_gw25_var variance ;;
    10|11) S=$((j - 8)); hybrid $S $SDA1 $REG 200 1 \
             $L96/eval_regular/hybrid_DU2dMs${S}_L96B_sda1_monai2dM_ep1200_seed1/tau0.1_gw1_var variance ;;
  esac
elif [ $i -lt 26 ]; then
  S=$((i - 22)); hybrid $S $SDA1 $CAN 200 1 \
    $L96/eval_rlayout_n10-100_k4-16_w200/hybrid_DU2dMs${S}_L96B_sda1_monai2dM_ep1200_seed1/tau0.1_gw1 scores
else
  S=$((i - 25)); GW=${GW:?set GW to the validation-selected guidance weight}
  hybrid $S $SDA3 $REG 200 $GW \
    $L96/eval_regular/hybrid_DU2dMs${S}_L96B_sda3fix_monai2dM_ep1200_seed1/tau0.1_gw${GW}_var variance
fi
echo "Done: task $i at $(date)"
