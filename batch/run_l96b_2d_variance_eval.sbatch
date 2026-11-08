#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --job-name=l96b-2d-var
#SBATCH --output=sbatch_logs/l96b_2d_var_%A_%a.out
#SBATCH --error=sbatch_logs/l96b_2d_var_%A_%a.err
# Re-evaluates the P1 2D learned rows (regular test set, S0 and S1) with the ensemble variance
# stored (--members-file variance), for the pooled spread-skill ratio and the Gaussian-form FMS.
# Same samplers and seed as the benchmark rows (batch/run_l96b_vcfm2d.sbatch,
# batch/run_l96b_sda1_2d.sbatch). Outputs go to sibling *_var dirs under experiments/l96/.
# Task 0: VanillaCFM-2D-M; 1: SDA1-2D-M (gw 25); 2: DirectUNet-2D-M -> SDA1-2D-M (tau0 0.1, gw 1).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
SRC=${SRC:-$R/4dvarnet-fm-l96-fdv-2d/experiments}
L96=$R/experiments/l96
NORM=$R/experiments/l96_norm_stats_obsj2.pt
DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt
CFM=$SRC/L96B_vanillacfm_monai2dM_ep1200_seed1
SDA=$SRC/L96B_sda1_monai2dM_ep1200_seed1
DU=$L96/L96B_directunet2d_monaiM_seed1
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) OUT=$L96/L96B_vanillacfm_monai2dM_ep1200_seed1/ens30_no20_var; mkdir -p "$OUT"
     python eval_neural_l96.py --step-power 0.5 --checkpoint $CFM/checkpoints/stage1_best.ckpt --dataset $DS \
       --num-windows 200 --n-members 30 --n-outer 20 --batch-size 25 --members-file variance \
       --output "$OUT/neural_eval.json" ;;
  1) OUT=$L96/eval_regular/L96B_sda1_monai2dM_ep1200_seed1/ens30_gw25_var; mkdir -p "$OUT"
     python eval_sda_l96.py --checkpoint $SDA/checkpoints/stage1_best.ckpt --config $SDA/resolved_config.yaml \
       --dataset $DS --num-windows 200 --batch-size 16 --n-members 30 --n-outer 10 --r-var 0.5 \
       --guidance-weight 25 --normalize-stats $NORM --members-file variance --output "$OUT/sda_eval.json" ;;
  2) OUT=$L96/eval_regular/hybrid_DU2dMs1_L96B_sda1_monai2dM_ep1200_seed1/tau0.1_gw1_var; mkdir -p "$OUT"
     python eval_sda_directunet_hybrid_l96.py \
       --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
       --sda-checkpoint $SDA/checkpoints/stage1_best.ckpt --sda-config $SDA/resolved_config.yaml \
       --dataset $DS --num-windows 200 --batch-size 25 --n-members 30 --n-outer 10 --tau0 0.1 \
       --guidance-weight 1 --normalize-stats $NORM --members-file variance --output "$OUT/hybrid_eval.json" ;;
esac
echo "Done: task ${SLURM_ARRAY_TASK_ID:-0} at $(date)"
