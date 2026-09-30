#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-35%6
#SBATCH --exclude=sl-mee-br-202,sl-mee-br-203,sl-mee-br-204,sl-mee-br-205,sl-mee-br-208,sl-mee-br-210,sl-mee-br-211,sl-mee-br-212,sl-mee-br-213,sl-mee-br-214,sl-mee-br-215,sl-mee-br-216
#SBATCH --job-name=l96-fmscore
#SBATCH --output=sbatch_logs/l96_fmscore_%A_%a.out
#SBATCH --error=sbatch_logs/l96_fmscore_%A_%a.err
# FM-operator score (reports/l96/probe_fm_score_eval.py) of the current L96 benchmark
# ensemble rows, members kept in memory: PSC-M / Van-M (1200 ep, ens30_no20),
# SDA1 / SDA2 / SDA3-fix-M (1200 ep, gw 25), DU-M(1200) -> SDA3-fix-M(1200, seed 1)
# hybrid (tau0 0.1, gw 2); seeds 1-3 x {regular grid, canonical random set} = 36 tasks.
# Only fmscore_*.json come back (to $OUT). Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
export PYTHONPATH=$PWD
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
E=$R/experiments
L=$E/l96
S12=$R/4dvarnet-fm-sda-ep1200/experiments
OUT=$E/l96_fm_score_current
mkdir -p "$OUT"
NS=$E/l96_norm_stats_obsj2.pt
declare -A DS=([reg]=$E/l96_datasets_obsj2_int100_nwin200.pt [rand]=$E/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt)

TASKS=()
for row in psc van sda1 sda2 sda3fix hybrid; do
  for s in 1 2 3; do for lay in reg rand; do TASKS+=("$row $lay $s"); done; done
done
read -r ROW LAY S <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

COMMON=(--dataset "${DS[$LAY]}" --num-windows 200 --normalize-stats "$NS" --n-members 30 --seed 0 --output "$NODE_TMP/eval.json")
case $ROW in
  psc|van)
    fam=$([ "$ROW" = psc ] && echo predictstatecfm || echo vanillacfm)
    r=$L/L96B_${fam}_monaiM_ep1200_seed$S; name=${fam}_ep1200_seed$S
    ARGS=(eval_neural_l96 --checkpoint "$r/checkpoints/stage1_best.ckpt" --config "$r/resolved_config.yaml"
          --n-outer 20 --step-power 0.5 --batch-size 50) ;;
  sda1|sda2|sda3fix)
    case $ROW in
      sda1) n=B4_sda1_monaiM_ep1200_l96_seed$S ;;
      sda2) n=A3_sda2_monaiM_ep1200_l96_seed$S ;;
      sda3fix) n=A3_sda3fix_monaiM_ep1200_l96_seed$S ;;
    esac
    r=$S12/$n; name=$n
    ARGS=(eval_sda_l96 --checkpoint "$r/checkpoints/stage1_best.ckpt" --config "$r/resolved_config.yaml"
          --n-outer 10 --guidance-weight 25 --r-var 0.5 --batch-size 16) ;;
  hybrid)
    m=$L/L96B_directunet_monaiM_ep1200_seed$S; r=$S12/A3_sda3fix_monaiM_ep1200_l96_seed1
    name=hybrid_DU1200s${S}_sda3fix1200s1
    ARGS=(eval_sda_directunet_hybrid_l96 --mean-checkpoint "$m/checkpoints/stage1_best.ckpt" --mean-config "$m/resolved_config.yaml"
          --sda-checkpoint "$r/checkpoints/stage1_best.ckpt" --sda-config "$r/resolved_config.yaml"
          --n-outer 10 --tau0 0.1 --guidance-weight 2 --r-var 0.5 --batch-size 25) ;;
esac
echo "task $SLURM_ARRAY_TASK_ID: $name $LAY | commit $(git rev-parse --short HEAD) | $(hostname) | $(date +%T)"
python -u reports/l96/probe_fm_score_eval.py "${ARGS[@]}" "${COMMON[@]}"
for c in s0 s1; do cp "$NODE_TMP/fmscore_$c.json" "$OUT/${name}_${LAY}_$c.json"; done
echo "done $name $LAY $(date +%T)"
