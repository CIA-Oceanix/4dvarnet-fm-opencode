#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-29%6
#SBATCH --job-name=l96-pooled
#SBATCH --output=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_pooled_calibration/log_%A_%a.out
#SBATCH --error=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_pooled_calibration/log_%A_%a.err
# Pooled spread/RMSE (reports/l96/probe_pooled_calibration.py) for the P1 structure doc:
# 6 rows (PSC-M, Van-M 1200 ep; SDA1/SDA2/SDA3-fix-M gw 25; DU-M(1200) -> SDA3-fix hybrid)
# x {regular grid seeds 2-3, canonical random layout seeds 1-3} = 30 tasks.
# Members stay in memory; estimates go to node-local /tmp; only pooled_*.json come back.
# Submit from the repository root after `mkdir -p experiments/l96_pooled_calibration`
# (in the main checkout).
set -euo pipefail
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
E=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments
L=$E/l96
OUT=$E/l96_pooled_calibration
NS=$E/l96_norm_stats_obsj2.pt
declare -A DS=([reg]=$E/l96_datasets_obsj2_int100_nwin200.pt [rand]=$E/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt)

TASKS=()
for row in psc van sda1 sda2 sda3fix hybrid; do
  for s in 2 3; do TASKS+=("$row reg $s"); done
  for s in 1 2 3; do TASKS+=("$row rand $s"); done
done
read -r ROW LAY S <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
sfx() { [ "$1" -eq 1 ] && echo "" || echo "_seed$1"; }

T=/tmp/pooled_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
trap 'rm -rf "$T"' EXIT
mkdir -p "$T"
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH=$PWD

COMMON=(--dataset "${DS[$LAY]}" --normalize-stats "$NS" --n-members 30 --seed 0 --output "$T/eval.json")
case $ROW in
  psc|van)
    fam=$([ "$ROW" = psc ] && echo predictstatecfm || echo vanillacfm)
    r=$L/L96B_${fam}_monaiM_ep1200_seed$S; name=${fam}1200s$S
    ARGS=(eval_neural_l96 --checkpoint "$r/checkpoints/stage1_best.ckpt" --config "$r/resolved_config.yaml"
          --n-outer 20 --step-power 0.5 --batch-size 50) ;;
  sda1|sda2|sda3fix)
    case $ROW in
      sda1) n=B4_sda1_monaiM_l96$(sfx "$S") ;;
      sda2) n=A3_sda2_monaiM_l96$(sfx "$S") ;;
      sda3fix) n=A3_sda3fix_monaiM_l96_seed$S ;;
    esac
    r=$L/$n; name=$n
    ARGS=(eval_sda_l96 --checkpoint "$r/checkpoints/stage1_best.ckpt" --config "$r/resolved_config.yaml"
          --n-outer 10 --guidance-weight 25 --r-var 0.5 --batch-size 50) ;;
  hybrid)
    m=$L/L96B_directunet_monaiM_ep1200_seed$S; r=$L/A3_sda3fix_monaiM_l96_seed1; name=hybrid_DU1200s${S}_sda3fix
    ARGS=(eval_sda_directunet_hybrid_l96 --mean-checkpoint "$m/checkpoints/stage1_best.ckpt" --mean-config "$m/resolved_config.yaml"
          --sda-checkpoint "$r/checkpoints/stage1_best.ckpt" --sda-config "$r/resolved_config.yaml"
          --n-outer 10 --tau0 0.1 --guidance-weight 2 --r-var 0.5 --batch-size 16) ;;
esac
echo "task $SLURM_ARRAY_TASK_ID: $name $LAY $(hostname) $(date +%T)"
python -u reports/l96/probe_pooled_calibration.py "${ARGS[@]}" "${COMMON[@]}"
for c in s0 s1; do cp "$T/pooled_$c.json" "$OUT/${name}_${LAY}_$c.json"; done
echo "done $name $LAY $(date +%T)"
