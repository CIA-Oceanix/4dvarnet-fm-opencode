#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --job-name=l96-ps_m_t5
#SBATCH --output=sbatch_logs/l96_a2_ps_m_t5_train_%a_%A.out
#SBATCH --error=sbatch_logs/l96_a2_ps_m_t5_train_%a_%A.err
# T5 (var10, var100) x seeds {1, 2}, plus T0b (the P1 config at seed 2); then the P1 flow eval
# (ens30 x 10 steps, S0) with the uniform grid (--step-power 1, P1-comparable) and the default
# early-fine grid (p = 0.5).
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-tau-t5-runs
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
EXPS=(A2_predictstatecfm_monaiM_var10_l96 A2_predictstatecfm_monaiM_var10_l96 A2_predictstatecfm_monaiM_var100_l96 A2_predictstatecfm_monaiM_var100_l96 A2_predictstatecfm_monaiM_l96)
SEEDS=(1 2 1 2 2)
EXP=${EXPS[$SLURM_ARRAY_TASK_ID]}; SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "===== ${EXP} seed=${SEED} | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
python train.py --config-name experiment/${EXP} hydra.run.dir=. hydra.output_subdir=null \
  ++training.seed=${SEED} ++data.norm_stats_path=$R/experiments/l96_norm_stats_obsj2.pt
D="experiments/${EXP}_seed${SEED}"
for P in 1.0 0.5; do
  OUT="$D/ens30_no10_p${P}"; mkdir -p "$OUT"
  python eval_neural_l96.py --checkpoint "$D/checkpoints/stage1_best.ckpt" \
    --dataset $R/experiments/l96_datasets_obsj2_int100_nwin200.pt --num-windows 200 \
    --n-members 30 --n-outer 10 --step-power ${P} --batch-size 25 --cases s0 --output "$OUT/neural_eval.json"
done
echo "Done at $(date)"
