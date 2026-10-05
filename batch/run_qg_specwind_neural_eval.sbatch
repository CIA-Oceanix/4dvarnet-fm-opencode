#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=qg-specwind-neural-eval
#SBATCH --output=sbatch_logs/qg_specwind_neural_eval_%j.out
#SBATCH --error=sbatch_logs/qg_specwind_neural_eval_%j.err
# Scores a trained round-1 network on the gyrostat test windows like the DA rows
# (evaluation/run_qg_specwind_neural.py). The test split is stored at full resolution, so
# any GPU works; SPLIT=val regenerates windows and needs --gres=gpu:rtx8000:1. Example:
#   RUN_DIR=experiments/qg_specwind_neural/G1_direct_unet_tchannels_specwind_s0 TAG=G1_s0 \
#     sbatch batch/run_qg_specwind_neural_eval.sbatch
# Output: experiments/qg_specwind_da/<SPEC>/<SPLIT>/neural_<TAG>_<INTERP>_N<N>/shard_0_of_1_meta.json,
# where the gyrostat DA report reads its rows. SDA priors (G3): GW = guidance weight (L2 step over
# one window's normalized state), SDA_STEPS = guided Euler steps; put the GW in TAG.
# S1=1 scores on the realistic-S1 observations (the DA S1 rows' exact altimetry-error draws).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin/python
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
: "${RUN_DIR:?set RUN_DIR}" "${TAG:?set TAG}"
SPEC="${SPEC:-qg_specwind_gyrostat_v1}"; SPLIT="${SPLIT:-test}"; N="${N:-30}"; INTERP="${INTERP:-cubic}"
COLS="${COLS:-3}"; N_WINDOWS="${N_WINDOWS:-100}"; GW="${GW:-100}"; SDA_STEPS="${SDA_STEPS:-10}"; S1="${S1:-0}"
ROOT="${ROOT:-/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/qg_datasets}"
OUT="experiments/qg_specwind_da/${SPEC}/${SPLIT}/neural_${TAG}_c${COLS}_${INTERP}_N${N}$([ "$S1" = 1 ] && echo "_k1_rbase" || true)/shard_0_of_1.json"
mkdir -p "$(dirname "$OUT")"
"$PY" -m evaluation.run_qg_specwind_neural --run-dir "$RUN_DIR" --spec "$SPEC" --root "$ROOT" \
  --split "$SPLIT" --n-windows "$N_WINDOWS" --cols-per-day "$COLS" --n-members "$N" \
  --interp "$INTERP" --guidance-weight "$GW" --sda-steps "$SDA_STEPS" $([ "$S1" = 1 ] && echo "--s1" || true) --out "$OUT" --device cuda
