#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=qg-specwind-neural
#SBATCH --output=sbatch_logs/qg_specwind_neural_%j.out
#SBATCH --error=sbatch_logs/qg_specwind_neural_%j.err
# Round-1 neural schemes on the spectral-wind gyrostat QG benchmark
# (docs/plans/analysis/qg_specwind_neural.md): trains one config/experiment/G*_specwind.yaml
# on G5-regenerated windows. rtx8000 because the val split is re-materialized at full
# resolution and load_full_res_windows refuses another GPU type. Examples:
#   EXP_ID=G1_direct_unet_tchannels_specwind EPOCHS=2 TAG=e0 sbatch batch/run_qg_specwind_neural_train.sbatch
#   EXP_ID=G1_direct_unet_tchannels_specwind EPOCHS=50 QSCALE=0.3 TAG=lq0.3 sbatch ...
# MODEL_TYPE defaults to the YAML's model_type; QSCALE overrides training.q_loss_weight_scale;
# EXTRA is passed through to train_qg_neural.py.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin/python
export OMP_NUM_THREADS=4
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
: "${EXP_ID:?set EXP_ID}"
SPEC="${SPEC:-qg_specwind_gyrostat_v1}"
ROOT="${ROOT:-/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/qg_datasets}"
SEED="${SEED:-0}"; TAG="${TAG:-s${SEED}}"
MODEL_TYPE="${MODEL_TYPE:-$("$PY" -c "from omegaconf import OmegaConf; print(OmegaConf.load('config/experiment/${EXP_ID}.yaml').model_type)")}"
OUT="experiments/qg_specwind_neural/${EXP_ID}_${TAG}"
ARGS=(--model-type "$MODEL_TYPE" --exp-id "$EXP_ID" --exp-dir "$OUT"
      --specwind-spec "$SPEC" --specwind-root "$ROOT"
      --regen-windows "${REGEN_WINDOWS:-1000}" --regen-every "${REGEN_EVERY:-5}"
      --batch-size "${BATCH:-2}" --num-workers 4 --seed "$SEED")
[ -n "${EPOCHS:-}" ] && ARGS+=(--epochs "$EPOCHS")
[ -n "${QSCALE:-}" ] && ARGS+=(--q-loss-weight-scale "$QSCALE")
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/time -v "$PY" train_qg_neural.py "${ARGS[@]}" ${EXTRA:-} 2>&1 | tee "$OUT/train.log"
