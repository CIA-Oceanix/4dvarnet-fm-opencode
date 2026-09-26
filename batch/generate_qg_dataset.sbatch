#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=qg-dataset
#SBATCH --output=sbatch_logs/qg_dataset_%A_%a.out
#SBATCH --error=sbatch_logs/qg_dataset_%A_%a.err
# Generate one shard of one split of a QG spectral-wind dataset (scripts/generate_qg_dataset.py).
# One array task = one shard. Submit one array per split, then run the assemble step once
# (it writes manifests, independence and diversity reports; the test split becomes read-only):
#   SPEC=qg_specwind_gyrostat_v1 SPLIT=train N_SHARDS=20 sbatch --array=0-19 batch/generate_qg_dataset.sbatch
#   SPEC=qg_specwind_gyrostat_v1 SPLIT=val   N_SHARDS=2  sbatch --array=0-1  batch/generate_qg_dataset.sbatch
#   SPEC=qg_specwind_gyrostat_v1 SPLIT=test  N_SHARDS=2  sbatch --array=0-1  batch/generate_qg_dataset.sbatch
#   $PY scripts/generate_qg_dataset.py assemble --spec qg_specwind_gyrostat_v1
# Storage: `$PY scripts/generate_qg_dataset.py estimate --spec <name>` (the full v1 spec is ~23 GB).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin/python
export OMP_NUM_THREADS=4
: "${SPEC:?set SPEC}" "${SPLIT:?set SPLIT}" "${N_SHARDS:?set N_SHARDS}"
ROOT="${ROOT:-experiments/qg_datasets}"
"$PY" scripts/generate_qg_dataset.py generate --spec "$SPEC" --split "$SPLIT" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "$N_SHARDS" --root "$ROOT" \
  --batch-size "${BATCH_SIZE:-256}" --device cuda
