#!/bin/bash
#SBATCH --job-name=qg_chunk
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:35:00
#SBATCH --output=batch/logs/qg_chunk_%A_%a.out
#SBATCH --error=batch/logs/qg_chunk_%A_%a.err

set -e

# SPLIT and CHUNK are set via `sbatch --export=SPLIT=train,CHUNK=25,OUT_DIR=...`
: "${SPLIT:?must set SPLIT via --export}"
: "${CHUNK:?must set CHUNK via --export}"
: "${OUT_DIR:?must set OUT_DIR via --export}"

START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python \
    reports/qg/generate_qg_window_chunk.py \
    --split "$SPLIT" --start "$START" --count "$CHUNK" --out-dir "$OUT_DIR"
