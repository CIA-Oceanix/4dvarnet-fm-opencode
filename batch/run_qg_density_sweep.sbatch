#!/bin/bash
#SBATCH --job-name=qg_density
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

COLS=(4 6 12)
C=${COLS[$SLURM_ARRAY_TASK_ID]}

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 60 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list etkf \
    --init lagged \
    --geometry random_columns \
    --cols-per-day ${C} \
    --init-lag-days-list 0.5 \
    --obs-var q \
    --scenarios test_s0,test_s1a \
    --outdir reports/outputs/qg_density_sweep \
    --tag density_sweep_cols${C}
