#!/bin/bash
#SBATCH --job-name=qg_s1
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

# S1 cross-resolution case: DA model at 16x16 vs truth at 64x64, psi-obs
# (streamfunction columns), cols=4, 1% obs noise, lag 1.0/2.0.
# The DA model carries param bias (rd,rek *0.85) + corrupted wind (S1 scenario),
# exercising init downsampling, the upsampled H-function, physical-coordinate
# localization, and trajectory upsample for metrics.
LAGS=(1.0 2.0)
LAG=${LAGS[$SLURM_ARRAY_TASK_ID]}

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --da-nx 16 \
    --window-days 30 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list etkf \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 4 \
    --obs-noise-frac-list 0.01 \
    --init-lag-days-list ${LAG} \
    --band 0.25 \
    --obs-var psi \
    --scenarios test_s0,test_s1 \
    --outdir reports/outputs/qg_s1_lag${LAG//./p} \
    --tag s1
