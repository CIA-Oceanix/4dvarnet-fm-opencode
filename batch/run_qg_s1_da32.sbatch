#!/bin/bash
#SBATCH --job-name=qg_s1_da32
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=qgs1d32_%A_%a.out
#SBATCH --error=qgs1d32_%A_%a.err

set -e

# S1 cross-resolution case at da_nx=32: DA model at 32x32 vs truth at 64x64
# (ratio 2:1, milder mismatch than the da_nx=16 run's 4:1), psi-obs
# (streamfunction columns), cols=4, 1% obs noise, lag 1.0/2.0.
# Same seams as da_nx=16: param bias (rd,rek *0.85) + corrupted wind (S1),
# init downsampling, upsampled H-function, physical-coordinate localization,
# trajectory upsample for metrics. S1-only (S0 is invariant to --da-nx, so it
# is identical to the da_nx=16 run and skipped here).
LAGS=(1.0 2.0)
LAG=${LAGS[$SLURM_ARRAY_TASK_ID]}

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --da-nx 32 \
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
    --scenarios test_s1 \
    --outdir reports/outputs/qg_s1_da32_lag${LAG//./p} \
    --tag s1_da32