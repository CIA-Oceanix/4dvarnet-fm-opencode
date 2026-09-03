#!/bin/bash
#SBATCH --job-name=qg_s1_psid32
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=qgs1pd32_%A_%a.out
#SBATCH --error=qgs1pd32_%A_%a.err

set -e

# S1 cross-resolution psi-state case at da_nx=32: DA model at 32x32 vs truth
# at 64x64 (ratio 2:1), obs_var='psi_state' (the DA state is the
# streamfunction; the H-mode operator spectrally upsamples the psi-state to
# the obs grid before column selection), cols=4, 1% obs noise, lag 1.0/2.0.
# Same seams as the da_nx=32 q-state+psi-obs reference: param bias (rd,rek
# *0.85) + corrupted wind (S1), init downsampling, upsampled H-function,
# physical-coordinate localization, trajectory upsample for metrics.
# S1-only (S0 is invariant to --da-nx, so it is identical to the same-res run
# and skipped here, matching qg_s1_da32.sbatch). Comparable to the cached
# q-state+psi-obs reference qg_s1_da32_lag1p0 (EV_full 0.3398).
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
    --obs-var psi_state \
    --scenarios test_s1 \
    --outdir reports/qg/outputs/qg_s1_psi_state_da32_lag${LAG//./p} \
    --tag s1_psi_state_da32
