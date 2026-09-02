#!/bin/bash
#SBATCH --job-name=qg_nz1pct_c8
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

# Post-ridge-fix probe: S0 ETKF at 1% obs noise, nx=64, reference lag (EV_free~0.7),
# with 8 random meridional columns per day (doubles obs from the cols=4 run).
# task 0 = q-obs, task 1 = psi-obs
OBSVARS=(q psi)
OV=${OBSVARS[$SLURM_ARRAY_TASK_ID]}

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 30 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list etkf \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 8 \
    --obs-noise-frac-list 0.01 \
    --init-lag-days-list 1.0 \
    --band 0.25 \
    --obs-var ${OV} \
    --scenarios test_s0 \
    --outdir reports/qg/outputs/qg_nz1pct_c8_${OV} \
    --tag nz1pctc8
