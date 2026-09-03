#!/bin/bash
#SBATCH --job-name=qg_matrix_metrics
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

# S0 ETKF at 1% obs noise, nx=64, 2 init lags (1.0, 2.0) x 2 obs vars, capturing
# the new per-field (q/psi) per-layer (layer1/layer2/full) RMSE/EV metrics.
# 4 tasks: cols {4,8} x obsvar {q,psi}; each task builds its dataset once and runs
# lags 1.0 and 2.0.
COLS=(4 4 8 8)
OBSVARS=(q psi q psi)
C=${COLS[$SLURM_ARRAY_TASK_ID]}
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
    --cols-per-day ${C} \
    --obs-noise-frac-list 0.01 \
    --init-lag-days-list 1.0,2.0 \
    --band 0.25 \
    --obs-var ${OV} \
    --scenarios test_s0 \
    --outdir reports/qg/outputs/qg_matrix_c${C}_${OV} \
    --tag qgmatrix \
    --save-traj
