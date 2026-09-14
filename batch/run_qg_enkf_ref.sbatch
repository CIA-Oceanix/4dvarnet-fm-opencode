#!/bin/bash
#SBATCH --job-name=qg_enkf_ref
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --nodes=1

set -e

# EnKF at the reference-case settings (mirror of run_qg_matrix_metrics.sbatch,
# method-list=enkf): nx=64, c4, random_columns, psi-obs, lag 1.0 & 2.0, to add
# the EnKF row to qg_matrix_c4_psi alongside the existing ETKF rows.

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 30 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list enkf \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 4 \
    --obs-noise-frac-list 0.01 \
    --init-lag-days-list 1.0,2.0 \
    --band 0.25 \
    --obs-var psi \
    --scenarios test_s0 \
    --outdir reports/qg/outputs/qg_matrix_c4_psi \
    --tag qgmatrix
