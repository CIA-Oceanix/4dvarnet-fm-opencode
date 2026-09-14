#!/bin/bash
#SBATCH --job-name=qg_s0_psi_01
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=qgs0p01_%A.out
#SBATCH --error=qgs0p01_%A.err

set -e

# S0 psi-state DA ETKF at the SAME obs-noise level as the cached q-state
# (psi-obs) DA benchmark (noise 0.01), so the q-state vs psi-state S0 per-field
# comparison is apples-to-apples with full metrics_per_field (the default-noise
# 0.05 q-state reference qg_s0_lag_sweep_psi only stores the aggregate qall).
# 4 cols/day, lag 1.0, same-res (da_nx == nx == 64), ETKF N=80 infl 1.0 loc 6.
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python

$PY evaluation/sweep_qg_baselines.py \
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
    --cols-per-day 4 \
    --obs-noise-frac-list 0.01 \
    --init-lag-days-list 1.0 \
    --band 0.25 \
    --obs-var psi_state \
    --scenarios test_s0 \
    --outdir reports/qg/outputs/qg_s0_psi_state_nz0p01_lag1p0 \
    --tag s0_psi_state_nz0p01