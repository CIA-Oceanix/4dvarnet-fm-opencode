#!/bin/bash
#SBATCH --job-name=qg_ff_lag_calib
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

# Stage A: select the band-center lag_param where the shared-init FREE FORECAST
# reaches EV ~ 0.7. One task per lag_param in the calibration grid. Each task
# runs ETKF + free forecast from a single shared init (band half-width 0.25 d)
# and reports expvar_full (DA), expvar_free (free forecast) and improv.
LAGS=(0.5 1.0 1.5 2.0 2.5)
LAG=${LAGS[$SLURM_ARRAY_TASK_ID]}

mkdir -p reports/qg/outputs/qg_ff_lag_calib_q

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 30 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble-list 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list etkf \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 4 \
    --init-lag-days-list ${LAG} \
    --band 0.25 \
    --obs-var q \
    --scenarios test_s0 \
    --outdir reports/qg/outputs/qg_ff_lag_calib_q \
    --tag ffcal
