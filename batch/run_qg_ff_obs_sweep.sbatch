#!/bin/bash
#SBATCH --job-name=qg_ff_obs_sweep
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

# Stage B: obs-enrichment sweep at the chosen band-center lag (FF_LAG below,
# set from Stage A where free-forecast EV ~ 0.7). Each array task is one obs
# config: cols-per-day (daily coverage 6.25%..50%) x obs_noise_std_frac.
# Reports DA expvar_full vs free-forecast expvar_free; we seek a config where
# DA beats the free forecast (forecast_improvement > 1).

# --- Set this to the lag_param chosen from Stage A (EV_free ~ 0.7) ---
FF_LAG=${FF_LAG:-1.0}

COLS=(4 8 16 32 4 8 16 32)
NOISES=(0.05 0.05 0.05 0.05 0.02 0.02 0.02 0.02)
C=${COLS[$SLURM_ARRAY_TASK_ID]}
NZ=${NOISES[$SLURM_ARRAY_TASK_ID]}

ODIR=reports/outputs/qg_ff_obs_sweep
mkdir -p ${ODIR}

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
    --cols-per-day ${C} \
    --obs-noise-frac-list ${NZ} \
    --init-lag-days-list ${FF_LAG} \
    --band 0.25 \
    --obs-var q \
    --scenarios test_s0 \
    --outdir ${ODIR} \
    --tag ffsweep
