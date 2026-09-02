#!/bin/bash
#SBATCH --job-name=qg_ff_obs_sweep2
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

# Stage B2: low-noise obs-enrichment sweep at band-center lag FF_LAG (set from
# Stage A where free-forecast EV ~ 0.7). After fixing the run() config-rebuild
# bug that silently dropped obs_noise_std_frac (so the Stage B noise sweep was
# pinned to the 0.05 default), re-test whether genuinely low obs noise lets DA
# beat the free forecast (forecast_improvement > 1). obs_std = frac * q1_std
# (~2.6e-5 at nx=64), vs free-forecast rmse ~7.3e-6: frac 1e-3 => obs noisier
# than the background, frac <~3e-4 makes obs cleaner than the background.

FF_LAG=${FF_LAG:-1.0}

# 4 cols-per-day x 3 obs-noise fracs
COLS=(4 8 16 32 4 8 16 32 4 8 16 32)
NOISES=(0.05 0.05 0.05 0.05 0.01 0.01 0.01 0.01 0.001 0.001 0.001 0.001)
C=${COLS[$SLURM_ARRAY_TASK_ID]}
NZ=${NOISES[$SLURM_ARRAY_TASK_ID]}

ODIR=reports/qg/outputs/qg_ff_obs_sweep2
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
    --tag ffsweep2
