#!/bin/bash
#SBATCH --job-name=qg_psi_state_5w
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=qgps5w_%A_%a.out
#SBATCH --error=qgps5w_%A_%a.err

set -e

# 5-window psi_state DA ETKF, benchmarked apples-to-apples against the cached
# q-state (psi-obs) DA results:
#   task 0 = S0   (comparator: reports/qg/outputs/qg_s0_lag_sweep_psi, noise 0.05 default)
#   task 1 = S1 same-res da_nx=64 (comparator: reports/qg/outputs/qg_s1_nores_lag1p0, noise 0.01)
# Both use obs_var="psi_state" (index-lookup H into the upper-layer psi entries).
# Same-resolution guaranteed: da_nx == nx == 64 -> cross_res=False, psi_state guard passes.
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python

TASK=${SLURM_ARRAY_TASK_ID}

if [ "$TASK" = "0" ]; then
  # S0: mirror run_qg_s0_lag_sweep.sbatch (lag 1.0) at QGConfig default noise 0.05
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
      --init-lag-days-list 1.0 \
      --obs-var psi_state \
      --scenarios test_s0 \
      --outdir reports/qg/outputs/qg_s0_psi_state_lag1p0 \
      --tag s0_psi_state
else
  # S1 same-res (da_nx=64): mirror run_qg_s1_nores.sbatch, noise 0.01
  $PY evaluation/sweep_qg_baselines.py \
      --nx 64 \
      --da-nx 64 \
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
      --scenarios test_s1 \
      --outdir reports/qg/outputs/qg_s1_nores_psi_state_lag1p0 \
      --tag s1_nores_psi_state
fi
