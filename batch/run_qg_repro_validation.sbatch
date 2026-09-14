#!/bin/bash
#SBATCH --job-name=qg_repro_val
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

# Reproducibility validation on the integration branch (feature/qg-da-integration,
# rebased onto current origin/master): PR #156 changed the random-columns obs
# geometry (per-column independent intra-day timing vs the old
# constellation-style simultaneous-columns events the archived reference
# numbers were computed under -- see PLAN.md's "Obs-protocol repro note").
# Re-runs the exact S0 reference case (nx=64, c4, psi-obs, lag=1.0, N=80,
# loc=6.0, LBFGS w=60) for all 4 methods to see how much the numbers shift.
mkdir -p reports/qg/outputs/qg_repro_validation

COMMON="--nx 64 --num-windows 5 --window-days 30 --spinup-years 2.0 \
    --geometry random_columns --cols-per-day 4 --obs-var psi \
    --obs-noise-frac 0.01 --init-lag-days 1.0 --scenarios test_s0"

case $SLURM_ARRAY_TASK_ID in
0)
  /Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/run_qg_baselines.py $COMMON \
      --method-list etkf --ensemble 80 --inflation 1.0 --loc-radius 6.0 \
      --out reports/qg/outputs/qg_repro_validation/etkf.json
  ;;
1)
  /Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/run_qg_baselines.py $COMMON \
      --method-list enkf --ensemble 80 --inflation 1.0 --loc-radius 6.0 \
      --out reports/qg/outputs/qg_repro_validation/enkf.json
  ;;
2)
  /Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/run_qg_baselines.py $COMMON \
      --method-list strong4dvar \
      --da-window-steps 60 --fourdvar-optimizer lbfgs --fourdvar-max-iter 60 \
      --fourdvar-lr 1.0 --b-var-scale 1.0 \
      --out reports/qg/outputs/qg_repro_validation/strong4dvar.json
  ;;
3)
  /Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/run_qg_baselines.py $COMMON \
      --method-list weak4dvar \
      --da-window-steps 60 --fourdvar-optimizer lbfgs --fourdvar-max-iter 60 \
      --fourdvar-lr 1.0 --b-var-scale 1.0 --q-var-scale 0.1 \
      --out reports/qg/outputs/qg_repro_validation/weak4dvar.json
  ;;
esac
