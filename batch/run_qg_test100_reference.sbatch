#!/bin/bash
#SBATCH --job-name=qg_test100_ref
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=batch/logs/qg_test100_ref_%A_%a.out
#SBATCH --error=batch/logs/qg_test100_ref_%A_%a.err

set -e

mkdir -p reports/qg/outputs/qg_test100_reference

CACHE_DIR="$PWD/reports/qg/outputs/qg_windows_1000_100_100/cache"
COMMON="--nx 64 --num-windows 100 --window-days 30 --spinup-years 2.0 \
    --seed 20042 --cache-dir $CACHE_DIR \
    --geometry random_columns --cols-per-day 4 --obs-var psi \
    --obs-noise-frac 0.01 --init-lag-days 1.0 --scenarios test_s0"

PY=/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python

case $SLURM_ARRAY_TASK_ID in
0)
  $PY evaluation/run_qg_baselines.py $COMMON \
      --method-list etkf --ensemble 80 --inflation 1.0 --loc-radius 6.0 \
      --out reports/qg/outputs/qg_test100_reference/etkf.json
  ;;
1)
  $PY evaluation/run_qg_baselines.py $COMMON \
      --method-list enkf --ensemble 80 --inflation 1.0 --loc-radius 6.0 \
      --out reports/qg/outputs/qg_test100_reference/enkf.json
  ;;
2)
  $PY evaluation/run_qg_baselines.py $COMMON \
      --method-list strong4dvar \
      --da-window-steps 60 --fourdvar-optimizer lbfgs --fourdvar-max-iter 60 \
      --fourdvar-lr 1.0 --b-var-scale 1.0 \
      --out reports/qg/outputs/qg_test100_reference/strong4dvar.json
  ;;
3)
  $PY evaluation/run_qg_baselines.py $COMMON \
      --method-list weak4dvar \
      --da-window-steps 60 --fourdvar-optimizer lbfgs --fourdvar-max-iter 60 \
      --fourdvar-lr 1.0 --b-var-scale 1.0 --q-var-scale 0.1 \
      --out reports/qg/outputs/qg_test100_reference/weak4dvar.json
  ;;
esac
