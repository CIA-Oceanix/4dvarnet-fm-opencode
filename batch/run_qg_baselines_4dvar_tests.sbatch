#!/bin/bash
#SBATCH --job-name=qg_4dvar_tests
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:30:00
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --nodes=1

set -e

echo "============================================"
echo " QG 4D-Var baseline test gate | $(date)"
echo "============================================"

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python -m pytest \
    tests/test_qg_dynamics.py \
    tests/test_qg_data.py \
    tests/test_qg_baselines.py \
    tests/test_qg_baselines_4dvar.py \
    tests/test_qg_s0s1.py \
    tests/test_qg_random_columns.py \
    tests/test_qg1l_dynamics.py \
    -m "not slow" -q --tb=short

echo "============================================"
echo " Tests completed at $(date)"
echo "============================================"
