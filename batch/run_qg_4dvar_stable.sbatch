#!/bin/bash
#SBATCH --job-name=qg_4dvar_stable
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:20:00
#SBATCH --array=0-2
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --nodes=1

set -e

# Stability-focused Strong-4DVar tuning at nx=64 (c4, psi, lag1.0, single
# 30-day window). The first tune array diverged (EV ~ -3e5..-8e5) with high-lr
# Adam; here use the canonical robust solver (LBFGS) plus low-lr Adam with a
# tight gradient clip.
mkdir -p reports/qg/outputs/qg_tune_4dvar

# task0: LBFGS, 1-day window
# task1: LBFGS, 5-day window (fewer cycled windows -> faster)
# task2: Adam, low lr, tight grad clip
OPT=(lbfgs lbfgs adam)
W=(12 60 12)
IT=(60 60 40)
S=(0 0 200)
LR=(1.0 1.0 0.01)
B=(1.0 1.0 1.0)
CLIP=(1000 1000 10)

T=${SLURM_ARRAY_TASK_ID}
OUT=reports/qg/outputs/qg_tune_4dvar/stable_${OPT[$T]}_w${W[$T]}_it${IT[$T]}_lr${LR[$T]}_b${B[$T]}_c${CLIP[$T]}.json

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/run_qg_baselines.py \
    --method-list strong4dvar \
    --nx 64 \
    --num-windows 1 \
    --window-days 30 \
    --spinup-years 2.0 \
    --geometry random_columns \
    --cols-per-day 4 \
    --obs-var psi \
    --init-lag-days 1.0 \
    --band 0.25 \
    --scenarios test_s0 \
    --device cuda \
    --da-window-steps ${W[$T]} \
    --fourdvar-optimizer ${OPT[$T]} \
    --fourdvar-max-iter ${IT[$T]} \
    --fourdvar-opt-steps ${S[$T]} \
    --fourdvar-lr ${LR[$T]} \
    --b-var-scale ${B[$T]} \
    --fourdvar-grad-clip ${CLIP[$T]} \
    --out ${OUT}
