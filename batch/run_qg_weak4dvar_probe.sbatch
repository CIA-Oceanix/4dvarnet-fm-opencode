#!/bin/bash
#SBATCH --job-name=qg_w4dvar_probe
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --nodes=1

set -e

# Weak-4DVar sizing probe at the reference scale (nx=64, c4, psi, lag1.0) on a
# single 30-day eval window. Weak control is sd*(win+1) params; gauge wall-clock
# and finiteness before the full reference run. Sweep a small window+optimizer
# grid to see which is affordable / converges.
mkdir -p reports/qg/outputs/qg_tune_4dvar

# task0: Adam, 1-day window, short budget (baseline feel)
# task1: LBFGS, 1-day window, moderate iter
# task2: LBFGS, 5-day window (fewer cycled windows -> faster)
OPT=(adam lbfgs lbfgs)
W=(12 12 60)
IT=(60 60 60)
S=(100 0 0)
LR=(0.05 1.0 1.0)
B=(1.0 1.0 1.0)
Q=(0.1 0.1 0.1)
CLIP=(1000 1000 1000)

T=${SLURM_ARRAY_TASK_ID}
OUT=reports/qg/outputs/qg_tune_4dvar/weakprobe_${OPT[$T]}_w${W[$T]}_it${IT[$T]}_lr${LR[$T]}_b${B[$T]}_q${Q[$T]}_c${CLIP[$T]}.json

start=$(date +%s)
/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/run_qg_baselines.py \
    --method-list weak4dvar \
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
    --q-var-scale ${Q[$T]} \
    --fourdvar-grad-clip ${CLIP[$T]} \
    --out ${OUT}
end=$(date +%s)
echo "PROBE_WALL_SECONDS=$((end - start))"
