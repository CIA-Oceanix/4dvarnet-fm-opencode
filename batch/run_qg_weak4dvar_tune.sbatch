#!/bin/bash
#SBATCH --job-name=qg_w4dvar_tune
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --nodes=1

set -e

# Weak-4DVar tuning at the reference scale (nx=64, c4, psi, lag1.0, single
# 30-day eval window). The probe found LBFGS over the 5-day window (w=60, lr=1.0)
# beats the free forecast (improv 1.15 / EV 0.864 at q=0.1). Sweep the model-error
# penalty q-var-scale to confirm the best value before the full reference run.
mkdir -p reports/qg/outputs/qg_tune_4dvar

# task0: q=0.01   task1: q=0.1   task2: q=1.0
Q=(0.01 0.1 1.0)

T=${SLURM_ARRAY_TASK_ID}
OUT=reports/qg/outputs/qg_tune_4dvar/weak_lbfgs_w60_lr1.0_b1.0_q${Q[$T]}.json

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
    --da-window-steps 60 \
    --fourdvar-optimizer lbfgs \
    --fourdvar-max-iter 60 \
    --fourdvar-lr 1.0 \
    --b-var-scale 1.0 \
    --q-var-scale ${Q[$T]} \
    --fourdvar-grad-clip 1000 \
    --out ${OUT}
end=$(date +%s)
echo "TUNE_WALL_SECONDS=$((end - start))"
