#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=bl-interp-s0s1
#SBATCH --output=sbatch_logs/bl_full_s0s1_%j.out
#SBATCH --error=sbatch_logs/bl_full_s0s1_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments

echo "============================================"
echo " Baseline S0/S1 Full Run | $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "  DWS=50, EnKF infl=2.0, ETKF infl=2.0, 200 windows"
echo "  Init: linear interpolation of sparse observations + noise"
echo "============================================"

python -c "
import os, sys, torch
sys.path.insert(0, '.')
from data.lorenz63 import Lorenz63Config, make_s0_s1_trainval
from evaluation.run import run_and_cache_baselines

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

base_cfg = Lorenz63Config()
datasets = make_s0_s1_trainval(
    base_cfg,
    num_train_windows=1,
    num_val_windows=1,
    num_test_windows=200,
    param_noise=0.2,
    bias_range=(0.0, 0.2),
)
print(f'test_s0: {len(datasets[\"test_s0\"])} windows')
print(f'test_s1: {len(datasets[\"test_s1\"])} windows')

results = run_and_cache_baselines(
    datasets, device,
    batch_size=50, da_window_steps=50,
    weak_config={'opt_steps': 150, 'lr': 0.02},
    strong_config={'max_iter': 40, 'lr': 0.1},
    enkf_config={'N_ensemble': 30, 'inflation': 2.0},
    etkf_config={'N_ensemble': 30, 'inflation': 2.0},
    suffix='_s0s1',
)

all_finite = True
for case in ['s0', 's1']:
    if case not in results: continue
    for method in ['Weak-4DVar', 'Strong-4DVar', 'EnKF', 'ETKF']:
        if method not in results[case]: continue
        m = results[case][method]['mean']
        for coord in ['X', 'Y', 'Z']:
            if not __import__('numpy').isfinite(results[case][method][coord]['mean']):
                print(f'  {case}/{method}/{coord}: NaN!')
                all_finite = False
        print(f'  {case}/{method:<20} mean={m:.4f}')

if all_finite:
    print('\nAll results finite!')
else:
    print('\nNaN detected!')
"

echo ""
echo "============================================"
echo " Done at $(date)"
echo "============================================"