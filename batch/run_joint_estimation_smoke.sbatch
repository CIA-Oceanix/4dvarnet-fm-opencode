#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=4dvarnet-joint-smoke
#SBATCH --output=sbatch_logs/joint_smoke_%j.out
#SBATCH --error=sbatch_logs/joint_smoke_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " 4DVarNet-FM Joint Estimation Smoke Test | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv -i 0)"
echo ""

# === Phase 1: Unit tests (CPU) ===
echo "--- Phase 1: Unit tests ---"
python -m pytest tests/test_joint_estimation.py -v -k "not slow" --tb=short
echo "Unit tests: PASSED"

# === Phase 2: Slow GPU tests ===
echo "--- Phase 2: Slow GPU tests ---"
python -m pytest tests/test_joint_estimation.py -v -k "slow" --tb=short
echo "Slow tests: PASSED"

# === Phase 3: Training smoke test (joint_cfm, 1 epoch) ===
echo "--- Phase 3: JointCFM training smoke (1 epoch) ---"
rm -f experiments/H1_joint_cfm_default/results.json
python train.py \
  --config-name experiment/H1_joint_cfm_default \
  training.stage1.epochs=1 \
  hydra.run.dir=. \
  hydra.output_subdir=null
echo "Training smoke: PASSED"

# === Phase 4: Run joint baselines (CS4 only, batch=200) ===
echo "--- Phase 4: Joint baselines CS4 smoke ---"
python -c "
import torch, sys, os
sys.path.insert(0, '.')
from data.lorenz63 import Lorenz63Config, Lorenz63Dataset
from evaluation.baselines import JointWeak4DVar, JointStrong4DVar, JointEnKF, JointETKF, Weak4DVar, Strong4DVar, EnKF, ETKF
from evaluation.run import evaluate_baseline
import numpy as np

device = torch.device('cuda')
print(f'Device: {device}')

cfg = Lorenz63Config(case=2, param_bias=0.15, forcing_state_bias=0.15,
                     forcing_coupling='quartic', T_max=3.0, seed=126,
                     num_windows=5)
ds = Lorenz63Dataset(cfg)
print(f'Dataset: {len(ds)} windows')

baselines = {
    'Weak-4DVar':        (Weak4DVar(da_window_steps=300, opt_steps=50, dt=0.01, device=device, coupling_type='quartic'),
                           JointWeak4DVar(da_window_steps=300, opt_steps=50, dt=0.01, device=device, coupling_type='quartic')),
    'Strong-4DVar':      (Strong4DVar(da_window_steps=300, max_iter=10, dt=0.01, device=device, coupling_type='quartic'),
                           JointStrong4DVar(da_window_steps=300, max_iter=10, dt=0.01, device=device, coupling_type='quartic')),
    'EnKF':              (EnKF(N_ensemble=30, dt=0.01, device=device, coupling_type='quartic'),
                           JointEnKF(N_ensemble=30, dt=0.01, device=device, coupling_type='quartic')),
    'ETKF':              (ETKF(N_ensemble=30, dt=0.01, device=device, coupling_type='quartic'),
                           JointETKF(N_ensemble=30, dt=0.01, device=device, coupling_type='quartic')),
}
for name, (vanilla, joint) in baselines.items():
    (m_v, _), _ = evaluate_baseline(vanilla, ds, cfg, device, return_trajs=True, batch_size=200)
    (m_j, _), _ = evaluate_baseline(joint, ds, cfg, device, return_trajs=True, batch_size=200)
    ratio = np.mean(m_j) / max(np.mean(m_v), 1e-10)
    status = 'OK' if ratio < 1.2 else 'WARN'
    print(f'  {name:20s} vanilla={np.mean(m_v):.4f} joint={np.mean(m_j):.4f} ratio={ratio:.3f} [{status}]')
print('Baseline smoke: DONE')
"

echo ""
echo "  Done at: $(date)"
echo "============================================"
