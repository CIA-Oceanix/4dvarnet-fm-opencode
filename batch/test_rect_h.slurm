#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/test_rect_h_%j.out
#SBATCH --error=experiments/test_rect_h_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

echo "============================================"
echo " Test: Rectangular H + J=2 S1  |  $(date)"
echo "============================================"

# Test 1: ObsOperator basics
echo ""
echo "--- Test 1: ObsOperator basics ---"
$PYTHON -c "
import torch
import numpy as np
from evaluation.baselines import ObsOperator, _safe_ref

H_id = ObsOperator(5)
x = torch.tensor([1.,2.,3.,4.,5.])
assert (H_id(x) == x).all()
assert H_id.obs_dim == 5

H_sel = ObsOperator(5, obs_indices=[0, 2, 4])
y = H_sel(x)
assert y.shape == (3,)
assert (y == torch.tensor([1.,3.,5.])).all()
assert H_sel.obs_dim == 3

X = torch.randn(10, 5)
y_batch = H_sel(X)
assert y_batch.shape == (10, 3)

full = H_sel.expand_to_state(y, 5)
assert full.shape == (5,)
assert (full[torch.tensor([0,2,4])] == y).all()
assert (full[torch.tensor([1,3])] == 0).all()

ref = np.random.randn(10, 5)
analysis = np.random.randn(10, 5)
result = _safe_ref(ref, analysis, H_id)
assert result.shape == (10, 5)

analysis_sel = np.random.randn(10, 3)
result = _safe_ref(ref, analysis_sel, H_sel)
assert result.shape == (10, 3)

print('All ObsOperator tests passed!')
"
echo ""

# Test 2: Identity H regression (no rectangular H)
echo "--- Test 2: Identity H regression (Strong-4DVar on 5 windows) ---"
$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "test_id_regression" \
    --param-bias 0.0 \
    --forcing-state-bias 0.0 \
    --num-windows 5 \
    --da-window-steps 500 \
    --skip-weak

echo ""
echo "Test 2 completed (check RMSE values above)"
echo ""

# Test 3: Rectangular H with S0 (J=4, obs 2/4 fast vars) + S1 (J=2)
echo "--- Test 3: Rectangular H with J=2 S1 (5 windows) ---"
$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "test_rect_h_j2" \
    --param-bias 0.0 \
    --forcing-state-bias 0.0 \
    --num-windows 5 \
    --da-window-steps 500 \
    --skip-weak \
    --obs-j 2 \
    --s1-j 2

echo ""
echo "Test 3 completed (check RMSE values above)"
echo ""

echo "============================================"
echo " All tests completed at $(date)"
echo "============================================"