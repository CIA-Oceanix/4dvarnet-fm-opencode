#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=sw-debug3
#SBATCH --output=sbatch_logs/sw_debug3_%j.out
#SBATCH --error=sbatch_logs/sw_debug3_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

python -c "
import torch, numpy as np
from models.shallow_water_dynamics import ShallowWaterDynamics

dyn = ShallowWaterDynamics(Nx=8, Ny=64)
sd = dyn.state_dim
print(f'state_dim={sd}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dyn.to(device)

# Simulate an ensemble initialization
rng = np.random.RandomState(42)
N = 30
x0 = torch.zeros(N, sd, device=device)
x0[:, :512] = 1.0  # h1 baseline
x0[:, 512:1024] = 0.0  # u1 baseline
x0[:, 1024:1536] = 0.0  # v1 baseline
x0[:, 1536:2048] = 1.0  # h2 baseline
x0[:, 2048:2560] = 0.0  # u2 baseline
x0[:, 2560:] = 0.0  # v2 baseline
# Add small noise
x0 += torch.randn(N, sd, device=device) * 0.05

# Generate forcing
forcing = torch.randn(50, 2, device=device) * 0.1

print(f'Step 0: finite={x0.isfinite().all().item()}')
for t in range(1, 50):
    W = forcing[t-1]
    x0 = dyn.step(x0, W)
    if not x0.isfinite().all():
        nan_mask = torch.isnan(x0).any(dim=-1)
        print(f'Step {t}: NaN in {nan_mask.sum().item()}/{N} members')
        print(f'  max abs = {x0[nan_mask].abs().max().item() if nan_mask.any() else 0:.4f}')
        print(f'  min h1 = {x0[nan_mask, :512].min().item() if nan_mask.any() else 1:.4f}')
        break
    if t % 10 == 0:
        print(f'Step {t}: finite=True, h1 range=[{x0[:,:512].min():.4f},{x0[:,:512].max():.4f}]')

print('All finite' if x0.isfinite().all() else 'NaN found')
" 2>&1

echo "============================================"