#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=sw-debug4
#SBATCH --output=sbatch_logs/sw_debug4_%j.out
#SBATCH --error=sbatch_logs/sw_debug4_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

python -c "
import torch, numpy as np
from models.shallow_water_dynamics import ShallowWaterDynamics
from evaluation.baselines import ObsOperator

dyn = ShallowWaterDynamics(Nx=8, Ny=64)
sd = dyn.state_dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dyn.to(device)
N = 30
N1 = N - 1

# Build obs operator (sparse)
from data.shallow_water import make_sw_obs_indices
from data.shallow_water import ShallowWaterConfig
config = ShallowWaterConfig(Nx=8, Ny=64)
obs_indices = make_sw_obs_indices(config)
H = ObsOperator(sd, obs_indices)
H.to(device)
od = H.obs_dim
print(f'sd={sd}, od={od}')

rng = np.random.RandomState(42)
x0 = torch.zeros(N, sd, device=device)
x0[:, :512] = 1.0
x0[:, 512:1024] = 0.0
x0[:, 1024:1536] = 0.0
x0[:, 1536:2048] = 1.0
x0[:, 2048:2560] = 0.0
x0[:, 2560:] = 0.0
x0 += torch.randn(N, sd, device=device) * 0.05
forcing = torch.randn(50, 2, device=device) * 0.1

R_sym_sqrt_inv = 1.0 / np.sqrt(0.5)

for t in range(1, 50):
    W = forcing[t-1]
    ens = dyn.step(x0, W)
    
    if not ens.isfinite().all():
        print(f'Step {t}: NaN after dynamics only!')
        break
    
    # ETKF analysis
    mu = torch.mean(ens, dim=0)
    A = ens - mu
    mu_obs = H(mu)
    HA = H(ens) - mu_obs.unsqueeze(0)
    HA_w = HA * R_sym_sqrt_inv
    
    if not HA_w.isfinite().all():
        print(f'Step {t}: NaN in HA_w after dynamics')
        break
    
    try:
        U, s, Vt = torch.linalg.svd(HA_w, full_matrices=False)
        s2 = s ** 2
        d = s2 + N1
        Pw = U @ torch.diag(1.0 / d) @ U.T
        Tmat = U @ torch.diag(torch.sqrt(N1 / d)) @ U.T
        R_inv = 1.0 / 0.5
        dy = torch.zeros(od, device=device)
        w_vec = (dy * R_inv) @ HA.T @ Pw
        ens = mu + w_vec @ A + Tmat @ A
    except Exception as e:
        print(f'Step {t}: SVD/analysis error: {e}')
        print(f'  HA_w shape={HA_w.shape}, min={HA_w.min():.4f}, max={HA_w.max():.4f}')
        break
    
    # Inflation
    mu = torch.mean(ens, dim=0)
    ens = mu + 1.05 * (ens - mu)
    
    if not ens.isfinite().all():
        nan_mask = torch.isnan(ens).any(dim=-1)
        print(f'Step {t}: NaN after analysis+inflation, {nan_mask.sum().item()}/{N} members')
        print(f'  s values: [{s[0]:.4f} ... {s[-1]:.4f}], sum={s.sum():.4f}')
        print(f'  mu finite: {mu.isfinite().all().item()}')
        print(f'  w_vec finite: {torch.isfinite(w_vec).all().item()}')
        print(f'  Tmat finite: {Tmat.isfinite().all().item()}')
        # Check which members are NaN
        for n in range(min(5, N)):
            print(f'  member {n}: finite={ens[n].isfinite().all().item()}')
        break
    
    x0 = ens
    
    if t % 10 == 0:
        print(f'Step {t}: OK, h1=[{ens[:,:512].min():.4f},{ens[:,:512].max():.4f}]')

if ens.isfinite().all():
    print(f'All 50 steps OK')
" 2>&1

echo "============================================"