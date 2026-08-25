#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=sw-debug2
#SBATCH --output=sbatch_logs/sw_debug2_%j.out
#SBATCH --error=sbatch_logs/sw_debug2_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

python -c "
import torch, numpy as np
from data.shallow_water import ShallowWaterConfig, ShallowWaterDataset, make_sw_obs_indices
from evaluation.baselines import ETKF, EnKF, ObsOperator
from evaluation.run_sw import _create_sw_dynamics, _extract_obs_values

config = ShallowWaterConfig(Nx=8, Ny=64)
ds_config = ShallowWaterConfig(**{**config.__dict__, 'K': 10, 'num_windows': 1})
ds = ShallowWaterDataset(ds_config, scenario='S0')
w = ds[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
obs_indices = make_sw_obs_indices(config)
obs_operator = ObsOperator(config.state_dim, obs_indices)
dyn = _create_sw_dynamics(ds_config, scenario='S0')

obs_vals = _extract_obs_values(w['obs'].unsqueeze(0), obs_indices).squeeze(0)
tmask = torch.ones(ds_config.K, dtype=torch.bool, device=device)
force = w['forcing'].to(device)
truth = w['true_state']

# Manual ETKF loop to trace NaN
N = 30
sd = config.state_dim
H = obs_operator
od = H.obs_dim
N1 = N - 1
R_sym_sqrt_inv = 1.0 / np.sqrt(0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

interp_obs = obs_vals
ensemble = (obs_vals[0].clone().to(device) + torch.randn(sd, device=device) * 0.05).unsqueeze(0).repeat(N, 1)
noise = torch.randn_like(ensemble) * 0.05
noise[..., obs_indices] += torch.randn(N, od, device=device) * np.sqrt(0.5)
ensemble += noise

print(f'Initial ensemble: finite={ensemble.isfinite().all().item()}, min={ensemble.min():.4f}, max={ensemble.max():.4f}')
print(f'  h1 min={ensemble[:, :sd//6].min():.4f}, h1 mean={ensemble[:, :sd//6].mean():.4f}')

for t in range(1, 10):
    W = force[t - 1]
    ensemble = dyn.step(ensemble, W, **{})
    print(f'  step {t}: finite={ensemble.isfinite().all().item()}, min={ensemble.min():.4f}, max={ensemble.max():.4f}')
    if not ensemble.isfinite().all():
        nan_mask = torch.isnan(ensemble).any(dim=-1)
        print(f'    NaN members: {nan_mask.sum().item()}/{N}')
        break
    
    dy = obs_vals[t].to(device)
    mu = torch.mean(ensemble, dim=0)
    A = ensemble - mu
    mu_obs = H(mu)
    HA = H(ensemble) - mu_obs.unsqueeze(0)
    
    HA_w = torch.nan_to_num(HA * R_sym_sqrt_inv)
    try:
        U, s, Vt = torch.linalg.svd(HA_w, full_matrices=False)
    except:
        U, s, Vt = torch.linalg.svd(HA_w.cpu(), full_matrices=False)
        U, s, Vt = U.to(device), s.to(device), Vt.to(device)
    
    s2 = s ** 2
    d = s2 + N1
    Pw = U @ torch.diag(1.0 / d) @ U.T
    Tmat = U @ torch.diag(torch.sqrt(N1 / d)) @ U.T
    R_inv = 1.0 / 0.5
    w_vec = (dy * R_inv) @ HA.T @ Pw
    ensemble = mu + w_vec @ A + Tmat @ A
    
    mu = torch.mean(ensemble, dim=0)
    ensemble = mu + 1.05 * (ensemble - mu)
    
    if not ensemble.isfinite().all():
        nan_mask = torch.isnan(ensemble).any(dim=-1)
        print(f'    After analysis step {t}: NaN members: {nan_mask.sum().item()}/{N}')
        print(f'    s values: min={s.min():.6f}, max={s.max():.6f}, sum={s.sum():.6f}')
        print(f'    mu finite: {mu.isfinite().all().item()}')
        print(f'    w finite: {torch.isfinite(w_vec).all().item()}')
        print(f'    Tmat finite: {Tmat.isfinite().all().item()}')
        break

print('DONE')
"

echo "============================================"
echo " Debug done: $(date)"
echo "============================================"