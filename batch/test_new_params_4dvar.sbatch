#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=sw-new-4dvar
#SBATCH --output=sbatch_logs/sw_new_4dvar_%j.out
#SBATCH --error=sbatch_logs/sw_new_4dvar_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

# Quick Strong-4DVar test with new parameters (Nx=16, 1 window)
python -c "
import torch, sys, os
sys.path.insert(0, '.')
from data.shallow_water import ShallowWaterConfig, make_sw_obs_indices, make_sw_obs_noise_std, _generate_sw_observations
from models.shallow_water_dynamics import ShallowWaterDynamics
from evaluation.baselines import ObsOperator
from evaluation.baselines import Strong4DVar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

cfg = ShallowWaterConfig(Nx=16, Ny=16)
dyn = ShallowWaterDynamics(Nx=cfg.Nx, Ny=cfg.Ny, dt=cfg.dt, tau0=cfg.tau0,
    f_cor=cfg.f_cor, g1=cfg.g1, g2=cfg.g2, coupling=cfg.coupling,
    friction=cfg.friction, viscosity=cfg.viscosity)

traj, forcing = dyn.generate_full_trajectory(num_steps=200, seed=123, spinup_steps=500)
traj = traj.to(device)
forcing = forcing.to(device)

obs_indices = make_sw_obs_indices(cfg).to(device)
obs_operator = get_obs_operator(cfg.state_dim, obs_indices)
noise_std = make_sw_obs_noise_std(cfg).to(device)

# Single window: first 50 steps
K = 50
true_seg = traj[:K]
frc_seg = forcing[:K]
obs_seg, obs_mask = _generate_sw_observations(true_seg.cpu(), obs_indices.cpu(), noise_std.cpu(), seed=42)
obs_seg = obs_seg.to(device)
obs_mask = obs_mask.to(device)

obs_vals = obs_seg[:, obs_indices]
tmask = torch.ones(K, dtype=torch.bool, device=device)

# Background variance: matching h1 variance (0.134^2 ≈ 0.018)
B_var = 0.018

# Observation noise variance: mean of per-obs noise variances
R_var = noise_std.pow(2).mean().item()
print(f'B_var={B_var}, R_var={R_var:.6e}')

# R_var as per-component Tensor
R_var_vec = noise_std.pow(2).to(device)

for da_steps, label in [(50, '50 steps'), (100, '100 steps'), (200, '200 steps')]:
    print(f'\n=== Strong-4DVar: da_window_steps={da_steps} ({label}) ===')
    method = Strong4DVar(
        da_window_steps=da_steps,
        B_var=B_var,
        R_var=R_var_vec,
        max_iter=200,
        lr=0.01,
        dt=cfg.dt,
        device=device,
        dynamics=dyn,
        obs_operator=obs_operator,
        noise_init_std=0.05,
        opt_steps=200,
    )
    result = method.assimilate(obs_vals, tmask, frc_seg, true_seg)
    analyzed = torch.tensor(result.trajectory, device=device)
    
    # Per-component RMSE
    Nxy = cfg.Nx * cfg.Ny
    names = ['h1','u1','v1','h2','u2','v2']
    for i, name in enumerate(names):
        seg = true_seg[:, i*Nxy:(i+1)*Nxy]
        res = analyzed[:, i*Nxy:(i+1)*Nxy]
        rmse = torch.sqrt((seg - res).pow(2).mean()).item()
        clim_std = seg.std().item()
        explained_var = 1.0 - (rmse**2) / (clim_std**2 + 1e-12)
        print(f'  {name}: RMSE={rmse:.4f}, clim_std={clim_std:.4f}, EV={explained_var:.3f}')
    overall_rmse = torch.sqrt((true_seg - analyzed).pow(2).mean()).item()
    print(f'  Overall RMSE: {overall_rmse:.4f}')
" 2>&1