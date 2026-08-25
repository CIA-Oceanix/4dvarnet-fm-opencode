#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=sw-debug
#SBATCH --output=sbatch_logs/sw_debug_%j.out
#SBATCH --error=sbatch_logs/sw_debug_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " SW Debug | $(date)"
echo "============================================"

python -c "
import torch, numpy as np
from data.shallow_water import ShallowWaterConfig, ShallowWaterDataset, make_sw_obs_indices
from evaluation.baselines import ETKF, EnKF, ObsOperator
from evaluation.run_sw import _create_sw_dynamics, _extract_obs_values
from evaluation.metrics import compute_sw_component_metrics

config = ShallowWaterConfig(Nx=8, Ny=64)
ds_config = ShallowWaterConfig(**{**config.__dict__, 'K': 10, 'num_windows': 1})
ds = ShallowWaterDataset(ds_config, scenario='S0')
w = ds[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
obs_indices = make_sw_obs_indices(config)
obs_operator = ObsOperator(config.state_dim, obs_indices)
dyn = _create_sw_dynamics(ds_config, scenario='S0')

print(f'state_dim={config.state_dim}, obs_dim={obs_operator.obs_dim}')
print(f'window K={ds_config.K}')

etkf = ETKF(dt=config.dt, device=device, dynamics=dyn, obs_operator=obs_operator, inflation=1.05, noise_init_std=0.05)
enkf = EnKF(dt=config.dt, device=device, dynamics=dyn, obs_operator=obs_operator, inflation=1.05, noise_init_std=0.05)

obs_vals = _extract_obs_values(w['obs'].unsqueeze(0), obs_indices).squeeze(0)
tmask = torch.ones(ds_config.K, dtype=torch.bool, device=device)
force = w['forcing'].to(device)
truth = w['true_state']

# Test ETKF
result = etkf.assimilate(obs_vals.to(device), tmask, force.to(device), truth.to(device))
ana = result.trajectory
print(f'ETKF analysis shape: {ana.shape}, finite: {np.isfinite(ana).all()}')
print(f'  ana[0,:8]: {ana[0,:8]}')
print(f'  truth[0,:8]: {truth[:8].numpy()}')

comp = compute_sw_component_metrics(ana, truth.numpy(), config.Nx, config.Ny)
print(f'ETKF overall RMSE={comp[\"overall\"][\"rmse\"]:.4f} EV={comp[\"overall\"][\"ev\"]:.4f}')
print(f'  ocean agg RMSE={comp[\"ocean\"][\"aggregate\"][\"rmse\"]:.4f} EV={comp[\"ocean\"][\"aggregate\"][\"ev\"]:.4f}')
"

echo "============================================"
echo " Done: $(date)"
echo "============================================"