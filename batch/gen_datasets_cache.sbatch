#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --job-name=gen_cache
#SBATCH --output=sbatch_logs/gen_cache_%j.out
#SBATCH --error=sbatch_logs/gen_cache_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "Generating datasets cache..."
python3 -c "
import sys, os, time
sys.path.insert(0, os.getcwd())
import torch
from data.lorenz63 import Lorenz63Config, make_mixed_datasets

device = torch.device('cuda')
base_cfg = Lorenz63Config(
    dt=0.01, T_max=3.0, obs_interval=20,
    R_var=0.5, B_var=2.0,
    num_windows=2000, window_spacing=2000,
    spinup_steps=10000, seed=42,
)

t0 = time.time()
datasets = make_mixed_datasets(
    base_cfg, num_test_windows=200,
    include_randparam_test=True, param_noise=0.2,
)
print(f'Generated datasets in {time.time()-t0:.1f}s')

cache = 'experiments/datasets.pt'
torch.save(datasets, cache)
print(f'Saved to {cache}')
"
echo "Done."
