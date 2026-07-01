#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=4dvarnet-config
#SBATCH --output=sbatch_logs/config_val_%j.out
#SBATCH --error=sbatch_logs/config_val_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " 4DVarNet-FM Config Validation | $(date)"
echo "============================================"
echo ""

python -c "
import hydra, torch, sys
sys.path.insert(0, '.')
from train import model_factory
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM
from models.solver import TweedieSolver

EXPS = [
    'E1_direct_unet_default', 'E2_direct_unet_small', 'E3_direct_unet_rand',
    'F1_vanilla_cfm_default', 'F2_vanilla_cfm_small', 'F3_vanilla_cfm_rand',
    'G1_vanilla_cfm_t0_default', 'G2_vanilla_cfm_t0_small', 'G3_vanilla_cfm_t0_rand',
]

for exp in EXPS:
    with hydra.initialize(config_path='config'):
        cfg = hydra.compose('experiment/' + exp)
    model = model_factory(cfg, torch.device('cpu'))
    if exp.startswith('E'):
        expected = DirectUNet
    elif exp.startswith(('F', 'G')):
        expected = VanillaCFM
    else:
        raise ValueError(f'Unknown prefix: {exp}')
    # For G experiments, verify tau flag is set
    if exp.startswith('G'):
        assert model.train_tau_0_only, f'{exp}: expected train_tau_0_only=True'
        print(f'  OK  {exp}  ->  {type(model).__name__} (tau=0 mode)')
    else:
        assert isinstance(model, expected), f'{exp}: expected {expected.__name__}, got {type(model).__name__}'
        print(f'  OK  {exp}  ->  {type(model).__name__}')

# Also verify the base config loads and produces TweedieSolver
with hydra.initialize(config_path='config'):
    base_cfg = hydra.compose('lorenz63_default')
model = model_factory(base_cfg, torch.device('cpu'))
assert isinstance(model, TweedieSolver), f'base: expected TweedieSolver, got {type(model).__name__}'
print(f'  OK  lorenz63_default  ->  TweedieSolver')
print()
print('All 10 configs validated successfully.')
"

echo ""
echo "============================================"
echo " Config validation passed at $(date)"
echo "============================================"
