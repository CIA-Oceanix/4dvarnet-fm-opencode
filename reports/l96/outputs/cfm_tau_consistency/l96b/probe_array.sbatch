#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --job-name=l96b-tau-probes
#SBATCH --output=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-tau-probe/experiments/l96b_probes/p_%a_%A.out
#SBATCH --error=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-tau-probe/experiments/l96b_probes/p_%a_%A.err
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-tau-probe
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"; mkdir -p "$MPLCONFIGDIR"
GRID=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt
RLAY=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt
CK=(/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix/experiments/L96B_predictstatecfm_monaiM_seed1 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix/experiments/L96B_predictstatecfm_monaiM_seed2 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix/experiments/L96B_predictstatecfm_monaiM_seed3 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix/experiments/L96B_predictstatecfm_monaiM_ep1200_seed1 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-bench-default/experiments/L96B_vanillacfm_monaiM_seed1 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-bench-default/experiments/L96B_vanillacfm_monaiM_seed2 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-bench-default/experiments/L96B_vanillacfm_monaiM_seed3 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix/experiments/L96B_vanillacfm_monaiM_ep1200_seed1 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix/experiments/L96B_predictstatecfm_monaiM_seed1 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix/experiments/L96B_predictstatecfm_monaiM_ep1200_seed1 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-bench-default/experiments/L96B_vanillacfm_monaiM_seed1 /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix/experiments/L96B_vanillacfm_monaiM_ep1200_seed1)
TAG=(psc400_s1 psc400_s2 psc400_s3 psc1200_s1 van400_s1 van400_s2 van400_s3 van1200_s1 psc400_s1 psc1200_s1 van400_s1 van1200_s1)
DSN=(grid grid grid grid grid grid grid grid rlay rlay rlay rlay)
i=$SLURM_ARRAY_TASK_ID; C=${CK[$i]}/checkpoints/stage1_best.ckpt; T=${TAG[$i]}_${DSN[$i]}
if [ ${DSN[$i]} = grid ]; then DS=$GRID; else DS=$RLAY; fi
python reports/l96/probe_ode_mean_along_path.py --checkpoint $C --dataset $DS --batch-size 50 --n-sweep 1,10 --out /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-tau-probe/experiments/l96b_probes/ode_path_$T.json
python reports/l96/probe_tau_consistency.py --checkpoint $C --dataset $DS --out /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-tau-probe/experiments/l96b_probes/tau_consistency_$T.json
python reports/l96/probe_tau_consistency_truth_marginal.py --checkpoint $C --dataset $DS --out /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-tau-probe/experiments/l96b_probes/ns1_$T.json
python reports/l96/probe_stochastic_sampler.py --checkpoint $C --dataset $DS --grid 10:0,10:0:0.5,20:0:0.5,80:0 --out /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-tau-probe/experiments/l96b_probes/sampler_$T.json
echo done
