import os, sys
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz63 import Lorenz63Config
from data.random_param_dataset import RandomParamLorenz63Dataset
from evaluation.baselines import (
    Weak4DVar, Strong4DVar, EnKF, ETKF,
    JointWeak4DVar, JointStrong4DVar, JointEnKF, JointETKF,
)
from evaluation.metrics import rmse, param_rmse
from evaluation.run import evaluate_baseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

N = 50
batch_size = 200

weak_cfg = {"opt_steps": 200, "Q_var": 1.0}
weak_cfg_joint = {**weak_cfg, "P_var": 1.0}
strong_cfg = {"max_iter": 50}
strong_cfg_joint = {**strong_cfg, "P_var": 1.0}
enkf_cfg = {"N_ensemble": 30, "inflation": 1.0}
etkf_cfg = {"N_ensemble": 30, "inflation": 1.0}

cases = [
    ("CS3", Lorenz63Config(case=1, param_bias=0.0, T_max=3.0, seed=125, num_windows=100)),
    ("CS4", Lorenz63Config(case=2, param_bias=0.15, forcing_state_bias=0.15,
                            forcing_coupling="quartic", T_max=3.0, seed=126, num_windows=100)),
]

datasets = {}
for case_name, cfg in cases:
    dd = RandomParamLorenz63Dataset(cfg, param_noise=0.2)
    datasets[case_name] = dd
    print(f"{case_name}: {len(dd)} windows")

method_configs = {
    "Weak-4DVar":      (lambda ct: Weak4DVar(dt=0.01, da_window_steps=N, device=device, coupling_type=ct, **weak_cfg)),
    "Joint-Weak-4DVar": (lambda ct: JointWeak4DVar(dt=0.01, da_window_steps=N, device=device, coupling_type=ct, **weak_cfg_joint)),
    "Strong-4DVar":    (lambda ct: Strong4DVar(dt=0.01, da_window_steps=N, device=device, coupling_type=ct, **strong_cfg)),
    "Joint-Strong-4DVar": (lambda ct: JointStrong4DVar(dt=0.01, da_window_steps=N, device=device, coupling_type=ct, **strong_cfg_joint)),
    "EnKF":            (lambda ct: EnKF(dt=0.01, device=device, coupling_type=ct, **enkf_cfg)),
    "Joint-EnKF":      (lambda ct: JointEnKF(dt=0.01, device=device, coupling_type=ct, **enkf_cfg)),
    "ETKF":            (lambda ct: ETKF(dt=0.01, device=device, coupling_type=ct, **etkf_cfg)),
    "Joint-ETKF":      (lambda ct: JointETKF(dt=0.01, device=device, coupling_type=ct, **etkf_cfg)),
}

def get_coupling(cfg):
    return cfg.forcing_coupling if cfg.forcing_coupling == "quartic" else "linear"

header = f"{'Case':<6} {'Method':<20} {'State RMSE (X/Y/Z = mean)':<40} {'Param RMSE (s/r/b)':<30} {'Ratio':<10}"
sep = "=" * len(header)
print(f"\n{sep}")
print(header)
print(sep)

for case_name, cfg in cases:
    ds = datasets[case_name]
    coupling = get_coupling(cfg)

    true_state_list = [w["true_state"].numpy() for w in ds]
    true_params_list = [np.array([w["sigma"], w["rho"], w["beta"]]) for w in ds]

    vanilla_results = {}
    for base_name in ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]:
        m = method_configs[base_name](coupling)
        v_stats, v_results = evaluate_baseline(
            m, ds, cfg, device, return_trajs=True, batch_size=batch_size)
        vanilla_results[base_name] = (v_stats, v_results)

    for method_name, make_method in method_configs.items():
        method = make_method(coupling)
        stats, results = evaluate_baseline(
            method, ds, cfg, device, return_trajs=True, batch_size=batch_size)
        mean_rmse = stats[0]

        state_rmse_str = f"{mean_rmse[0]:.4f}/{mean_rmse[1]:.4f}/{mean_rmse[2]:.4f} = {np.mean(mean_rmse):.4f}"

        is_joint = method_name.startswith("Joint-")
        if is_joint and results[0].params is not None:
            all_pred_params = np.stack([r.params for r in results], axis=0)
            num_steps_i = all_pred_params.shape[1]
            all_true_params = np.repeat(np.stack(true_params_list, axis=0)[:, np.newaxis, :], num_steps_i, axis=1)
            prmse = param_rmse(all_pred_params.reshape(-1, 3), all_true_params.reshape(-1, 3))
            param_rmse_str = f"{prmse[0]:.4f}/{prmse[1]:.4f}/{prmse[2]:.4f}"
        else:
            param_rmse_str = "N/A"

        if is_joint:
            vanilla_name = method_name.replace("Joint-", "")
            v_mean = np.mean(vanilla_results[vanilla_name][0][0])
            j_mean = np.mean(mean_rmse)
            ratio = j_mean / v_mean if v_mean > 0 else 0
            ratio_str = f"{ratio:.4f}"
        else:
            ratio_str = "N/A"

        print(f"{case_name:<6} {method_name:<20} {state_rmse_str:<40} {param_rmse_str:<30} {ratio_str:<10}")

print(sep)
