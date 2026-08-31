import os
import sys
import json
import time
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz96 import Lorenz96Config
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF, ETKF, ObsOperator
from models.lorenz96_dynamics import Lorenz96Dynamics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)

_BASELINE_METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
_L96_SCALAR_PARAMS = ["F", "c1", "h", "hx", "eps"]
_L96_PARAMS = _L96_SCALAR_PARAMS + ["fast_weights"]
_BASELINE_CASES = [
    ("s0", "test_s0", 1, 0.0, "S0", 1.6),
    ("s1", "test_s1", 1, 0.15, "S1", 1.0),
]


def _fast_weights_active(cfg) -> bool:
    r = getattr(cfg, "randomize", None) or {}
    spec = r.get("fast_weights") if isinstance(r, dict) else None
    if spec is None:
        return False
    return bool(spec.get("randomized") or spec.get("biased"))


def make_obs_j_indices(NO, J_truth, J_obs):
    if J_obs is None or J_obs >= J_truth:
        return None
    X_idx = list(range(NO))
    Y_idx = []
    for k in range(NO):
        for j in range(J_obs):
            Y_idx.append(NO + k * J_truth + j)
    return tuple(X_idx + Y_idx)


def _per_window_params(w, cfg, da_J=None):
    params = {}
    for k in _L96_SCALAR_PARAMS:
        params[k] = w.get(f"{k}_da", w.get(k, getattr(cfg, "F_da" if k == "F" else k, 1.0)))
    if _fast_weights_active(cfg):
        fw = w.get("fast_weights_da", w.get("fast_weights", list(cfg.fast_weights)))
        if fw is None:
            params["fast_weights"] = fw
        elif da_J is not None:
            params["fast_weights"] = list(fw)[:da_J]
        else:
            raise ValueError(
                "fast_weights randomization active but da_J=None; weighting length is "
                "ambiguous for reduced-J dynamics. Pass da_J explicitly."
            )
    return params


def _build_eval_kwargs(pw, device):
    kw = {k: torch.tensor([p[k] for p in pw], device=device) for k in _L96_SCALAR_PARAMS}
    if pw and "fast_weights" in pw[0]:
        kw["fast_weights"] = torch.tensor([p["fast_weights"] for p in pw], device=device)
    return kw


def _to_tensor_kw(kw, device):
    """Convert list/tuple kwarg values (e.g. fast_weights) to device tensors.
    Scalar floats are left as-is; only sequences are converted."""
    for k, v in kw.items():
        if isinstance(v, (list, tuple)):
            kw[k] = torch.tensor(v, device=device)
    return kw


def _baseline_traj_path(case_name, method_name, dws_suffix="", param_suffix=""):
    key = f"{case_name}_{method_name.replace('-', '_').replace(' ', '_')}"
    return os.path.join(EXP_DIR, f"l96_baselines_trajs{dws_suffix}{param_suffix}_{key}.npz")


def _per_group_rmse(mean_rmse, NO=8, obs_j=2):
    groups = {}
    groups["slow"] = float(np.mean(mean_rmse[:NO]))
    groups["obs_fast"] = float(np.mean(mean_rmse[NO:]))
    groups["all_obs"] = float(np.mean(mean_rmse))
    return groups


def fmt_rmse(mean_arr, std_arr, NO=8, obs_j=2):
    d = {f"X{i+1}": {"mean": float(mean_arr[i]), "std": float(std_arr[i])}
         for i in range(len(mean_arr))} | {"mean": float(np.mean(mean_arr))}
    d["groups"] = _per_group_rmse(mean_arr, NO=NO, obs_j=obs_j)
    return d


def _per_group_ev(ev_arr, NO=8, obs_j=2):
    groups = {}
    groups["slow"] = float(np.mean(ev_arr[:NO]))
    groups["obs_fast"] = float(np.mean(ev_arr[NO:]))
    groups["all_obs"] = float(np.mean(ev_arr))
    return groups


def fmt_ev(ev_arr, NO=8, obs_j=2):
    d = {f"X{i+1}": float(ev_arr[i]) for i in range(len(ev_arr))}
    d["groups"] = _per_group_ev(ev_arr, NO=NO, obs_j=obs_j)
    return d


def _per_group_es(es_arr, NO=8, obs_j=2):
    groups = {}
    groups["slow"] = float(np.mean(es_arr[:NO]))
    groups["obs_fast"] = float(np.mean(es_arr[NO:]))
    groups["all_obs"] = float(np.mean(es_arr))
    return groups


def fmt_es(es_arr, NO=8, obs_j=2):
    d = {f"X{i+1}": float(es_arr[i]) for i in range(len(es_arr))}
    d["groups"] = _per_group_es(es_arr, NO=NO, obs_j=obs_j)
    return d


def _method_truth(truth: torch.Tensor, method, obs_var_indices) -> torch.Tensor:
    """Slice truth to the method's state dims so in-run ES/RMSE refs are valid.

    Reduced-dynamics methods (e.g. S1 with J=2) have ``state_dim`` equal to the
    observed subspace while cached windows store the full 40D truth; without
    this slice their ``ref_full`` guard yields None and no ES is accumulated.
    """
    sd = getattr(method, "state_dim", None)
    if (
        sd is not None
        and obs_var_indices is not None
        and truth.shape[-1] > sd
        and len(obs_var_indices) == sd
    ):
        return truth[..., obs_var_indices]
    return truth


def evaluate_baseline(method, dataset, cfg, device, return_trajs=False, batch_size=1, da_J=None):
    rmse_list = []
    results_list = []
    all_sq_err = []
    all_ref = []
    es_list = []
    use_corrupted = getattr(cfg, 'use_corrupted_forcing', True)
    force_key = "forcing_corrupted" if use_corrupted else "forcing_true"
    obs_var_indices = cfg.obs_var_indices

    def _subsample_es(es, analysis_eval):
        if es is None:
            return None
        es = np.asarray(es)
        if obs_var_indices is not None and es.shape[-1] > len(obs_var_indices):
            return es[..., obs_var_indices]
        return es[..., :analysis_eval.shape[-1]]

    if batch_size > 1 and callable(getattr(method, 'assimilate_batch', None)):
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            obs = torch.stack([w["obs"].to(device) for w in batch], dim=0)
            mask = torch.stack([w["obs_mask"].to(device) for w in batch], dim=0)
            truth = torch.stack([w["true_state"] for w in batch], dim=0)
            force = torch.stack([w[force_key].to(device) for w in batch], dim=0)
            pw = [_per_window_params(w, cfg, da_J=da_J) for w in batch]
            kw = _build_eval_kwargs(pw, device)
            results = method.assimilate_batch(obs, mask, force, _method_truth(truth, method, obs_var_indices), **kw)
            for result_idx, result in enumerate(results):
                analysis = result.trajectory
                if obs_var_indices is not None:
                    ref = truth[result_idx].detach().cpu().numpy()[..., obs_var_indices]
                    analysis_eval = analysis
                    if analysis_eval.shape[-1] > len(obs_var_indices):
                        analysis_eval = analysis_eval[..., obs_var_indices]
                else:
                    ref = truth[result_idx].detach().cpu().numpy()
                    analysis_eval = analysis
                    if analysis_eval.shape[-1] != ref.shape[-1]:
                        ref = ref[..., :analysis_eval.shape[-1]]
                if not np.isfinite(analysis_eval).all():
                    continue
                result.rmse = np.sqrt(np.mean((analysis_eval - ref) ** 2, axis=0))
                es_s = _subsample_es(getattr(result, "es", None), analysis_eval)
                if es_s is not None:
                    es_list.append(es_s)
                all_sq_err.append((analysis_eval - ref) ** 2)
                all_ref.append(ref)
                rmse_list.append(result.rmse)
                results_list.append(result)
    else:
        for i in range(len(dataset)):
            w = dataset[i]
            obs = w["obs"].to(device)
            mask = w["obs_mask"].to(device)
            truth = w["true_state"]
            force = w[force_key].to(device)
            kw = _per_window_params(w, cfg, da_J=da_J)
            _to_tensor_kw(kw, device)
            result = method.assimilate(obs, mask, force, _method_truth(truth, method, obs_var_indices), **kw)
            analysis = result.trajectory
            if obs_var_indices is not None:
                ref = truth.numpy()[..., obs_var_indices]
                analysis_eval = analysis
                if analysis_eval.shape[-1] > len(obs_var_indices):
                    analysis_eval = analysis_eval[..., obs_var_indices]
            else:
                ref = truth.numpy()
                analysis_eval = analysis
                if analysis_eval.shape[-1] != ref.shape[-1]:
                    ref = ref[..., :analysis_eval.shape[-1]]
            if not np.isfinite(analysis_eval).all():
                continue
            result.rmse = np.sqrt(np.mean((analysis_eval - ref) ** 2, axis=0))
            es_s = _subsample_es(getattr(result, "es", None), analysis_eval)
            if es_s is not None:
                es_list.append(es_s)
            all_sq_err.append((analysis_eval - ref) ** 2)
            all_ref.append(ref)
            rmse_list.append(result.rmse)
            results_list.append(result)

    all_rmse = np.stack(rmse_list, axis=0)
    rmse_stats = (np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0))
    all_sq_err = np.concatenate(all_sq_err, axis=0)
    all_ref = np.concatenate(all_ref, axis=0)
    pooled_mse = np.mean(all_sq_err, axis=0)
    pooled_var = np.var(all_ref, axis=0)
    pooled_var = np.maximum(pooled_var, 1e-12)
    expvar = 1 - pooled_mse / pooled_var
    expvar_stats = (expvar, np.zeros_like(expvar))
    if es_list:
        all_es = np.stack(es_list, axis=0)
        es_stats = (np.mean(all_es, axis=0), np.std(all_es, axis=0))
    else:
        es_stats = (np.zeros(expvar.shape), np.zeros(expvar.shape))
    if return_trajs:
        return (rmse_stats, expvar_stats, es_stats), results_list
    return rmse_stats, expvar_stats, es_stats


def run_and_cache_baselines(datasets, device, batch_size=1, da_window_steps=None,
                             weak_config=None, strong_config=None, enkf_config=None,
                             etkf_config=None, suffix="", exclude_methods=None,
                             obs_j=2, obs_interval=100, fw_randomized=False):
    if da_window_steps is None:
        N = int(3.0 / 0.001)
    else:
        N = da_window_steps
    dws_suffix = f"_dws{N}"
    param_suffix = suffix
    if enkf_config and enkf_config.get("inflation", 1.0) != 1.0:
        param_suffix += f"_inf{enkf_config['inflation']}"
    if etkf_config and etkf_config.get("inflation", 1.0) != 1.0:
        param_suffix += f"_etkf_inf{etkf_config['inflation']}"
    if obs_j is not None and obs_j < 4:
        param_suffix += f"_obsj{obs_j}"
    if obs_interval is not None:
        param_suffix += f"_int{obs_interval}"
    if fw_randomized:
        param_suffix += "_fw"
    cache_path = os.path.join(EXP_DIR, f"l96_baselines{dws_suffix}{param_suffix}.json")

    partial = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            partial = json.load(f)
        print(f"  Found partial results ({cache_path}), resuming...")
    else:
        print(f"  Running L96 baselines (da_window_steps={N}, obs_j={obs_j}, obs_interval={obs_interval})...")

    weak_cfg = weak_config or {}
    strong_cfg = strong_config or {}
    enkf_cfg = enkf_config or {}
    etkf_cfg = etkf_config or {}

    exclude = set(exclude_methods or [])
    active_methods = [m for m in _BASELINE_METHODS if m not in exclude]

    dt_l96 = 0.001
    NO = 8
    J_truth = 4
    obs_var_indices = make_obs_j_indices(NO, J_truth, obs_j)
    obs_dim = len(obs_var_indices) if obs_var_indices is not None else NO * (1 + J_truth)

    s0_obs_op = ObsOperator(NO + NO * J_truth, obs_var_indices)
    s0_dynamics = Lorenz96Dynamics(dt=dt_l96, coupling_exponent=1.6)

    s1_J = obs_j
    s1_state_dim = NO + NO * s1_J
    s1_obs_indices = list(range(s1_state_dim))
    s1_obs_op = ObsOperator(s1_state_dim, s1_obs_indices)
    if s1_J != J_truth:
        s1_dynamics = Lorenz96Dynamics(dt=dt_l96, NO=NO, J=s1_J, h=1.0, hx=1.0, eps=0.1,
                                       coupling_exponent=1.0)
    else:
        s1_dynamics = Lorenz96Dynamics(dt=dt_l96, coupling_exponent=1.0)

    baseline_pool = {}
    for case_name, _, _, _, _, coupling_exponent in _BASELINE_CASES:
        if case_name == "s0":
            dyn, obs_op = s0_dynamics, s0_obs_op
        else:
            dyn, obs_op = s1_dynamics, s1_obs_op
        pool = {}
        if "Weak-4DVar" not in exclude:
            pool["Weak-4DVar"] = Weak4DVar(dt=dt_l96, da_window_steps=N, device=device,
                                             coupling_exponent=coupling_exponent, dynamics=dyn,
                                             obs_operator=obs_op, **weak_cfg)
        if "Strong-4DVar" not in exclude:
            pool["Strong-4DVar"] = Strong4DVar(dt=dt_l96, da_window_steps=N, device=device,
                                                  coupling_exponent=coupling_exponent, dynamics=dyn,
                                                  obs_operator=obs_op, **strong_cfg)
        if "EnKF" not in exclude:
            pool["EnKF"] = EnKF(dt=dt_l96, device=device, coupling_exponent=coupling_exponent,
                                  dynamics=dyn, obs_operator=obs_op, NO=NO, J=(J_truth if case_name == "s0" else s1_J),
                                  **enkf_cfg)
        if "ETKF" not in exclude:
            pool["ETKF"] = ETKF(dt=dt_l96, device=device, coupling_exponent=coupling_exponent,
                                  dynamics=dyn, obs_operator=obs_op, NO=NO, J=(J_truth if case_name == "s0" else s1_J),
                                  **etkf_cfg)
        baseline_pool[case_name] = pool

    cfg_s0 = Lorenz96Config(param_bias=0.0, forcing_state_bias=0.0, T_max=3.0, seed=123,
                             obs_interval=obs_interval, obs_var_indices=obs_var_indices)
    cfg_s1 = Lorenz96Config(param_bias=0.15, forcing_state_bias=0.1, T_max=3.0, seed=131,
                             obs_interval=obs_interval, obs_var_indices=obs_var_indices)
    cfg_map = {"s0": cfg_s0, "s1": cfg_s1}

    if "config" not in partial:
        partial["config"] = {"T_max": 3.0, "da_window_steps": N, "obs_j": obs_j,
                              "obs_interval": obs_interval, "obs_dim": obs_dim,
                              "s1_J": s1_J, "s1_state_dim": s1_state_dim}

    total_t0 = time.time()

    for case_name, ds_key, case_val, bias, label, coupling_exponent in _BASELINE_CASES:
        if ds_key not in datasets:
            continue
        ds = datasets[ds_key]
        cfg = cfg_map[case_name]
        method_map = baseline_pool[case_name]
        for name in active_methods:
            if partial.get(case_name, {}).get(name) is not None:
                print(f"    {label}/{name:<15} already done, skipping")
                continue

            method = method_map[name]
            print(f"    {label}/{name:<15} ...", end=" ", flush=True)
            t1 = time.time()
            da_J = J_truth if case_name == "s0" else s1_J
            ((m, s), (ev_arr, _), (es_arr, es_std)), bl_results = evaluate_baseline(method, ds, cfg, device, return_trajs=True, batch_size=batch_size, da_J=da_J)
            elapsed = time.time() - t1

            if case_name not in partial:
                partial[case_name] = {}
            partial[case_name][name] = fmt_rmse(m, s, NO=NO, obs_j=obs_j)
            partial[case_name][name]["ev"] = fmt_ev(ev_arr, NO=NO, obs_j=obs_j)
            partial[case_name][name]["es"] = fmt_es(es_arr, NO=NO, obs_j=obs_j)
            partial["total_time_seconds"] = time.time() - total_t0
            with open(cache_path, "w") as f:
                json.dump(partial, f, indent=2)

            trajs = np.stack([r.trajectory for r in bl_results], axis=0)
            traj_data = {"trajectories": trajs}
            if bl_results[0].ensemble_variance is not None:
                traj_data["ensemble_variance"] = np.stack(
                    [r.ensemble_variance for r in bl_results], axis=0)
            np.savez_compressed(_baseline_traj_path(case_name, name, dws_suffix, param_suffix), **traj_data)

            rmse_mean = np.mean(m)
            groups = _per_group_rmse(m, NO=NO, obs_j=obs_j)
            ev_groups = _per_group_ev(ev_arr, NO=NO, obs_j=obs_j)
            es_groups = _per_group_es(es_arr, NO=NO, obs_j=obs_j)
            print(f"  mean={rmse_mean:.4f} slow={groups['slow']:.4f} obs_fast={groups['obs_fast']:.4f} "
                  f"ev={ev_groups['all_obs']:.4f} (slow={ev_groups['slow']:.4f} obs_fast={ev_groups['obs_fast']:.4f}) "
                  f"es={es_groups['all_obs']:.4f} (slow={es_groups['slow']:.4f} obs_fast={es_groups['obs_fast']:.4f}) "
                  f"[{elapsed:.1f}s]")

    traj_path = os.path.join(EXP_DIR, f"l96_baselines_trajectories{dws_suffix}{param_suffix}.npz")
    all_present = all(
        os.path.exists(_baseline_traj_path(case_name, name, dws_suffix, param_suffix))
        for case_name, _, _, _, _, _ in _BASELINE_CASES
        for name in active_methods
    )

    if all_present:
        print("  Combining trajectories...")
        traj_arrays = {}
        for case_name, _, _, _, _, _ in _BASELINE_CASES:
            for name in active_methods:
                src = _baseline_traj_path(case_name, name, dws_suffix, param_suffix)
                data = np.load(src)
                prefix = f"{case_name}_{name.replace('-', '_').replace(' ', '_')}"
                for key in data.files:
                    traj_arrays[f"{prefix}_{key}"] = data[key]
                data.close()
                os.remove(src)
        np.savez_compressed(traj_path, **traj_arrays)
        print(f"    Saved: {traj_path}")
    else:
        print("    (incomplete — skipping trajectory combination)")

    return partial
