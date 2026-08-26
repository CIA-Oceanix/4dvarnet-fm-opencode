"""Generic estimator evaluation for L96 state reconstructions.

Takes stored state estimates (any scheme: neural, DA, ...) as ``(W, T, D)``
arrays plus the reference truth and returns pooled RMSE / explained variance /
energy-score grouped by state component. This is deliberately scheme-agnostic:
it only consumes ``trajectories`` and ``truth`` arrays, so the same evaluation
is applied identically to the neural schemes and to the DA baselines.
"""
import numpy as np
import torch

NO = 8  # number of slow variables (L96 fast/slow split for grouped scoring)

# Canonical L96 joint parameter order (matches eval_joint_neural_l96 / baselines).
L96_PARAM_ORDER = ("F", "c1", "hx", "eps", "w1", "w2", "w3", "w4")


def _groups_from(full: np.ndarray) -> dict:
    """Group a per-dimension array into slow / obs_fast / all_obs means."""
    return {
        "slow": float(np.mean(full[:NO])),
        "obs_fast": float(np.mean(full[NO:])),
        "all_obs": float(np.mean(full)),
    }


def pooled_mse_sq_err(trajectories: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-dimension pooled mean squared error across all windows/timesteps."""
    return np.mean((trajectories - truth) ** 2, axis=(0, 1))


def pooled_variance(ref: np.ndarray) -> np.ndarray:
    """Per-dimension pooled variance of the reference truth (matches DA)."""
    return np.var(ref, axis=(0, 1))


def evaluate_estimates(trajectories: np.ndarray, truth: np.ndarray) -> dict:
    """Compute pooled RMSE / EV / (N=1) ES grouped by component.

    Parameters
    ----------
    trajectories : np.ndarray, shape (W, T, D)
        State estimates (any scheme).
    truth : np.ndarray, shape (W, T, D)
        Reference truth in the same observed subspace.

    Returns
    -------
    dict with ``rmse`` (scalar), ``groups`` {slow, obs_fast, all_obs},
    ``ev`` {groups: {...}} and ``es`` {groups: {...}}.
    """
    mse = pooled_mse_sq_err(trajectories, truth)
    var_ref = pooled_variance(truth)

    rmse_full = np.sqrt(mse)
    rmse_groups = _groups_from(rmse_full)

    var_ref_safe = np.maximum(var_ref, 1e-12)
    ev_full = 1.0 - mse / var_ref_safe
    ev_groups = _groups_from(ev_full)

    # For a deterministic reconstruction (N=1 ensemble), the Energy Score
    # reduces to the per-dimension mean absolute error.
    mae_full = np.mean(np.abs(trajectories - truth), axis=(0, 1))
    es_groups = _groups_from(mae_full)

    return {
        "rmse": float(np.mean(rmse_full)),
        "groups": rmse_groups,
        "ev": {"groups": ev_groups},
        "es": {"groups": es_groups},
        "num_samples": int(truth.shape[0]),
    }


def ensemble_es_terms(members: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension accuracy / spread terms of the Energy Score.

    Parameters
    ----------
    members : np.ndarray, shape (W, T, D, M)
        Per-window ensemble member trajectories.
    truth : np.ndarray, shape (W, T, D)

    Returns
    -------
    ``(mae, pairwise)`` per-dim arrays where ``mae_d`` is the mean absolute
    error over all windows/timesteps/members and ``pairwise_d`` is the mean
    over windows/timesteps of the member-pairwise absolute difference
    ``(1/M^2) sum_i sum_j |x_i - x_j|``.
    """
    mae = np.mean(np.abs(members - truth[:, :, :, None]), axis=(0, 1, 3))
    pairwise = np.zeros(members.shape[2])
    for w in range(members.shape[0]):
        em = np.moveaxis(members[w], -1, 0)
        pairwise += np.abs(em[:, None] - em[None, :]).mean(axis=(0, 1, 2))
    pairwise /= members.shape[0]
    return mae, pairwise


def pooled_ensemble_es(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-dimension pooled ensemble Energy Score (proper scoring rule).

    ``ES_d = mae_d - 0.5 * pairwise_d`` where ``mae_d`` is the mean absolute
    error over windows/timesteps/members and ``pairwise_d`` the mean pairwise
    member distance — identical to ``metrics.energy_score`` averaged over
    windows.
    """
    mae, pairwise = ensemble_es_terms(members, truth)
    return mae - 0.5 * pairwise


def evaluate_ensemble_estimates(members: np.ndarray, truth: np.ndarray) -> dict:
    """Evaluate an ensemble reconstruction: member-mean metrics + ensemble ES.

    The deterministic RMSE/EV/ES block is computed on the member **mean**
    trajectory (what a downstream consumer would use as the point estimate);
    the ``ensemble`` block adds the proper ensemble ES plus the spread.
    """
    mean_traj = members.mean(axis=-1)
    out = evaluate_estimates(mean_traj, truth)
    n = int(members.shape[3])
    es = pooled_ensemble_es(members, truth)
    spread = np.std(members, axis=-1).mean(axis=(0, 1))
    out["ensemble"] = {
        "num_members": n,
        "es": {"groups": _groups_from(es)},
        "spread": {"groups": _groups_from(spread)},
    }
    return out


def evaluate_npz(path: str) -> dict:
    """Evaluate a stored ``.npz`` with ``trajectories`` and ``truth`` arrays."""
    data = np.load(path)
    return evaluate_estimates(data["trajectories"], data["truth"])


def evaluate_ensemble_npz(path: str) -> dict:
    """Evaluate a stored ``.npz`` with ``members`` and ``truth`` arrays."""
    data = np.load(path)
    return evaluate_ensemble_estimates(data["members"], data["truth"])


def save_estimates(path: str, trajectories: np.ndarray, truth: np.ndarray) -> None:
    """Save estimate + truth arrays to a scheme-agnostic ``.npz``."""
    np.savez_compressed(path, trajectories=trajectories, truth=truth)


def nrmse_param(pred_params: np.ndarray, true_params: np.ndarray) -> dict:
    """Per-parameter normalized RMSE and its mean across parameters.

    ``NRMSE_p = sqrt(mean((pred - true)^2)) / mean(|true|)`` per parameter,
    normalizing away the scale difference between parameters of very different
    magnitudes (e.g. L96 F~8 vs eps~0.1) so each contributes equally. Accepts
    either a single ``(P,)`` or a batched ``(W, P)`` array. Returns a dict with
    ``{"per_param": np.ndarray (P,), "mean": float}``.
    """
    pred = np.atleast_2d(np.asarray(pred_params, dtype=float))
    true = np.atleast_2d(np.asarray(true_params, dtype=float))
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=0))
    scale = np.mean(np.abs(true), axis=0)
    nrmse = rmse / np.maximum(scale, 1e-12)
    return {"per_param": nrmse, "mean": float(np.mean(nrmse))}


def trajectory_forecast_skill(
    dynamics,
    x0_batch: np.ndarray,
    forcing_batch: np.ndarray,
    params_true: np.ndarray,
    params_est: np.ndarray,
    n_steps: int,
    obs_var_indices: tuple | None = None,
    state_indices: tuple | None = None,
) -> dict:
    """Short-term forecast divergence between true and estimated parameters.

    For each of the W windows, roll two trajectories of ``n_steps`` out of the
    same initial state ``x0`` driven by the same forcing sequence ``forcing_true``
    — one with the true parameters, one with the estimated parameters — then
    measure the RMSE/EV between them (pooled across windows, grouped by
    slow/obs_fast/all_obs). This quantifies the sensitivity of short-term
    forecast quality to parameter estimation error, in the same group
    convention as ``evaluate_estimates``.

    ``dynamics`` is a ``Lorenz96Dynamics`` (or any object exposing ``step``);
    ``params_true``/``params_est`` are ``(W, P)`` arrays of F, c1, hx, eps,
    w1..w4 (see ``L96_PARAM_ORDER``). ``state_indices`` selects the columns of
    the full state passed to the dynamics (default: all, i.e. full 40D state).
    """
    x0 = np.asarray(x0_batch)
    forcing = np.asarray(forcing_batch)
    pt = np.asarray(params_true, dtype=float)
    pe = np.asarray(params_est, dtype=float)
    W = x0.shape[0]
    if state_indices is not None:
        state_indices = tuple(state_indices)
    else:
        state_indices = tuple(range(x0.shape[1]))

    ref_all, est_all = [], []
    for w in range(W):
        x0w = torch.from_numpy(x0[w]).float()
        fw = torch.from_numpy(forcing[w][:n_steps]).float()
        ref = _rollout_from(dynamics, x0w, fw, pt[w], n_steps, state_indices)
        est = _rollout_from(dynamics, x0w, fw, pe[w], n_steps, state_indices)
        ref_all.append(ref)
        est_all.append(est)

    ref_all = np.stack(ref_all)
    est_all = np.stack(est_all)

    if obs_var_indices is not None:
        obs_idx = list(obs_var_indices)
        ref_all = ref_all[..., obs_idx]
        est_all = est_all[..., obs_idx]

    mse = pooled_mse_sq_err(est_all, ref_all)
    var_ref = pooled_variance(ref_all)
    rmse_full = np.sqrt(mse)
    var_safe = np.maximum(var_ref, 1e-12)
    ev_full = 1.0 - mse / var_safe

    return {
        "rmse": {"groups": _groups_from(rmse_full), "mean": float(np.mean(rmse_full))},
        "ev": {"groups": _groups_from(ev_full), "mean": float(np.mean(ev_full))},
        "n_steps": int(n_steps),
    }


def _rollout_from(dynamics, x0: torch.Tensor, forcing: torch.Tensor,
                  params: np.ndarray, n_steps: int, state_indices: tuple) -> np.ndarray:
    """Roll a L96 trajectory out of x0 with the given params; return ``(n_steps, D)``.

    ``params`` is the 8-vector ``(F, c1, hx, eps, w1..w4)`` (h held fixed at the
    dynamics' configured value). The forcing is applied step-wise as the scalar
    ``W``, matching ``Lorenz96Dynamics.step``.
    """
    names = L96_PARAM_ORDER
    pv = {names[i]: float(params[i]) for i in range(len(names))}
    fw = [pv["w1"], pv["w2"], pv["w3"], pv["w4"]]
    state = x0[list(state_indices)].clone()
    traj = np.zeros((n_steps, len(state_indices)))
    for t in range(n_steps):
        W = forcing[t]
        state = dynamics.step(
            state, W,
            F=pv["F"], c1=pv["c1"], hx=pv["hx"], eps=pv["eps"],
            fast_weights=fw,
        )
        traj[t] = state.detach().cpu().numpy()
    return traj
