"""Generic estimator evaluation for L96 state reconstructions.

Takes stored state estimates (any scheme: neural, DA, ...) as ``(W, T, D)``
arrays plus the reference truth and returns pooled RMSE / explained variance /
energy-score grouped by state component. This is deliberately scheme-agnostic:
it only consumes ``trajectories`` and ``truth`` arrays, so the same evaluation
is applied identically to the neural schemes and to the DA baselines.
"""
import numpy as np

NO = 8  # number of slow variables (L96 fast/slow split for grouped scoring)


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


def evaluate_npz(path: str) -> dict:
    """Evaluate a stored ``.npz`` with ``trajectories`` and ``truth`` arrays."""
    data = np.load(path)
    return evaluate_estimates(data["trajectories"], data["truth"])


def save_estimates(path: str, trajectories: np.ndarray, truth: np.ndarray) -> None:
    """Save estimate + truth arrays to a scheme-agnostic ``.npz``."""
    np.savez_compressed(path, trajectories=trajectories, truth=truth)
