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


def pooled_ensemble_es(
    members: np.ndarray, truth: np.ndarray, convention: str = "cache"
) -> np.ndarray:
    """Per-dimension pooled ensemble ES.

    ``convention="cache"`` reproduces exactly what ``_ESAccumulator``
    (``evaluation/baselines.py``) stores in the DA baseline caches so neural
    ensembles are directly comparable with EnKF/ETKF cached ES:

        ES_d = mae_d / M - 0.5 * pairwise_d

    ``convention="textbook"`` is the proper scoring rule
    (``metrics.energy_score`` pooled over windows):

        ES_d = mae_d - 0.5 * pairwise_d
    """
    mae, pairwise = ensemble_es_terms(members, truth)
    if convention == "cache":
        return mae / members.shape[3] - 0.5 * pairwise
    if convention == "textbook":
        return mae - 0.5 * pairwise
    raise ValueError(f"Unknown ES convention: {convention}")


def evaluate_ensemble_estimates(members: np.ndarray, truth: np.ndarray) -> dict:
    """Evaluate an ensemble reconstruction: member-mean metrics + ensemble ES.

    The deterministic RMSE/EV/ES block is computed on the member **mean**
    trajectory (what a downstream consumer would use as the point estimate);
    the ``ensemble`` block adds both ES conventions plus the ensemble spread.
    """
    mean_traj = members.mean(axis=-1)
    out = evaluate_estimates(mean_traj, truth)
    n = int(members.shape[3])
    es_cache = pooled_ensemble_es(members, truth, convention="cache")
    es_textbook = pooled_ensemble_es(members, truth, convention="textbook")
    spread = np.std(members, axis=-1).mean(axis=(0, 1))
    out["ensemble"] = {
        "num_members": n,
        "es_cache_convention": {"groups": _groups_from(es_cache)},
        "es_textbook": {"groups": _groups_from(es_textbook)},
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
