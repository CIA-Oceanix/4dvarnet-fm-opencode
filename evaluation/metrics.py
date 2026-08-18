from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Existing metric helpers
# ---------------------------------------------------------------------------

def rmse(analysis: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((analysis - truth) ** 2, axis=0))


def param_rmse(pred_params: np.ndarray, true_params: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred_params - true_params) ** 2, axis=0))


def spread(ensemble_variance: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(ensemble_variance, axis=0))


def crps(ensemble: np.ndarray, truth: np.ndarray) -> float:
    N, T, D = ensemble.shape
    scores = np.zeros(D)
    for d in range(D):
        e = ensemble[:, :, d]
        t = truth[:, d]
        abs_diff = np.abs(e[np.newaxis, :, :] - e[:, np.newaxis, :])
        pairwise = np.mean(abs_diff, axis=(0, 1))
        abs_err = np.mean(np.abs(e - t[np.newaxis, :]), axis=0)
        scores[d] = np.mean(pairwise - abs_err)
    return scores


# ---------------------------------------------------------------------------
# Explained Variance (EV)
# ---------------------------------------------------------------------------

def explained_variance(
    analysis: np.ndarray,
    truth: np.ndarray,
    clim_var: np.ndarray | None = None,
) -> np.ndarray:
    """Per-dimension Explained Variance.

    EV = 1 - MSE / clim_var

    * EV == 1  → perfect reconstruction
    * EV == 0  → no skill (climatological mean)
    * EV <  0  → worse than climatological mean

    Parameters
    ----------
    analysis : np.ndarray, shape (T, D)
        Analysis (forecast / reconstruction) time series.
    truth : np.ndarray, shape (T, D)
        Reference (truth) time series.
    clim_var : np.ndarray, shape (D,), optional
        Climatological variance per dimension.  When *None* the sample
        variance of *truth* along the time axis is used as baseline.

    Returns
    -------
    np.ndarray, shape (D,)
        Explained variance for each state dimension.
    """
    if analysis.shape != truth.shape:
        raise ValueError(
            f"analysis shape {analysis.shape} != truth shape {truth.shape}"
        )
    mse = np.mean((analysis - truth) ** 2, axis=0)  # (D,)
    if clim_var is None:
        clim_var = np.var(truth, axis=0)
    # Avoid division by zero – dimensions with zero variance get EV = NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        ev = np.where(clim_var > 0.0, 1.0 - mse / clim_var, np.nan)
    return ev


# ---------------------------------------------------------------------------
# Shallow-Water (SW) component metrics
# ---------------------------------------------------------------------------

def _component_slice(idx: int, nxy: int) -> slice:
    """Return the slice for component *idx* (0-based) of size *nxy*."""
    return slice(idx * nxy, (idx + 1) * nxy)


def compute_sw_component_metrics(
    analysis: np.ndarray,
    truth: np.ndarray,
    Nx: int,
    Ny: int,
    clim_var: np.ndarray | None = None,
) -> dict:
    """Per-component (ocean / atmosphere) and per-field (h, u, v) RMSE + EV.

    Expected state layout per timestep::

        [ h₁(Nx*Ny), u₁(Nx*Ny), v₁(Nx*Ny),
          h₂(Nx*Ny), u₂(Nx*Ny), v₂(Nx*Ny) ]

    Layer 1 (indices 0-2) = ocean (slow)
    Layer 2 (indices 3-5) = atmosphere (fast)

    Parameters
    ----------
    analysis, truth : np.ndarray, shape (T, 6*Nx*Ny)
    Nx, Ny : int
        Grid dimensions.
    clim_var : np.ndarray, shape (6*Nx*Ny,), optional
        Per-dimension climatological variance.

    Returns
    -------
    dict
        ``{"ocean":    {"h": {rmse, ev}, "u": ..., "v": ..., "aggregate": ...},
          "atmosphere": {"h": {rmse, ev}, "u": ..., "v": ..., "aggregate": ...},
          "overall":    {rmse, ev}}``
    """
    nxy = Nx * Ny
    total = 6 * nxy

    if analysis.shape[1] != total or truth.shape[1] != total:
        raise ValueError(
            f"Expected {total} state dims (Nx={Nx}, Ny={Ny}), "
            f"got analysis={analysis.shape[1]}, truth={truth.shape[1]}"
        )

    field_names = ["h", "u", "v"]
    layer_names = ["ocean", "atmosphere"]

    ev_all = explained_variance(analysis, truth, clim_var=clim_var)  # (6*Nxy,)
    rmse_all = np.sqrt(np.mean((analysis - truth) ** 2, axis=0))    # (6*Nxy,)

    result: dict = {}
    for layer_idx, layer_name in enumerate(layer_names):
        base = layer_idx * 3  # offset into the 6-component layout
        layer_metrics: dict[str, dict[str, float]] = {}
        layer_mse_parts: list[float] = []
        for fi, fname in enumerate(field_names):
            comp_slice = _component_slice(base + fi, nxy)
            f_rmse = float(np.mean(rmse_all[comp_slice]))
            f_ev = float(np.mean(ev_all[comp_slice]))
            layer_metrics[fname] = {"rmse": f_rmse, "ev": f_ev}
            layer_mse_parts.append(f_rmse ** 2)
        # Aggregate over h, u, v within the layer
        agg_mse = float(np.mean(layer_mse_parts))
        layer_var = (
            np.var(truth, axis=0) if clim_var is None else clim_var
        )
        layer_clim = float(np.mean(layer_var[base * nxy : (base + 3) * nxy]))
        agg_ev = (
            1.0 - agg_mse / layer_clim if layer_clim > 0.0 else float("nan")
        )
        layer_metrics["aggregate"] = {"rmse": float(np.sqrt(agg_mse)), "ev": agg_ev}
        result[layer_name] = layer_metrics

    # Overall metrics
    overall_rmse = float(np.mean(rmse_all))
    overall_mse = float(np.mean((analysis - truth) ** 2))
    if clim_var is not None:
        overall_clim = float(np.mean(clim_var))
    else:
        overall_clim = float(np.mean(np.var(truth, axis=0)))
    overall_ev = (
        1.0 - overall_mse / overall_clim if overall_clim > 0.0 else float("nan")
    )
    result["overall"] = {"rmse": overall_rmse, "ev": overall_ev}

    return result


# ---------------------------------------------------------------------------
# EV target validation
# ---------------------------------------------------------------------------

def validate_ev_targets(
    metrics: dict,
    targets: dict,
    scenario: str,
) -> dict:
    """Check whether EV targets are met for a given scenario.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_sw_component_metrics`.
    targets : dict
        ``{"ocean": <float>, "atmosphere": <float>}`` minimum acceptable EV.
    scenario : str
        ``"S0"`` or ``"S1"`` (used for labelling only).

    Returns
    -------
    dict
        ``{component: {"target": float, "actual": float, "passed": bool}}``
    """
    results: dict[str, dict[str, float | bool]] = {}
    for component, target_ev in targets.items():
        if component not in metrics:
            raise KeyError(
                f"Component '{component}' not found in metrics; "
                f"available: {list(metrics.keys())}"
            )
        actual_ev = metrics[component]["aggregate"]["ev"]
        results[component] = {
            "target": target_ev,
            "actual": actual_ev,
            "passed": bool(actual_ev >= target_ev),
        }
    return results


# ---------------------------------------------------------------------------
# Formatted printing
# ---------------------------------------------------------------------------

def print_sw_metrics_table(
    results: dict,
    case_name: str,
    Nx: int,
    Ny: int,
    clim_var: np.ndarray | None = None,
) -> None:
    """Print a formatted RMSE + EV table per component for the SW model.

    Parameters
    ----------
    results : dict
        Mapping ``method_name → (analysis, truth)`` where each is a
        ``(T, 6*Nx*Ny)`` array **or** a pre-computed output of
        :func:`compute_sw_component_metrics`.
    case_name : str
        Label printed in the table header (e.g. ``"S0"``).
    Nx, Ny : int
        Grid dimensions.
    clim_var : np.ndarray, shape (6*Nx*Ny,), optional
        Passed to :func:`compute_sw_component_metrics`.
    """
    header = (
        f"\n{'=' * 90}\n"
        f"  SW Metrics — {case_name}  (Nx={Nx}, Ny={Ny})\n"
        f"{'=' * 90}"
    )
    print(header)

    col_fmt = "{:<20} {:>10} {:>10} {:>10} {:>10}"
    print(col_fmt.format("Method", "Ocean RMSE", "Ocean EV", "Atm RMSE", "Atm EV"))
    print("-" * 90)

    for name, val in results.items():
        # Accept pre-computed metric dicts or raw (analysis, truth) pairs
        if isinstance(val, dict) and "ocean" in val:
            comp = val
        else:
            analysis, truth = val
            comp = compute_sw_component_metrics(
                analysis, truth, Nx, Ny, clim_var=clim_var
            )
        o = comp["ocean"]["aggregate"]
        a = comp["atmosphere"]["aggregate"]
        print(
            col_fmt.format(
                name,
                f"{o['rmse']:.4f}",
                f"{o['ev']:.4f}",
                f"{a['rmse']:.4f}",
                f"{a['ev']:.4f}",
            )
        )

    # Overall row
    print("-" * 90)
    # Compute overall from first entry for the summary line
    first_val = next(iter(results.values()))
    if isinstance(first_val, dict) and "ocean" in first_val:
        ov = first_val["overall"]
    else:
        analysis, truth = first_val
        ov = compute_sw_component_metrics(
            analysis, truth, Nx, Ny, clim_var=clim_var
        )["overall"]
    print(
        col_fmt.format(
            "OVERALL",
            f"{ov['rmse']:.4f}",
            f"{ov['ev']:.4f}",
            "",
            "",
        )
    )
    print("=" * 90)


# ---------------------------------------------------------------------------
# Generic metrics table (legacy L96)
# ---------------------------------------------------------------------------

def print_metrics_table(results: dict, case_name: str):
    print(f"\n{'=' * 70}")
    print(f"  {case_name}")
    print(f"{'=' * 70}")
    print(f"{'Method':<20} {'RMSE X':<12} {'RMSE Y':<12} {'RMSE Z':<12}")
    print(f"{'-' * 56}")
    for name, res in results.items():
        r = res.rmse
        print(f"{name:<20} {r[0]:<12.4f} {r[1]:<12.4f} {r[2]:<12.4f}")
    print(f"{'=' * 70}")
