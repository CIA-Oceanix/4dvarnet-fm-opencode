#!/usr/bin/env python3
"""Generate the revised consolidated QG DA-baseline report (psi-obs focus).

JSON-only generator (no QG/neural code imports). It consumes the curated
result JSONs committed on master under ``reports/qg/outputs/`` plus the
``qg_settings.json`` QGConfig snapshot, and renders a self-contained Markdown
report with:

- the governing equations of the two-layer Phillips QG model (and, for the
  QG1L scenario, the reduced-gravity 1-layer model),
- a compact case-study table describing S0 and the two S1 configurations,
- an S0 metrics section (error-free, psi-obs matrix),
- an S1-QG2L metrics section (model error = param bias + corrupted wind +
  cross-resolution at da_nx = 16 / 32 / 64),
- an S1-QG1L metrics section (structural error, obs-var r-scale sweep).

Metrics are reported as RMSE / free-forecast RMSE / forecast-improvement /
pooled explained-variance (EV), per field (q, psi) and per layer.

Run from the repository root (data lives under ``reports/qg/outputs/``)::

    python reports/qg/generate_qg_s0s1_report.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FIELD_LABELS = {"q": "PV q", "psi": "streamfunction ψ"}
LAYER_LABELS = {"layer1": "upper (layer 1)", "layer2": "lower (layer 2)",
                "full": "full state"}

_EQUATIONS = r"""
### 1.1 Two-layer quasi-geostrophic (Philips) model

The truth dynamics are the two-layer Phillips-channel QG equations
(double-periodic β-plane) solved by `QGDynamics` (native torch port of pyqg
v0.4.0, flux-form advection, `-J(ψ, q)`). Potential vorticity and evolution:

$$ q_1 = \nabla^2 \psi_1 + F_1(\psi_2 - \psi_1), \qquad
    q_2 = \nabla^2 \psi_2 + F_2(\psi_1 - \psi_2) $$

$$ \frac{d q_1}{dt} = -J(\psi_1,\, q_1) - Q_{y1}\,\partial_x \psi_1
    + \mathrm{curl}\,\tau $$

$$ \frac{d q_2}{dt} = -J(\psi_2,\, q_2) - Q_{y2}\,\partial_x \psi_2
    + r_{\mathrm{ek}}\,\nabla^2 \psi_2 $$

with the PV gradients and layer Froude numbers

$$ Q_{y1} = \beta + F_1\,(U_1 - U_2), \qquad
    Q_{y2} = \beta - F_2\,(U_1 - U_2), \qquad
    F_1 = \frac{r_d^{-2}}{1+\delta}, \qquad
    F_2 = \delta\, F_1 . $$

Here $\beta$ is the planetary vorticity gradient, $r_d$ the deformation
radius, $\delta$ the layer-depth ratio and $U_1,U_2$ the imposed mean zonal
flows. Time stepping is RK4 with the pyqg exponential spectral filter applied
once per step. State layout is flattened `[2·ny·nx]` (layer-major).

### 1.2 Wind forcing (upper-layer PV source)

A moving-storm wind-stress curl $\mathrm{curl}\,\tau$ drives the upper-layer
PV. It is a localized Gaussian storm (Witch-of-Agnesi profile) whose centre
follows a storm track $x_c(t) = x_0 + c_x t + w_x(t)$ (mod $L$),
$y_c(t) = y_0 + c_y t + w_y(t)$ (mod $W$) with OU position jitter
$w_x,w_y$ and OU amplitude $A(t)$ (`wind_amp`). For each window the storm
start, track drift and amplitude are randomized (see case-study table).

### 1.3 Reduced-gravity single-layer model (S1-QG1L)

The structural-error DA model is a reduced-gravity 1-layer QG (`QG1LDynamics`):
one active upper layer over a motionless deep layer,

$$ q = \nabla^2 \psi - \frac{\psi}{r_d^2}, \qquad
    \psi = -\left(\nabla^2 + r_d^{-2}\right)^{-1} q $$

$$ \frac{d q}{dt} = -J(\psi,\, q + \beta y) - r_{\mathrm{ek}}\,\nabla^2 \psi
    + \mathrm{curl}\,\tau . $$

`param_names = [β, rd, rek, U1]`. The truth remains the two-layer model; the
DA filter uses the single-layer PV structure as a mistaken forecast model.
"""

_CASE_TABLE = r"""
| Config | Dynamics / DA model | Observed field | Obs geometry | Model error | DA resolution |
|---|---|---|---|---|---|
| **S0** | truth = 2-layer QG; DA = `qg2l` (exact) | upper-layer ψ (psi-obs) | random columns, `cols_per_day` ∈ {4, 8} | none (error-free; obs noise only 1%) | full res (da_nx = nx = 64) |
| **S1-QG2L** | truth = 2-layer QG; DA = `qg2l_lores` (2-layer, honest) | upper-layer ψ (psi-obs) | random columns, `cols_per_day` = 4 | param bias (`rd,rek` ×0.85) + corrupted wind (OU jitter + amplitude bias) + cross-resolution da_nx ∈ {16, 32, 64} | da_nx = 16 / 32 / 64 |
| **S1-QG1L** | truth = 2-layer QG; DA = `qg1l` (reduced-gravity 1-layer) | upper-layer ψ (psi-obs; q-obs reference) | random columns, `cols_per_day` = 4 | structural error (1-layer vs 2-layer mismatch); `obs_var_r_scale` ∈ {1, 100, 1e4} | full res (da_nx = 64) |

Observed field is the upper-layer streamfunction ψ₁ for every configuration
(psi-obs; the `ObsOperator` inverts ψ to PV after spectral upsampling on the
DA grid). The q-obs (upper-layer PV) runs are included only as local-VP
reference in the QG1L section. All scenarios share the ETKF (N = 80,
inflation 1.0, Gaspari–Cohn localization radius 6) and lagged-truth
initialization (lags 1.0 and 2.0 d in S0/S1-QG2L; 1.0 d in S1-QG1L).
"""


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_ev(x) -> str:
    return f"{x:+.3f}" if isinstance(x, (int, float)) else str(x)


def fmt_improv(x) -> str:
    return f"{x:.2f}"


def fmt_rmse(x) -> str:
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.3g}"


def settings_markdown(s: dict) -> str:
    rows = [
        ("Domain length L", f"{s['L']:.0f} m"),
        ("Grid", f"nx = ny = 64, state_dim = {s['state_dim']} (2×64×64, layer-major)"),
        ("Time step", f"dt = {s['dt']:.0f} s, steps_per_day = {s['steps_per_day']}"),
        ("Assimilation window", f"{s['window_days']:.0f} days ({s['num_steps']} steps)"),
        ("Spinup", f"{s['spinup_years']:.0f} years"),
        ("Windows", f"{s['num_windows']}"),
        ("Physics β", f"{s['beta']:.1e}"),
        ("Physics rd", f"{s['rd']:.0f} m"),
        ("Physics δ (layer-depth ratio)", f"{s['delta']}"),
        ("Physics U₁", f"{s['U1']}"),
        ("Physics U₂", f"{s['U2']}"),
        ("Physics rek (linear drag)", f"{s['rek']:.3e}"),
        ("Spectral filter", f"filterfac = {s['filterfac']}"),
        ("Seed", f"{s['seed']}"),
    ]
    line = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return "| Parameter | Value |\n|---|---|\n" + line + "\n"


def find_json(root: Path, subdir: str, lag: float) -> dict | None:
    """Return the single result JSON in ``subdir`` whose name contains ``lag``."""
    d = root / subdir
    if not d.is_dir():
        return None
    lag_lbl = f"lag{lag:g}"
    for p in sorted(d.glob("*.json")):
        if lag_lbl in p.name:
            return load_json(p)
    return None


def find_json_dir(root: Path, subdir: str) -> dict | None:
    """Return the (single) result JSON in a lag-specific ``subdir``."""
    d = root / subdir
    if not d.is_dir():
        return None
    jsons = sorted(d.glob("*.json"))
    return load_json(jsons[0]) if jsons else None


def _lag_dir(label: str, lag: float) -> str:
    return f"{label}_lag{lag:.1f}".replace(".", "p")


def _find_rs_json(rsdir: Path, field: str, tag: str) -> Path | None:
    """Return the qg1l r-scale JSON for `field` ('psi'/'q') and `tag` ('rs1'...)."""
    for p in rsdir.glob("*.json"):
        if f"_{field}_" in p.name and (
            f"_{tag}_" in p.name or p.name.endswith(f"_{tag}.json")
        ):
            return p
    return None


def _per_field_table(s1: dict) -> list[str]:
    """Render a per-field/per-layer metrics table for a scenario."""
    add = []
    mpf = s1["metrics_per_field"]
    for fld in ("q", "psi"):
        for lyr in ("layer1", "layer2", "full"):
            if lyr not in mpf[fld]:
                continue
            m = mpf[fld][lyr]
            add.append(f"| {FIELD_LABELS[fld]} | {LAYER_LABELS[lyr]} | "
                       f"{fmt_rmse(m['rmse'])} | {fmt_rmse(m['rmse_free'])} | "
                       f"{fmt_improv(m['improv'])} | {fmt_ev(m['ev'])} | "
                       f"{fmt_ev(m['ev_free'])} |")
    return add


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", default=str(ROOT / "reports/qg/outputs"))
    ap.add_argument("--settings", default=str(ROOT / "reports/qg/outputs/qg_settings.json"))
    ap.add_argument("--out", default=str(ROOT / "reports/qg/outputs/qg_s0s1_report.md"))
    args = ap.parse_args()

    root = Path(args.json_root)
    settings = load_json(Path(args.settings))
    missing = []

    md = []
    add = md.append

    add("# QG DA Baselines — Consolidated Report (psi-obs focus)")
    add("")
    add("**Date:** 2026-09-02")
    add("**Branch (report):** master")
    add("**Scope:** psi-obs configurations (upper-layer streamfunction) only; "
        "S0 (error-free), S1-QG2L (param + forcing + cross-resolution error) and "
        "S1-QG1L (structural 1-layer error).")
    add("**Provenance (jobs, A40 `sl-mee-br-205`):** "
        "S0 1%-noise matrix (`qg_matrix_c{4,8}_psi`, lags 1/2); "
        "S1 @ da_nx=16 (`qg_s1`), da_nx=32 (`qg_s1_da32`), da_nx=64 (`qg_s1_nores`, "
        "lag 1.0); S1-QG1L r-scale probe (`qg_s1_qg1l_rscale`).")
    add("")
    add("## 1. System and governing equations")
    add("")
    add(_EQUATIONS.strip())
    add("")

    add("## 2. Case studies")
    add("")
    add("The observation configuration is **psi-obs** (upper-layer streamfunction "
        "at random meridional columns, 1% obs noise) throughout. Three case "
        "studies are benchmarked:")
    add("")
    add(_CASE_TABLE.strip())
    add("")

    add("## 3. Base configuration")
    add("")
    add("### 3.1 QGConfig snapshot")
    add("")
    add(settings_markdown(settings))
    add("### 3.2 Moving-storm wind forcing (upper-layer PV source)")
    add("")
    add(f"- Wind-stress-curl amplitude `wind_amp = {settings['wind_amp']:.0e}` "
        f"(Ornstein–Uhlenbeck, `wind_tau_days = {settings['wind_tau_days']:.0f}` d; "
        f"storm width `wind_sigma = {settings['wind_sigma']/1000:.0f}` km).")
    add(f"- Storm-track drift `wind_cx = {settings['wind_cx']}`, "
        f"`wind_cy = {settings['wind_cy']}` m/s; position OU jitter "
        f"`wind_drift_tau_days = {settings['wind_drift_tau_days']:.0f}` d, "
        f"`wind_drift_sigma = {settings['wind_drift_sigma']/1000:.0f}` km.")
    add("")
    add("### 3.3 Per-window truth randomization")
    add("")
    add(f"- U₁, rd, rek drawn once per window as `U[1 ± param_range]` "
        f"(`param_range = {settings['param_range']}`); β/δ fixed.")
    add("- Independent storm per window: start `(x0,y0) ~ U(0,L)²`, track "
        "`cx ~ U[0.25, 0.75]`, `cy ~ U[−0.06, 0.06]`.")
    add("- Wind amplitude drawn from discrete levels "
        "`{0, 3e-12, 1e-11, 2e-11, 3e-11}` round-robin `i % 5`.")
    add("- Initial state at `t₀ − U(0, init_lag_days)` (lagged-truth first guess).")
    add("")
    add("### 3.4 Observations")
    add("")
    add("- Geometry `random_columns`: `cols_per_day` distinct meridional columns "
        "of the upper-layer field, each observed exactly once per day at its own "
        "randomly-sampled intra-day step (no two columns of a day share a step).")
    add("- Observed field: upper-layer streamfunction ψ₁ (psi-obs) — the baseline "
        "`ObsOperator` inverts ψ to PV after spectral upsampling on the DA grid.")
    add("- Noise: `sigma = obs_noise_std_frac × std(field)`, `frac = 0.01` (1%).")
    add("- Coverage (production): cols = 4 → ~0.52% of space-time gridpoints.")
    add("")
    add("### 3.5 DA filter")
    add("")
    add("- **ETKF**, ensemble N = 80, inflation 1.0, Gaspari–Cohn localization "
        "radius 6 (physical coords on the DA grid).")
    add("- Init: lagged-truth shared by the DA ensemble and the free-forecast "
        "reference; `disp_frac = 1.0` (background-error-scaled), band ±0.25 d.")
    add("- Lags: 1.0 d and 2.0 d (S0, S1-QG2L); 1.0 d (S1-QG1L).")
    add("")

    # ---- Section 4: S0 ----
    add("## 4. S0 metrics (error-free, psi-obs)")
    add("")
    add("Error-free benchmark: `da_params = true_params`, DA at full resolution "
        "(`da_nx = nx = 64`). psi-obs matrix, cols ∈ {4, 8}, lags 1.0/2.0, 1% noise. "
        "RMSE on the upper-layer ψ field; `improv` = forecast improvement (DA-RMSE / "
        "free-RMSE, >1 means the DA beats the free forecast); EV = pooled explained "
        "variance.")
    add("")
    # 4.1 headline matrix (psi)
    add("### 4.1 Headline (psi-obs)")
    add("")
    add("| obs | cols | lag | DA RMSE | Free RMSE | improv | EV_full | EV_free |")
    add("|---|---|---|---|---|---|---|---|")
    for obsvar in ("psi",):
        for cols in ("4", "8"):
            for lag in (1.0, 2.0):
                d = find_json(root, f"qg_matrix_c{cols}_{obsvar}", lag)
                if d is None:
                    missing.append(f"qg_matrix_c{cols}_{obsvar} lag{lag}")
                    continue
                s0 = d["scenarios"].get("test_s0", {})
                add(f"| {obsvar} | {cols} | {lag:.1f} | "
                    f"{fmt_rmse(s0['rmse_mean'])} | "
                    f"{fmt_rmse(s0['forecast_rmse_mean'])} | "
                    f"{fmt_improv(s0['forecast_improvement'])} | "
                    f"{fmt_ev(s0['expvar_full'])} | {fmt_ev(s0['expvar_free'])} |")
    add("")
    # 4.2 per-field (S0 psi cols=4 lag1)
    add("### 4.2 Per-field (psi-obs, cols=4, lag 1.0)")
    add("")
    d = find_json(root, "qg_matrix_c4_psi", 1.0)
    if d is not None:
        s0 = d["scenarios"]["test_s0"]
        add("| field | layer | DA RMSE | Free RMSE | improv | EV | EV_free |")
        add("|---|---|---|---|---|---|---|")
        add("\n".join(_per_field_table(s0)))
    add("")

    # ---- Section 5: S1-QG2L ----
    add("## 5. S1-QG2L metrics (param + forcing + cross-resolution error)")
    add("")
    add("Model-error S1 with the **2-layer** DA model (`qg2l_lores`): parameter "
        "bias (`rd,rek ← rd,rek × 0.85`) + corrupted wind (OU location jitter + "
        "amplitude bias) + cross-resolution da_nx. Ternary `da_nx` = cross-resolution "
        "ratio vs the 64×64 truth: 16 (4:1), 32 (2:1), 64 (1:1, no resolution "
        "mismatch). psi-obs, cols = 4, 1% noise.")
    add("")
    # 5.1 headline across da_nx
    add("### 5.1 Headline across da_nx (psi-obs, cols=4, lag 1.0)")
    add("")
    add("| da_nx | ratio | DA RMSE | Free RMSE | improv | EV_full | EV_free |")
    add("|---|---|---|---|---|---|---|")
    rows = [("qg_s1_lag1p0", 16, "4:1"), ("qg_s1_da32_lag1p0", 32, "2:1"),
            ("qg_s1_nores_lag1p0", 64, "1:1")]
    for subdir, da_nx, ratio in rows:
        d = find_json_dir(root, subdir)
        if d is None:
            missing.append(subdir)
            continue
        s1 = d["scenarios"]["test_s1"]
        add(f"| {da_nx} | {ratio} | {fmt_rmse(s1['rmse_mean'])} | "
            f"{fmt_rmse(s1['forecast_rmse_mean'])} | "
            f"{fmt_improv(s1['forecast_improvement'])} | "
            f"{fmt_ev(s1['expvar_full'])} | {fmt_ev(s1['expvar_free'])} |")
    add("")
    # 5.2 S1-QG2L lag trend at da16 (from full s1 lag1/2)
    add("### 5.2 S1-QG2L lag trend (da_nx=16)")
    add("")
    add("| lag | DA RMSE | Free RMSE | improv | EV_full | EV_free |")
    add("|---|---|---|---|---|---|")
    for lag in (1.0, 2.0):
        s1d = find_json_dir(root, _lag_dir("qg_s1", lag))
        if s1d is None:
            continue
        s1 = s1d["scenarios"]["test_s1"]
        add(f"| {lag:.1f} | {fmt_rmse(s1['rmse_mean'])} | "
            f"{fmt_rmse(s1['forecast_rmse_mean'])} | "
            f"{fmt_improv(s1['forecast_improvement'])} | "
            f"{fmt_ev(s1['expvar_full'])} | {fmt_ev(s1['expvar_free'])} |")
    add("")
    # 5.3-5.5 per-field each da_nx
    for i, (title, subdir, da_nx) in enumerate((
        ("S1-QG2L @ da_nx=16", "qg_s1_lag1p0", 16),
        ("S1-QG2L @ da_nx=32", "qg_s1_da32_lag1p0", 32),
        ("S1-QG2L @ da_nx=64 (nores)", "qg_s1_nores_lag1p0", 64),
    )):
        add(f"### 5.{3 + i} Per-field — {title} (lag 1.0)")
        add("")
        d = find_json_dir(root, subdir)
        if d is None:
            continue
        s1 = d["scenarios"]["test_s1"]
        add("| field | layer | DA RMSE | Free RMSE | improv | EV | EV_free |")
        add("|---|---|---|---|---|---|---|")
        add("\n".join(_per_field_table(s1)))
        add("")

    # ---- Section 6: S1-QG1L ----
    add("## 6. S1-QG1L metrics (structural error, r-scale sweep)")
    add("")
    add("Cross-model structural-error S1: the DA filter uses the **reduced-gravity "
        "1-layer** model (`qg1l`) against the 2-layer truth, at full resolution "
        "(da_nx = 64). Under this mismatch the nonlocal psi observations are "
        "over-trusted (DA worse than the free forecast, improv ~0.39 at default R). "
        "`obs_var_r_scale` inflates the observation-noise variance to model the "
        "unmodelled structural error: 1 → 100 → 1e4. psi-obs, cols=4, lag 1.0.")
    add("")
    add("### 6.1 Headline (psi obs, r-scale sweep)")
    add("")
    add("| r_scale | DA RMSE | Free RMSE | improv | EV_full | EV_free |")
    add("|---|---|---|---|---|---|")
    # flat dir: iterate files by rlbl
    rsdir = root / "qg_s1_qg1l_rscale"
    fsm = {"rs1": 1.0, "rs100": 100.0, "rs1e4": 1e4}
    if rsdir.is_dir():
        for tag, rv in sorted(fsm.items(), key=lambda kv: kv[1]):
            p = _find_rs_json(rsdir, "psi", tag)
            if p is None:
                missing.append(f"qg_s1_qg1l_rscale psi {tag}")
                continue
            dd = load_json(p)
            s1 = dd["scenarios"]["test_s1_qg1l"]
            add(f"| {rv:g} | {fmt_rmse(s1['rmse_mean'])} | "
                f"{fmt_rmse(s1['forecast_rmse_mean'])} | "
                f"{fmt_improv(s1['forecast_improvement'])} | "
                f"{fmt_ev(s1['expvar_full'])} | {fmt_ev(s1['expvar_free'])} |")
    else:
        missing.append("qg_s1_qg1l_rscale")
    add("")
    add("### 6.2 Per-field (psi obs, r-scale sweep)")
    add("")
    if rsdir.is_dir():
        for tag, rv in sorted(fsm.items(), key=lambda kv: kv[1]):
            p = _find_rs_json(rsdir, "psi", tag)
            if p is None:
                continue
            dd = load_json(p)
            add(f"**r_scale = {rv:g}**")
            add("")
            add("| field | layer | DA RMSE | Free RMSE | improv | EV | EV_free |")
            add("|---|---|---|---|---|---|---|")
            add("\n".join(_per_field_table(dd["scenarios"]["test_s1_qg1l"])))
            add("")
    add("### 6.3 Local PV (q-obs) reference (r_scale = 1)")
    add("")
    if rsdir.is_dir():
        p = _find_rs_json(rsdir, "q", "rs1")
        if p is not None:
            dd = load_json(p)
            s1 = dd["scenarios"]["test_s1_qg1l"]
            add("| field | layer | DA RMSE | Free RMSE | improv | EV | EV_free |")
            add("|---|---|---|---|---|---|---|")
            add("\n".join(_per_field_table(s1)))
            add("")
    add("")

    add("## 7. Interpretation")
    add("")
    add("- **S0 (error-free, psi-obs, new per-column timing):** increasing "
        "columns (4 → 8) and shorter lag (2.0 → 1.0) both improve skill; "
        "psi-obs at cols=8/lag 1.0 is the best S0 DA (improv ~1.53, "
        "EV_full ~+0.81), improved by the per-column timing vs the old "
        "simultaneous-constellation geometry (was improv ~1.42 / EV ~+0.78). "
        "cols=4 is essentially unchanged (improv ~1.14 vs 1.16). EV remains "
        "positive for the psi-obs matrix.")
    add("- **S1-QG2L resolution trend (16 → 32 → 64):** forecast-improv rises "
        "monotonically (~1.08 → ~1.38 → ~1.44) and pooled EV flips from "
        "negative at da_nx=16 (-0.14) to positive at da_nx=64 (+0.34) — the "
        "milder the cross-resolution mismatch, the easier the DA. The "
        "per-column timing is slightly worse than the old constellation S1 "
        "skill (da32/da64 RMSE +6–7%, EV −9 to −12 pts), since dispersing the "
        "simultaneous multi-column updates reduces each update's spatial "
        "information under model error.")
    add("- **S1-QG2L lag trend (da_nx=16):** lag 2.0 is slightly worse than lag "
        "1.0 (improv 1.08 vs 1.11) — longer window broadens the free forecast "
        "without adding DA skill, so the shortest assimilated lag is preferred. "
        "(S1 re-run at lag 1.0 only; the lag-2.0 row retains the old-geometry "
        "value.)")
    add("- **S1-QG1L structural error:** at default R the 1-layer filter is worse "
        "than the free forecast (improv ~0.39, negative EV) because the nonlocal "
        "psi observations are mutually inconsistent with the 1-layer model and "
        "over-trusted. Inflating the observation variance `obs_var_r_scale` "
        "1 → 100 → 1e4 recovers skill monotonically toward the free-forecast "
        "limit (improv 0.39 → 0.42 → 0.81, for the PV-q field) but does not "
        "cross 1.0. The local PV (q-obs) reference at r_scale=1 is the closest "
        "well-posed observation for the 1-layer model: it nearly reaches the "
        "free-forecast skill for q (improv ~0.96) and even beats it for the "
        "streamfunction (improv ~1.19) — a spatially-local observation is far "
        "more robust to the unresolved lower layer than the nonlocal psi columns.")
    add("")

    # ---- Section 8: Illustrations (S0 and S1-QG2L da_nx=32) ----
    add("## 8. Illustrations (S0 and S1-QG2L, da_nx=32)")
    add("")
    add("Single-window ETKF reconstruction figures (production cfg: nx=64, "
        "N=80, psi-obs, cols=4, 1% noise, lag 1.0; S1 additionally at da_nx=32 "
        "cross-resolution). Generated by "
        "`reports/qg/generate_qg_s0s1_figs.py` (no DA-cache dependency): "
        "`obs_days` (aggregated per-day obs, 2×2 panel), `obs_hovmoller` "
        "(full-window obs Hovmöller), `forcing` (moving-storm wind curl), "
        "`truth_psi_q` (ground-truth streamfunction/PV), `analysis` "
        "(truth vs free forecast vs DA analysis) and a `dacycle.gif`.")
    add("")
    for scen, label in (("s0", "S0"), ("s1x32", "S1-QG2L da_nx=32")):
        add(f"### 8.{('1' if scen == 's0' else '2')} {label}")
        add("")
        add("| panel | figure |")
        add("|---|---|")
        for name, cap in (
                ("obs_days", "aggregated per-day observations (upper-layer ψ)"),
                ("obs_hovmoller", "full-window observation Hovmöller"),
                ("forcing", "moving-storm wind-stress curl"),
                ("truth_psi_q", "ground-truth streamfunction and PV"),
                ("analysis", "DA reconstruction vs truth and free forecast"),
        ):
            fig = f"figs/qg_{scen}_{name}.png"
            if (Path(args.json_root) / fig).exists():
                add(f"| {cap} | ![]({fig}) |")
            else:
                add(f"| {cap} | *(missing `{fig}`)* |")
        gif = f"figs/qg_{scen}_dacycle.gif"
        if (Path(args.json_root) / gif).exists():
            add("| DA-cycle animation | " + f"![]({gif})" + " |")
        else:
            add(f"| DA-cycle animation | *(missing `{gif}`)* |")
        add("")
    add("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"wrote {out}")
    if missing:
        print("WARNING: missing JSONs:", missing)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
