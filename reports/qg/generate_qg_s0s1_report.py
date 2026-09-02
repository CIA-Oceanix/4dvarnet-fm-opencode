#!/usr/bin/env python3
"""Generate the consolidated QG S0/S1 DA-baseline report.

JSON-only generator (no QG/neural code imports). It consumes:

- the curated S0/S1 result JSONs (committed on master under
  ``reports/qg/outputs/`` -- S0 1%-noise matrix, S1 @ da_nx=16, S1 @ da_nx=32), and
- a settings snapshot ``reports/qg/outputs/qg_settings.json`` (the ``QGConfig``
  values used by the runs, captured from ``data/qg.py``).

It renders ``reports/qg/outputs/qg_s0s1_report.md`` with the full S0/S1 settings
and all metric tables (RMSE / pooled-EV / forecast-improv, per field q/psi and
per layer) inlined, so the report is self-contained on master.

Source JSONs are looked up relative to ``--json-root`` (default
``reports/qg/outputs``, where the relocated QG result JSONs live on master). Run it
from the repo root; the rendered Markdown is what is committed to master.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FIELD_LABELS = {"q": "PV q", "psi": "streamfunction ψ"}
LAYER_LABELS = {"layer1": "upper (layer 1)", "layer2": "lower (layer 2)",
                "full": "full state"}


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
    """Map a result-dir label to the ``qg_*_lag{d}`` directory (dot->'p')."""
    return f"{label}_lag{lag:.1f}".replace(".", "p")


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

    add("# QG S0/S1 DA Baselines — Consolidated Report (cross-resolution S1)")
    add("")
    add("**Date:** 2026-08-31")
    add("**Branch (report):** master")
    add("**Provenance (jobs, A40 `sl-mee-br-205`):** "
        "S0 1%-noise matrix (prior session, jobs 50927/50930/50932); "
        "S1 @ da_nx=16 (job 51069); S1 @ da_nx=32 (job 51075).")
    add("**Result JSONs (committed on master):** "
        "`reports/qg/outputs/qg_matrix_{c4,c8}_{q,psi}/`, "
        "`reports/qg/outputs/qg_s1_lag{1,2}p0/`, "
        "`reports/qg/outputs/qg_s1_da32_lag{1,2}p0/`.")
    add("")

    add("## 1. Full S0/S1 settings")
    add("")
    add("Two-layer quasi-geostrophic (Phillips-channel double-periodic β-plane) model, "
        "`da_model='qg2l'` (2-layer) / `qg2l_lores` (coarse 2-layer DA model). "
        "The DA truth is generated on the fly per the `QGConfig` snapshot "
        "(`reports/qg/outputs/qg_settings.json`); the expensive spinup is cached "
        "under `reports/qg_cache/` (gitignored, available locally on master).")
    add("")
    add("### Base configuration (QGConfig snapshot)")
    add("")
    add(settings_markdown(settings))
    add("### Moving-storm wind forcing (upper-layer PV source)")
    add("")
    add(f"- Wind-stress-curl amplitude `wind_amp = {settings['wind_amp']:.0e}` "
        f"(Ornstein–Uhlenbeck, `wind_tau_days = {settings['wind_tau_days']:.0f}` d; "
        f"storm width `wind_sigma = {settings['wind_sigma']/1000:.0f}` km).")
    add(f"- Storm-track drift `wind_cx = {settings['wind_cx']}`, "
        f"`wind_cy = {settings['wind_cy']}` m/s; position OU jitter "
        f"`wind_drift_tau_days = {settings['wind_drift_tau_days']:.0f}` d, "
        f"`wind_drift_sigma = {settings['wind_drift_sigma']/1000:.0f}` km.")
    add("")
    add("### Per-window truth randomization")
    add("")
    add("- U₁, rd, rek drawn once per window as `U[1 ± param_range]` "
        f"(`param_range = {settings['param_range']}`); β/δ fixed.")
    add("- Independent storm per window: start `(x0,y0) ~ U(0,L)²`, track "
        "`cx ~ U[0.25, 0.75]`, `cy ~ U[−0.06, 0.06]`.")
    add("- Wind amplitude drawn from discrete levels "
        "`{0, 3e-12, 1e-11, 2e-11, 3e-11}` round-robin `i % 5`.")
    add("- Initial state at `t₀ − U(0, init_lag_days)` (lagged-truth first guess).")
    add("")
    add("### Observations")
    add("")
    add("- Geometry `random_columns`: `cols_per_day` distinct meridional columns "
        "of the upper-layer field, one simultaneous event/day at a random step.")
    add("- Observed field: upper-layer streamfunction ψ₁ (psi-obs) — the baseline "
        "`ObsOperator` inverts ψ to PV after spectral upsampling on the DA grid.")
    add("- Noise: `sigma = obs_noise_std_frac × std(field)`, `frac = 0.01` (1%).")
    add("- Coverage (production): cols = 4 → ~0.52% of space-time gridpoints.")
    add("")
    add("### Scenarios")
    add("")
    add("- **S0** (error-free): `da_params = true_params`, `da_model = 'qg2l'`, "
        "`da_nx = cfg.nx` (DA at full resolution).")
    add("- **S1** (model error): `da_model = 'qg2l_lores'`, `da_nx ∈ {16, 32}` "
        "(cross-resolution, truth at 64×64); param bias "
        f"`rd,rek ← rd,rek × (1 − s1_param_bias)` (`s1_param_bias = {settings['s1_param_bias']}`);")
    add("  corrupted wind: storm-centre OU jitter "
        f"(`s1_loc_sigma_frac = {settings['s1_loc_sigma_frac']}` × wind_sigma, "
        f"τ = {settings['s1_tau_days']:.0f} d) and amplitude `A(1+s1_amp_bias)+η` "
        f"(`s1_amp_bias = {settings['s1_amp_bias']}`, OU η with "
        f"`s1_sigma_eta_frac = {settings['s1_sigma_eta_frac']}`, τ = "
        f"{settings['s1_tau_days']:.0f} d).")
    add("")
    add("### DA filter")
    add("")
    add("- **ETKF**, ensemble N = 80, inflation 1.0, Gaspari–Cohn localization "
        "radius 6 (physical coords on the DA grid).")
    add("- Init: lagged-truth shared by the DA ensemble and the free-forecast "
        "reference; `disp_frac = 1.0` (background-error-scaled), band ±0.25 d.")
    add("- Lags tested: 1.0 d and 2.0 d.")
    add("")

    add("## 2. S0 results (1% obs noise, error-free)")
    add("")
    add("| obs | cols | lag | DA RMSE | Free RMSE | improv | EV_full | EV_free |")
    add("|---|---|---|---|---|---|---|---|")
    for obsvar in ("q", "psi"):
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

    add("## 3. S1 results — cross-resolution (da_nx=16 vs da_nx=32)")
    add("")
    add("### Headline (S1, psi-obs, cols=4, 1% noise, truth 64×64)")
    add("")
    add("| da_nx | ratio | lag | DA RMSE | Free RMSE | improv | EV_full | EV_free |")
    add("|---|---|---|---|---|---|---|---|")
    for da_nx, ratio, label in ((16, "4:1", "qg_s1"), (32, "2:1", "qg_s1_da32")):
        for lag in (1.0, 2.0):
            d = find_json_dir(root, _lag_dir(label, lag))
            if d is None:
                missing.append(f"{_lag_dir(label, lag)}")
                continue
            s1 = d["scenarios"]["test_s1"]
            add(f"| {da_nx} | {ratio} | {lag:.1f} | "
                f"{fmt_rmse(s1['rmse_mean'])} | {fmt_rmse(s1['forecast_rmse_mean'])} | "
                f"{fmt_improv(s1['forecast_improvement'])} | "
                f"{fmt_ev(s1['expvar_full'])} | {fmt_ev(s1['expvar_free'])} |")
    add("")
    add("### Per-field metrics")
    add("")
    for title, label in (
        ("S1 @ da_nx=16", "qg_s1"),
        ("S1 @ da_nx=32", "qg_s1_da32"),
    ):
        add(f"#### {title}")
        add("")
        for lag in (1.0, 2.0):
            d = find_json_dir(root, _lag_dir(label, lag))
            if d is None:
                continue
            s1 = d["scenarios"]["test_s1"]
            add(f"**lag {lag:.1f}**")
            add("")
            add("| field | layer | DA RMSE | Free RMSE | improv | EV | EV_free |")
            add("|---|---|---|---|---|---|---|")
            mpf = s1["metrics_per_field"]
            for fld in ("q", "psi"):
                for lyr in ("layer1", "layer2", "full"):
                    m = mpf[fld][lyr]
                    add(f"| {FIELD_LABELS[fld]} | {LAYER_LABELS[lyr]} | "
                        f"{fmt_rmse(m['rmse'])} | {fmt_rmse(m['rmse_free'])} | "
                        f"{fmt_improv(m['improv'])} | {fmt_ev(m['ev'])} | "
                        f"{fmt_ev(m['ev_free'])} |")
            add("")

    add("## 4. Interpretation")
    add("")
    add("- **Milder resolution mismatch (da_nx=32) is much easier DA than "
        "da_nx=16**: forecast-improv rises ~1.10 → ~1.48 and pooled EV flips "
        "from negative to positive.")
    add("- At da_nx=32 every layer beats the free forecast (`improv > 1`), with "
        "the streamfunction (ψ) most informative (upper-ψ improv ≈ 1.6–1.7; "
        "q₁ improv ≈ 1.49).")
    add("- The DA-vs-free-forecast skill scales monotonically with the "
        "cross-resolution ratio (4:1 → 2:1): the free forecast itself diverges "
        "strongly under model error (EV_free < 0), and DA recovers a large "
        "share of that signal.")
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
