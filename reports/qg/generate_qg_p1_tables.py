"""QG cells of the P1 paper: perfect- and imperfect-model table rows and the tau-resolved series.

Reads the per-window score files (`shard_*_meta.json`) of the gyrostat benchmark rows and writes

- `reports/qg/outputs/qg_p1/qg_p1_rows.json`: per scheme and case, EV of psi1 / psi2 and their
  mean (the paper's EV: MSE pooled over the windows, over the climatological variance, so that
  FMSS_0 = EV up to the time sub-sampling of the flow-matching scores; `ev_window_psi` is the
  per-window EV of the gyrostat report, kept for cross-reference), CRPS skill score of psi against the climatological forecast (`crpss_psi`; for a
  deterministic scheme the MAE skill, from its exact absolute error when stored or from its RMSE
  assuming Gaussian errors, flagged `mae_from_rmse`), and the pooled spread-skill ratio `ss_psi`;
  `ev_q` is the EV of the potential vorticity (q1, q2) and `ev_free_psi` that of the free run of
  the DA model; `large_ensemble` holds the EV of the 320-member ETKF and ETKS (the ensemble-size
  contrast) and `s1_s0_tuned` the S1 ensembles with their S0 settings (the re-tuning contrast);
- `reports/qg/outputs/qg_p1/qg_p1_series.json`: FMSS_tau, SS_tau = 1/sqrt(c*(tau)), its grid-edge
  flag, the calibration loss L_tau and the shape diagnostic G_tau, over psi1 and psi2.

The climatological references (`qg_p1_climatology.json`, from `compute_qg_p1_climatology.py`) are
the CRPS of the climatological forecast per field and the climatological mean and standard
deviation that standardise the flow-matching scores (`--fm-clim` of the QG scorers).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST = "experiments/qg_specwind_da/qg_specwind_gyrostat_v1/test"
OUT = os.path.join(REPO, "reports/qg/outputs/qg_p1")
PSI = ("psi1", "psi2")
TAUS = (0.0, 0.25, 0.5, 0.75)
CS = np.exp(np.linspace(np.log(1 / 16), np.log(16), 49))

ROWS = {
    "s0": {
        "ETKF": ("etkf_c3_loc8.0_r0.1_x1.0_bred_ensrf_d1.0_k0_s0_n100_fms", "ensemble"),
        "EnKF": ("enkf_c3_loc6.0_r0.1_x1.0_bred_d1.0_k0_s0_n100_fms", "ensemble"),
        "ETKS": ("enks_c3_loc8.0_r0.1_x1.0_bred_lag0_tau8_d1.0_k0_s0_n100_fms", "ensemble"),
        "Strong-4DVar": ("strong4dvar_c3_loc8.0_r0.1_x1.0_bred_d1.0_k0_Bspectral1.0_W120_s0_n100", "rmse"),
        "D-Unet": ("neural_G1L3D_b16_w15_kpow_s0_full_c3_cubic_N1", "deterministic"),
        "CFM": ("neural_G2_3d_w15_kpow_s0_fms_c3_cubic_N30", "ensemble"),
        "SDA": ("neural_G3_3d_w15_pub_g0.01_st100_c1_fms_c3_cubic_N30", "ensemble"),
        "D-Unet+SDA": ("neural_G3_3d_w15_warmG1L3Dfull_t0.5_gw2_st10_fms_c3_cubic_N30", "ensemble"),
    },
    "s1": {
        "ETKF": ("etkf_c3_loc8.0_r0.1_x1.0_bred_ensrf_d1.0_k1_rbase_R6.0_s0_n100_fms", "ensemble"),
        "EnKF": ("enkf_c3_loc6.0_r0.1_x1.0_bred_d1.0_k1_rbase_R6.0_s0_n100_fms", "ensemble"),
        "ETKS": ("enks_c3_loc8.0_r0.1_x1.0_bred_lag0_tau8_d1.0_k1_rbase_R6.0_s0_n100_fms", "ensemble"),
        "Strong-4DVar": ("strong4dvar_c3_loc8.0_r0.1_x1.0_bred_d1.0_k1_rbase_Bspectral1.0_W120_s0_n100", "rmse"),
        "Weak-4DVar": ("weak4dvar_c3_loc8.0_r0.1_x1.0_bred_d1.0_k1_rbase_Bspectral1.0_Q0.1_W120_s0_n100", "rmse"),
        "D-Unet": ("neural_G1L3D_b16_w15_kpow_s0_full_c3_cubic_N1_k1_rbase", "deterministic"),
        "CFM": ("neural_G2_3d_w15_kpow_s0_fms_c3_cubic_N30_k1_rbase", "ensemble"),
        "SDA": ("neural_G3M_3d_w15_pub_g0.01_st100_c1_fms_c3_cubic_N30_k1_rbase", "ensemble"),
        "D-Unet+SDA": ("neural_G3_3d_w15_warmG1L3Dfull_t0.5_gw2_st10_fms_c3_cubic_N30_k1_rbase", "ensemble"),
    },
}
LARGE = {
    "s0": {
        "ETKF": "etkf_c3_loc12.0_r0.1_x1.0_bred_ensrf_d1.0_k0_N320_s0_n100",
        "ETKS": "enks_c3_loc12.0_r0.1_x1.0_bred_lag0_tau16_d1.0_k0_N320_s0_n100",
    },
    "s1": {
        "ETKF": "etkf_c3_loc8.0_r0.1_x1.0_bred_ensrf_d1.0_k1_rbase_R6_N320_s0_n100",
        "ETKS": "enks_c3_loc8.0_r0.1_x1.0_bred_lag0_tau8_d1.0_k1_rbase_R6_N320_s0_n100",
    },
}
S1_S0_TUNED = {
    "ETKF": "etkf_c3_loc8.0_r0.1_x1.0_bred_ensrf_d1.0_k1_rbase_s0_n100",
    "EnKF": "enkf_c3_loc6.0_r0.1_x1.0_bred_d1.0_k1_rbase_s0_n100",
    "ETKS": "enks_c3_loc8.0_r0.1_x1.0_bred_lag0_tau8_d1.0_k1_rbase_s0_n100",
}


def alpha(tau: float) -> float:
    return (tau / (1 - tau)) ** 2


def load_windows(run_dir: str) -> list[dict]:
    rows: dict[int, dict] = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "shard_*_meta.json"))):
        for r in json.load(open(f))["per_window"]:
            rows[int(r["index"])] = r
    return [rows[i] for i in sorted(rows)]


def explained_variance(windows: list[dict], clim: dict, fields=PSI, tag: str = "da") -> dict:
    sd = clim["mean_std"]
    return {f: 1 - float(np.mean([w[f"rmse_{tag}_{f}"] ** 2 for w in windows])) / sd[f]["std"] ** 2 for f in fields}


def ev_extras(windows: list[dict], clim: dict) -> dict:
    out = {"ev_q": float(np.mean(list(explained_variance(windows, clim, ("q1", "q2")).values())))}
    if "rmse_free_psi1" in windows[0]:
        out["ev_free_psi"] = float(np.mean(list(explained_variance(windows, clim, PSI, "free").values())))
    return out


def table_row(windows: list[dict], kind: str, clim: dict) -> dict:
    ev = explained_variance(windows, clim)
    ev_window = {f: float(np.mean([w[f"ev_da_{f}"] for w in windows])) for f in PSI}
    if kind == "rmse":
        crps = {f: float(np.sqrt(2 / np.pi) * np.mean([w[f"rmse_da_{f}"] for w in windows])) for f in PSI}
    else:
        crps = {f: float(np.mean([w[f"crps_{f}"] for w in windows])) for f in PSI}
    skill = {f: 1 - crps[f] / clim["crps_clim"][f] for f in PSI}
    out = {"n_windows": len(windows), "ev": ev, "ev_psi": float(np.mean(list(ev.values()))),
           "ev_window_psi": float(np.mean(list(ev_window.values()))), **ev_extras(windows, clim),
           "crpss": skill, "crpss_psi": float(np.mean(list(skill.values()))),
           "deterministic": kind != "ensemble", "mae_from_rmse": kind == "rmse", "ss_psi": None}
    if kind == "ensemble" and "ss_psi1_var" in windows[0]:
        var = sum(w[f"ss_{f}_var"] for w in windows for f in PSI)
        err2 = sum(w[f"ss_{f}_err2"] for w in windows for f in PSI)
        out["ss_psi"] = float(np.sqrt(var / err2))
    return out


def pooled(windows: list[dict], key: str, f: str):
    n = np.array([w[f"fm_{f}_n"] for w in windows], float)
    v = np.array([w[key.format(f=f)] for w in windows], float)
    return (v * (n if v.ndim == 1 else n[:, None])).sum(0) / n.sum()


def series_row(windows: list[dict], kind: str, clim: dict) -> dict | None:
    if kind == "ensemble":
        if "fm_psi1_gauss_0.0" not in windows[0]:
            return None
        fms = [float(np.mean([pooled(windows, f"fm_{{f}}_gauss_{t}", f) for f in PSI])) for t in TAUS]
        ss, edge, loss, g = [], [], [], []
        var = sum(w[f"ss_{f}_var"] for w in windows for f in PSI)
        err2 = sum(w[f"ss_{f}_err2"] for w in windows for f in PSI)
        for t in TAUS:
            gc = np.sum([pooled(windows, f"fm_{{f}}_gc_{t}", f) for f in PSI], 0)
            k = int(np.argmin(gc))
            c1 = float(np.sum([pooled(windows, f"fm_{{f}}_gauss_{t}", f) for f in PSI]))
            if t == 0:
                ss.append(float(np.sqrt(var / err2)))
                edge.append(False)
                loss.append(0.0)
                g.append(0.0)
                continue
            ss.append(float(CS[k] ** -0.5))
            edge.append(k in (0, len(CS) - 1))
            loss.append(float(1 - gc[k] / c1))
            if "fm_psi1_kde_0.25" in windows[0]:
                kde = float(np.sum([pooled(windows, f"fm_{{f}}_kde_{t}", f) for f in PSI]))
                kv = float(np.sum([pooled(windows, f"fm_{{f}}_kdevar_{t}", f) for f in PSI]))
                g.append(1 - kde / kv)
            else:
                g.append(None)
        out = {"fms": fms, "ss": ss, "ss_edge": edge, "loss": loss, "g": g}
    else:
        sd = clim["mean_std"]
        mse = float(np.mean([np.mean([w[f"rmse_da_{f}"] ** 2 for w in windows]) / sd[f]["std"] ** 2 for f in PSI]))
        out = {"fms": [mse] * len(TAUS), "ss": None, "ss_edge": None, "loss": None, "g": None}
    out["fmss"] = [1 - f * (1 + alpha(t)) for f, t in zip(out["fms"], TAUS)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=os.path.join(REPO, TEST), help="directory of the benchmark rows")
    p.add_argument("--clim", default=os.path.join(OUT, "qg_p1_climatology.json"))
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    clim = json.load(open(args.clim))
    rows, series = {}, {}
    for case, schemes in ROWS.items():
        rows[case], series[case] = {}, {}
        for name, (d, kind) in schemes.items():
            w = load_windows(os.path.join(args.root, d))
            if len(w) != 100:
                raise SystemExit(f"{case} {name}: {len(w)} windows in {d}")
            rows[case][name] = {"run": d, **table_row(w, kind, clim)}
            s = series_row(w, kind, clim)
            if s is not None:
                series[case][name] = s
    large = {}
    for case, schemes in LARGE.items():
        large[case] = {}
        for name, d in schemes.items():
            w = load_windows(os.path.join(args.root, d))
            if len(w) != 100:
                raise SystemExit(f"{case} {name} N=320: {len(w)} windows in {d}")
            ev = explained_variance(w, clim)
            large[case][name] = {"run": d, "n_members": 320, "ev": ev, "ev_psi": float(np.mean(list(ev.values())))}
    s0_tuned = {}
    for name, d in S1_S0_TUNED.items():
        w = load_windows(os.path.join(args.root, d))
        if len(w) != 100:
            raise SystemExit(f"s1 {name} S0-tuned: {len(w)} windows in {d}")
        ev = explained_variance(w, clim)
        s0_tuned[name] = {"run": d, "ev": ev, "ev_psi": float(np.mean(list(ev.values()))), **ev_extras(w, clim)}
    os.makedirs(args.out, exist_ok=True)
    json.dump({"fields": list(PSI), "rows": rows, "large_ensemble": large, "s1_s0_tuned": s0_tuned}, open(os.path.join(args.out, "qg_p1_rows.json"), "w"), indent=1)
    json.dump({"taus": list(TAUS), "cases": series}, open(os.path.join(args.out, "qg_p1_series.json"), "w"), indent=1)
    for case in rows:
        for name, r in rows[case].items():
            ss = "" if r["ss_psi"] is None else f"  SS {r['ss_psi']:.2f}"
            mark = "(det)" if r["deterministic"] else ""
            print(f"{case} {name:12s} EV {r['ev_psi']:.3f}  CRPSS {r['crpss_psi']:.3f}{mark}{ss}")


if __name__ == "__main__":
    main()
