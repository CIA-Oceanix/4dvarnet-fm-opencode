"""Figures of the P1 paper (docs/papers/p1_structural_hypotheses/figures/).

    python reports/p1/p1_figures.py series    # L96 tau-series -> reports/p1/outputs/l96_tau_series.json
    python reports/p1/p1_figures.py extras    # L96 ETKS-100 scores, filter/smoother identity along tau, the
                                              # S1-mean/S0-spread counterfactual and the SS / FMSS of the 2D
                                              # learned rows (3 seeds) -> reports/p1/outputs/l96_extras.json
    python reports/p1/p1_figures.py obs       # EV of the benchmark schemes on the observing-system cells
                                              # -> reports/p1/outputs/l96_obs_system.json
    python reports/p1/p1_figures.py figures   # the seven figures -> reports/p1/outputs/figures/
    python reports/p1/p1_figures.py all

Inputs:
- L96: the report bundle (reports/l96/_inputs.py roots `benchmark_extended/here` and `p1_benchmark/here`,
  including the 100-member ETKF/ETKS trajectories `etks100_traj_2026-11-08`),
  the committed flow-matching summaries `reports/l96/outputs/fm_score/current/*.json`, and the pooled
  skill table of `reports/l96/outputs/p1_l96_benchmark.md` (paper tables, 2D rows).
- QG: `reports/qg/outputs/qg_p1/qg_p1_rows.json` and `qg_p1_series.json` (reports/qg/generate_qg_p1_tables.py).

Figures: fig_budget_s0/s1 (error budgets, Sects. 5.1-5.2), fig_fmss, fig_ss_tau, fig_g_tau (Sect. 5.3),
fig_missp_tau (model error along tau, Sect. 5.3), fig_obs_system (observing system, Sect. 5.5). Copy the
PDFs to the paper folder (and Overleaf) to update the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports" / "l96"))
import _inputs  # noqa: E402

OUT = REPO / "reports" / "p1" / "outputs"
CUR = REPO / "reports" / "l96" / "outputs" / "fm_score" / "current"
QG = REPO / "reports" / "qg" / "outputs" / "qg_p1"
L96_SKILL_MD = REPO / "reports" / "l96" / "outputs" / "p1_l96_benchmark.md"
TAUS = [0.0, 0.25, 0.5, 0.75]
CS = np.exp(np.linspace(np.log(1 / 16), np.log(16), 49))
IDX = list(range(8)) + [8 + k * 4 + j for k in range(8) for j in range(2)]
DA_TAG = ("da_reg_l96_baselines_trajs_dws500_s0c_test_etks-correct-Lfull-infs0-1.15_s1-2.5_infs0-1.2_s1-3.0"
          "_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw")
ETKS100_TRAJ = "etks100_traj_2026-11-08/etks100_test_regular_s0_lam1.05_trajectories.npz"

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
STYLE = {"ETKF": ("#2a78d6", "o"), "EnKF": ("#eb6834", "s"), "ETKS": ("#1baf7a", "^"),
         "CFM": ("#eda100", "D"), "SDA": ("#e87ba4", "v"), "D-Unet+SDA": ("#008300", "P")}


def alpha(t: float) -> float:
    return (t / (1 - t)) ** 2


def _sel(a: np.ndarray) -> np.ndarray:
    return a[..., IDX] if a.shape[-1] == 40 else a


# ---------------------------------------------------------------- L96 tau-series

def _from_json(paths: list[Path]) -> dict:
    rows = []
    for p in paths:
        d = json.loads(p.read_text())
        sc = d["scale"]["all"]
        fms = [d["gauss"][str(t)]["all"] for t in TAUS]
        ss = [sc["pooled_spread_skill"]] + [sc[str(t)]["c_star"] ** -0.5 for t in TAUS[1:]]
        edge = [False] + [abs(np.log(sc[str(t)]["c_star"]) - np.log(8)) < 1e-6 for t in TAUS[1:]]
        g = [0.0] + [1 - d["kde"][str(t)]["all"] / sc[str(t)]["kde_var"] for t in TAUS[1:]]
        loss = [0.0] + [1 - sc[str(t)]["at_c_star"] / sc[str(t)]["c1"] for t in TAUS[1:]]
        rows.append((fms, ss, g, loss, edge))
    fms, ss, g, loss = (np.mean([r[i] for r in rows], 0).tolist() for i in range(4))
    edge = np.any([r[4] for r in rows], 0).tolist()
    return {"fms": fms, "ss": ss, "g": g, "loss": loss, "ss_edge": edge}


def _from_mean_var(mean: np.ndarray, var: np.ndarray | None, truth: np.ndarray, sd: np.ndarray) -> dict:
    e2 = ((mean - truth) / sd) ** 2
    out = {"fms": [], "ss": [], "g": None, "ss_edge": [], "loss": []}
    if var is None:
        return {"fms": [float(e2.mean())] * len(TAUS), "ss": None, "g": None, "ss_edge": [], "loss": None}
    v = var / sd ** 2
    for t in TAUS:
        a = alpha(t)

        def f(c: float) -> float:
            return float(((e2 + a * (c * v) ** 2) / (1 + a * c * v) ** 2).mean())
        out["fms"].append(f(1.0))
        if t == 0:
            out["ss"].append(float(np.sqrt(v.mean() / e2.mean())))
            out["ss_edge"].append(False)
            out["loss"].append(0.0)
        else:
            sc = [f(c) for c in CS]
            k = int(np.argmin(sc))
            out["ss"].append(float(CS[k] ** -0.5))
            out["ss_edge"].append(k in (0, len(CS) - 1))
            out["loss"].append(float(1 - sc[k] / f(1.0)))
    return out


def l96_series() -> Path:
    from reports.l96.probe_fm_score import norm_stats
    ex = _inputs.root("benchmark_extended", "here")
    p1 = _inputs.root("p1_benchmark", "here")
    sd = _sel(norm_stats()[1])
    rows = {
        "ETKF": lambda c: [CUR / f"{DA_TAG}_{c}_ETKF_members.json"],
        "ETKS": lambda c: [CUR / f"{DA_TAG}_{c}_ETKS_members.json"],
        "CFM": lambda c: [CUR / f"vanillacfm_ep1200_seed{s}_reg_{c}.json" for s in (1, 2, 3)],
        "SDA": lambda c: [CUR / f"A3_sda2_monaiM_ep1200_l96_seed{s}_reg_{c}.json" for s in (1, 2, 3)],
        "D-Unet+SDA": lambda c: [CUR / f"hybrid_DU1200s{s}_sda3fix1200s1_reg_{c}.json" for s in (1, 2, 3)],
    }
    enkf = np.load(ex / "da_inflation_2026-09-28" / ("l96_baselines_trajectories_dws500_s0c_crps_infs0-1.2_s1-3.0"
                                                     "_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz"))
    strong = np.load(p1 / "da_current_2026-09-29" / "l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw_dafw.npz")
    res = {}
    for case in ("s0", "s1"):
        truth = np.load(ex / "L96B_directunet_monaiM_ep1200_seed1" / "ens1_no1" / f"estimates_{case}.npz")["truth"]
        truth = truth.astype(np.float64)
        var_c = float((truth.reshape(-1, 24).var(0) / sd ** 2).mean())
        d = {k: _from_json(f(case)) for k, f in rows.items()}
        d["EnKF"] = _from_mean_var(_sel(enkf[f"{case}_EnKF_trajectories"]), _sel(enkf[f"{case}_EnKF_ensemble_variance"]),
                                   truth, sd)
        du = [_from_mean_var(np.load(ex / f"L96B_directunet_monaiM_ep1200_seed{s}" / "ens1_no1" / f"estimates_{case}.npz")
                             ["trajectories"].astype(np.float64), None, truth, sd) for s in (1, 2, 3)]
        d["D-Unet"] = {**du[0], "fms": np.mean([r["fms"] for r in du], 0).tolist()}
        d["Strong-4DVar"] = _from_mean_var(_sel(strong[f"{case}_Strong_4DVar_trajectories"]), None, truth, sd)
        clim = [var_c / (1 + alpha(t) * var_c) for t in TAUS]
        for k in d:
            d[k]["fmss"] = [1 - f / c for f, c in zip(d[k]["fms"], clim)]
        res[case] = d
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "l96_tau_series.json"
    path.write_text(json.dumps({"taus": TAUS, "cases": res}, indent=1) + "\n")
    return path


# ---------------------------------------------------------------- L96 extras (tables and text)

def _gauss_fms(e2: np.ndarray, v: np.ndarray, t: float) -> float:
    a = alpha(t)
    return float(((e2 + a * v ** 2) / (1 + a * v) ** 2).mean())


def _identity_along_tau(mf: np.ndarray, vf: np.ndarray, ms: np.ndarray, vs: np.ndarray, truth: np.ndarray) -> dict:
    """Filter/smoother identity at every tau (standardised units), Proposition B.4 applied to (x_tau, S).

    For exact posteriors, FMS_F - FMS_S = mmse_F - mmse_S = E||Psi_S - Psi_F||^2. Gaussian reports:
    mmse_q(tau) = v / (1 + alpha v) per element; Psi = m + k (y - m), k = alpha v / (1 + alpha v),
    y = x_tau / tau, and the shift is evaluated with y drawn from the smoother's predictive N(m_S, v_S + 1/alpha),
    so that only the "score" side uses the truth.
    """
    out = {"score_gap": [], "mmse_gap": [], "shift": []}
    ef2, es2 = (mf - truth) ** 2, (ms - truth) ** 2
    for t in TAUS:
        a = alpha(t)
        out["score_gap"].append(_gauss_fms(ef2, vf, t) - _gauss_fms(es2, vs, t))
        out["mmse_gap"].append(float((vf / (1 + a * vf) - vs / (1 + a * vs)).mean()))
        kf, ks = a * vf / (1 + a * vf), a * vs / (1 + a * vs)
        noise = 0.0 if t == 0 else float(((ks - kf) ** 2 * (vs + 1 / a)).mean())
        out["shift"].append(float((((1 - kf) * (ms - mf)) ** 2).mean()) + noise)
    return out


def l96_extras() -> Path:
    from reports.l96.probe_fm_score import norm_stats
    ex = _inputs.root("benchmark_extended", "here")
    p1 = _inputs.root("p1_benchmark", "here")
    sd = _sel(norm_stats()[1])
    da30 = np.load(p1 / "da_current_2026-09-29" / ("l96_baselines_trajectories_dws500_s0c_test_etks-correct-Lfull"
                                                    "-infs0-1.15_s1-2.5_infs0-1.2_s1-3.0_etkf_infs0-1.15_s1-2.5"
                                                    "_obsj2_int100_fw_dafw.npz"))
    enkf = np.load(ex / "da_inflation_2026-09-28" / ("l96_baselines_trajectories_dws500_s0c_crps_infs0-1.2_s1-3.0"
                                                     "_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz"))
    lam = {"s0": "1.05", "s1": "2.5"}
    res = {"etks100": {}, "identity": {}, "counterfactual": {}, "learned2d": {}}
    reg = ex / "eval_regular"
    learned2d = {  # paper rows (2D backbone, 3 seeds, scores averaged over seeds)
        "CFM": [ex / f"L96B_vanillacfm_monai2dM_ep1200_seed{s}" / "ens30_no20_var" for s in (1, 2, 3)],
        "SDA1": [reg / f"L96B_sda1_monai2dM_ep1200_seed{s}" / "ens30_gw25_var" for s in (1, 2, 3)],
        "SDA2": [reg / f"L96B_sda2_monai2dM_ep1200_seed{s}" / "ens30_gw25_var" for s in (1, 2, 3)],
        "SDA3": [reg / f"L96B_sda3fix_monai2dM_ep1200_seed{s}" / "ens30_gw25_var" for s in (1, 2, 3)],
        "D-Unet+SDA": [reg / f"hybrid_DU2dMs{s}_L96B_sda3fix_monai2dM_ep1200_seed1" / "tau0.1_gw1_var" for s in (1, 2, 3)],
        "D-Unet+SDA1": [reg / f"hybrid_DU2dMs{s}_L96B_sda1_monai2dM_ep1200_seed1" / "tau0.1_gw1_var" for s in (1, 2, 3)]}
    mv = {}
    for case in ("s0", "s1"):
        truth = np.load(ex / "L96B_directunet_monaiM_ep1200_seed1" / "ens1_no1" / f"estimates_{case}.npz")["truth"]
        truth = truth.astype(np.float64)
        var_c = float((truth.reshape(-1, 24).var(0) / sd ** 2).mean())
        clim = [var_c / (1 + alpha(t) * var_c) for t in TAUS]
        z = np.load(p1 / "etks100_traj_2026-11-08" / f"etks100_test_regular_{case}_lam{lam[case]}_trajectories.npz")
        assert np.allclose(z[f"{case}_truth"], truth, atol=1e-4)
        tz = truth / sd

        def std(npz, key: str) -> tuple[np.ndarray, np.ndarray]:
            return (_sel(npz[f"{case}_{key}_trajectories"]).astype(np.float64) / sd,
                    np.clip(_sel(npz[f"{case}_{key}_ensemble_variance"]).astype(np.float64), 0, None) / sd ** 2)
        r = _from_mean_var(*(a * b for a, b in zip(std(z, "ETKS"), (sd, sd ** 2))), truth, sd)
        r["fmss"] = [1 - f / c for f, c in zip(r["fms"], clim)]
        res["etks100"][case] = r
        res["identity"][case] = {f"{n}_members": _identity_along_tau(*std(npz, "ETKF"), *std(npz, "ETKS"), tz)
                                 for n, npz in ((30, da30), (100, z))}
        res["learned2d"][case] = {}
        for k, dirs in learned2d.items():
            per_seed = []
            for d in dirs:
                est = np.load(d / f"estimates_{case}.npz")
                assert np.allclose(est["truth"], truth, atol=1e-4)
                r = _from_mean_var(est["trajectories"].astype(np.float64),
                                   np.load(d / f"scores_{case}.npz")["ensemble_variance"].astype(np.float64), truth, sd)
                r["fmss"] = [1 - f / c for f, c in zip(r["fms"], clim)]
                per_seed.append(r)
            res["learned2d"][case][k] = {
                **{m: np.mean([r[m] for r in per_seed], 0).tolist() for m in ("fms", "fmss", "ss", "loss")},
                "ss_edge": np.any([r["ss_edge"] for r in per_seed], 0).tolist(),
                "per_seed": {m: [r[m] for r in per_seed] for m in ("fmss", "ss")}, "n_seeds": len(per_seed)}
        mv[case] = {"ETKF": std(da30, "ETKF"), "EnKF": std(enkf, "EnKF"), "ETKS": std(da30, "ETKS"), "truth": tz}
    for k in ("ETKF", "EnKF", "ETKS"):
        (m0, v0), (m1, v1) = mv["s0"][k], mv["s1"][k]
        e0, e1 = (m0 - mv["s0"]["truth"]) ** 2, (m1 - mv["s1"]["truth"]) ** 2
        d0 = float(e1.mean() - e0.mean())
        res["counterfactual"][k] = {
            "actual": [(_gauss_fms(e1, v1, t) - _gauss_fms(e0, v0, t)) / d0 for t in TAUS],
            "s1_mean_s0_spread": [(_gauss_fms(e1, v0, t) - _gauss_fms(e0, v0, t)) / d0 for t in TAUS]}
    path = OUT / "l96_extras.json"
    path.write_text(json.dumps({"taus": TAUS, **res}, indent=1) + "\n")
    return path


# ---------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.labelcolor": INK})
    return plt


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=160)


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _fig_legend(fig, axes, ncol: int, y: float) -> None:
    h, lab = [], []
    for a in np.ravel(axes):
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="upper center", ncol=ncol, frameon=False, fontsize=7, bbox_to_anchor=(0.5, y))


def tau_figures(series: dict, out: Path) -> None:
    plt = _plt()
    ref = {"D-Unet": "--", "Strong-4DVar": ":"}
    spec = {"fmss": dict(ylabel=r"FMSS$_\tau$", ref=None, scale=1.0, refs=True, fname="fig_fmss"),
            "ss": dict(ylabel=r"SS$_\tau = 1/\sqrt{c^\star(\tau)}$", ref=1.0, scale=1.0, refs=False, fname="fig_ss_tau"),
            "g": dict(ylabel=r"$G_\tau$ (%)", ref=0.0, scale=100.0, refs=False, fname="fig_g_tau")}
    ylim = {("fmss", "L96", "s0"): (-0.4, 1.0), ("fmss", "L96", "s1"): (-1.8, 1.0)}
    for kind, sp in spec.items():
        fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, gridspec_kw={"hspace": 0.32, "wspace": 0.3})
        for i, case in enumerate(("s0", "s1")):
            for j, cs in enumerate(("L63", "L96", "QG")):
                ax = axes[i, j]
                _style_axes(ax)
                ax.set_title(f"{cs}, {'S0' if case == 's0' else 'S1'}", color=INK, fontsize=8.5, fontweight="bold")
                if i == 1:
                    ax.set_xlabel(r"flow time $\tau$")
                if j == 0:
                    ax.set_ylabel(sp["ylabel"])
                if cs not in series:
                    ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center", color=INK2)
                    ax.set_yticks([])
                    continue
                taus = series[cs]["taus"]
                d = series[cs]["cases"][case]
                if sp["ref"] is not None:
                    ax.axhline(sp["ref"] * sp["scale"], color=INK2, lw=0.9)
                if sp["refs"]:
                    for k, ls in ref.items():
                        if k in d and d[k].get(kind) is not None:
                            ax.plot(taus, np.asarray(d[k][kind]) * sp["scale"], ls=ls, color=INK2, lw=1.2,
                                    label=f"{k} (no uncertainty)")
                for k, (col, mk) in STYLE.items():
                    if k not in d or d[k].get(kind) is None:
                        continue
                    y = np.asarray(d[k][kind], float) * sp["scale"]
                    ax.plot(taus, y, color=col, lw=1.8, marker=mk, ms=5, mec="white", mew=0.8, label=k)
                    if kind == "ss":
                        for t, v, e in zip(taus, y, d[k].get("ss_edge") or []):
                            if e:
                                ax.plot([t], [v], marker=mk, ms=9, mfc="none", mec=col, mew=1.0)
                if (kind, cs, case) in ylim:
                    lo, hi = ylim[(kind, cs, case)]
                    ax.set_ylim(lo, hi)
                    v = d["Strong-4DVar"]["fmss"][-1]
                    if v < lo:
                        ax.annotate(f"Strong-4DVar: {v:.1f}", xy=(0.6, lo + 0.01), xytext=(0.0, lo + 0.06 * (hi - lo)),
                                    fontsize=6, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7))
                ax.set_xticks(taus)
        _fig_legend(fig, axes, 4, 1.06)
        _save(fig, out, sp["fname"])
        plt.close(fig)


def missp_figure(series: dict, out: Path, min_contrast: float = 0.005) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True, gridspec_kw={"hspace": 0.3, "wspace": 0.28})
    for j, cs in enumerate(("L96", "QG")):
        taus = series[cs]["taus"]
        c0, c1 = series[cs]["cases"]["s0"], series[cs]["cases"]["s1"]
        for i in range(2):
            _style_axes(axes[i, j])
        axes[0, j].axhline(1.0, color=INK2, lw=1.0, ls="--", label="no reported uncertainty")
        axes[1, j].axhline(0.0, color=INK2, lw=0.9)
        for k, (col, mk) in STYLE.items():
            if k not in c0 or k not in c1:
                continue
            f0, f1 = np.asarray(c0[k]["fms"]), np.asarray(c1[k]["fms"])
            d = f1 - f0
            if abs(d[0]) < min_contrast:
                continue
            kw = dict(color=col, lw=1.8, marker=mk, ms=5, mec="white", mew=0.8, label=k)
            axes[0, j].plot(taus, d / d[0], **kw)
            if c0[k].get("loss") and c1[k].get("loss"):
                cal = 100 * (f1 * np.asarray(c1[k]["loss"]) - f0 * np.asarray(c0[k]["loss"])) / d
                axes[1, j].plot(taus, cal, **kw)
        axes[0, j].set_title(cs, color=INK, fontsize=9, fontweight="bold")
        axes[0, j].set_ylim(0, 1.1)
        axes[1, j].set_xlabel(r"flow time $\tau$")
        axes[1, j].set_xticks(taus)
    axes[0, 0].set_ylabel(r"$\Delta\mathrm{FMS}_\tau \,/\, \Delta\mathrm{FMS}_0$")
    axes[1, 0].set_ylabel("scalar-calibration share (%)")
    _fig_legend(fig, axes, 7, 1.0)
    _save(fig, out, "fig_missp_tau")
    plt.close(fig)


def _l96_pooled_ev() -> dict:
    labels = {"ETKF": "ETKF", "EnKF": "EnKF", "ETKS": "ETKS", "Strong-4DVar": "Strong-4DVar",
              "Weak-4DVar": "Weak-4DVar", "D-Unet (DirectUNet-2D-M)": "D-Unet", "CFM (VanillaCFM-2D-M)": "CFM",
              "SDA2-2D-M": "SDA", "D-Unet+SDA (DirectUNet-2D-M -> SDA3-fix-2D-M)": "D-Unet+SDA"}
    block = L96_SKILL_MD.read_text().split("## Pooled skill scores", 1)[1]
    out = {"S0": {}, "S1": {}}
    for line in block.splitlines()[1:]:
        if line.startswith("## "):
            break
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        if len(cells) == 6 and cells[0] in labels and cells[2] != "pending":
            out["S0"][labels[cells[0]]] = float(cells[2])
            out["S1"][labels[cells[0]]] = float(cells[4])
    assert len(out["S0"]) == len(labels), out
    return out


def l96_ev_100_members() -> dict:
    """EV of the 100-member ETKF (the ETKS's own filter pass) and ETKS at S0, as in the paper tables:
    per observed channel, 1 - MSE / Var(truth) pooled over windows and times, averaged over channels."""
    z = np.load(_inputs.root("p1_benchmark", "here") / ETKS100_TRAJ)
    truth = z["s0_truth"].astype(np.float64)
    var_c = truth.reshape(-1, truth.shape[-1]).var(0)
    return {m: float(np.mean(1 - ((z[f"s0_{m}_trajectories"].astype(np.float64) - truth) ** 2).mean((0, 1)) / var_c))
            for m in ("ETKF", "ETKS")}


def budget_figures(out: Path) -> None:
    from matplotlib.patches import Patch
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    plt = _plt()
    plt.rcParams.update({"ytick.color": INK, "hatch.color": SURF, "hatch.linewidth": 0.8})
    colors = {"floor": "#b9b8b2", "unres": "#eda100", "restr": "#2a78d6", "approx": "#1baf7a", "missp": "#e34948"}
    schemes = ["ETKF", "EnKF", "ETKS", "Strong-4DVar", "Weak-4DVar", "D-Unet", "CFM", "SDA", "D-Unet+SDA"]
    learned = {"D-Unet", "CFM", "SDA", "D-Unet+SDA"}
    qg = json.loads((QG / "qg_p1_rows.json").read_text())
    l96 = _l96_pooled_ev()
    ev = {"L96": {"S0": l96["S0"], "S1": l96["S1"], "big": l96_ev_100_members()},
          "QG": {"S0": {k: v["ev_psi"] for k, v in qg["rows"]["s0"].items()},
                 "S1": {k: v["ev_psi"] for k, v in qg["rows"]["s1"].items()},
                 "big": {k: v["ev_psi"] for k, v in qg["large_ensemble"]["s0"].items()}}}

    def segments(case: str, s: str, with_s1: bool):
        d = ev[case]
        if s not in d["S0"]:
            return None
        n0 = {k: 1 - v for k, v in d["S0"].items()}
        floor = min(n0.values())
        big = {k: 1 - v for k, v in d["big"].items()}
        if s == "ETKS":
            seg = [("floor", floor), ("unres", big["ETKS"] - floor), ("approx", n0["ETKS"] - big["ETKS"])]
        elif s == "ETKF":
            seg = [("floor", floor), ("unres", big["ETKS"] - floor), ("restr", big["ETKF"] - big["ETKS"]),
                   ("approx", n0["ETKF"] - big["ETKF"])]
        else:
            seg = [("floor", floor), ("unres", n0[s] - floor)]
        if with_s1 and s in d["S1"]:
            seg.append(("missp", max(0.0, (1 - d["S1"][s]) - n0[s])))
        return seg

    for with_s1, name in ((False, "fig_budget_s0"), (True, "fig_budget_s1")):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True,
                                 gridspec_kw={"width_ratios": [0.55, 1, 1], "wspace": 0.26})
        y = list(range(len(schemes)))[::-1]
        for ax, case in zip(axes, ["L63", "L96", "QG"]):
            ax.set_facecolor(SURF)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(case, color=INK, fontsize=9, fontweight="bold")
            if case not in ev:
                ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center", color=INK2)
                ax.set_xticks([])
                continue
            xmax = 0
            for yi, s in zip(y, schemes):
                seg = segments(case, s, with_s1)
                if seg is None:
                    if with_s1 and s in ev[case]["S1"]:
                        tot = 1 - ev[case]["S1"][s]
                        ax.barh(yi, tot, height=0.62, color=SURF, edgecolor=colors["missp"], hatch="////", lw=0.8)
                        ax.text(tot, yi, f" {tot:.3f} (no S0 run)", va="center", color=INK2, fontsize=6.5)
                        xmax = max(xmax, tot)
                    continue
                left = 0
                for k, w in seg:
                    ax.barh(yi, w, left=left, height=0.62, color=colors[k], edgecolor=SURF, lw=1.0,
                            hatch="////" if k == "missp" else None)
                    left += w
                ax.text(left, yi, f" {left:.3f}" + (" oracle" if with_s1 and s in learned else ""), va="center",
                        color=INK2, fontsize=6.5)
                xmax = max(xmax, left)
            ax.set_xlim(0, xmax * 1.32)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f" if xmax > 0.05 else "%.3f"))
        fig.supxlabel("normalised MSE  (1 $-$ EV)", fontsize=8, color=INK, x=0.5, y=-0.02)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(schemes)
        keys = [("floor", "floor: best S0 scheme (upper bound on irreducible)"),
                ("unres", "residual (misspecification + cross term + remaining approximation)"),
                ("restr", "restriction: filtering (ETKF $-$ ETKS, large ensembles)"),
                ("approx", "approximation: ensemble size")]
        if with_s1:
            keys.append(("missp", "misspecification + cross term (S1 $-$ S0)"))
        handles = [Patch(facecolor=colors[k], edgecolor=SURF, hatch="////" if k == "missp" else None, label=t)
                   for k, t in keys]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.5, 1.17 if with_s1 else 1.13), labelcolor=INK)
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=170)
        plt.close(fig)


OBS_CELLS = {
    "a": ("number of observation times", [(n, f"rlayout_n{n}-{n}_k16-16_w20_d3") for n in (6, 10, 20, 30, 50, 75, 100, 300, 1000)],
          (10, 300), True),
    "b": ("observed fast variables", [(k, f"rlayout_n30-30_k{k}-{k}_w20_d3") for k in (0, 2, 4, 8, 12, 16)], (4, 16), False),
    "c": ("observation-error variance", [(0.25, "rlayout_n30-30_k16-16_w20_d3_r0.25"), (0.5, "rlayout_n30-30_k16-16_w20_d3"),
                                         (1.0, "rlayout_n30-30_k16-16_w20_d3_r1")], (0.5, 0.5), True),
}
OBS_DA = ("ETKF", "EnKF", "ETKS", "Strong-4DVar")
OBS_LEARNED = {"D-Unet": ("L96B_directunet2d_monaiM_seed1", "ens1_no1"),
               "CFM": ("L96B_vanillacfm_monai2dM_ep1200_seed1", "ens30_no20"),
               "SDA": ("L96B_sda2_monai2dM_ep1200_seed1", "ens30_gw25"),
               "D-Unet+SDA": ("hybrid_DU2dMs1_L96B_sda3fix_monai2dM_ep1200_seed1", "tau0.1_gw1")}
OBS_DA_INF = "_infs0-1.2_s1-3.0_etkf_infs0-1.15_s1-2.5"


def _one(d: Path, pattern: str) -> Path:
    hits = sorted(d.glob(pattern))
    assert len(hits) == 1, (d, pattern, hits)
    return hits[0]


def _ev(est: np.ndarray, truth: np.ndarray) -> float:
    """Paper EV on the 24 observed channels: per channel 1 - MSE / Var(truth), pooled over the cell's runs and
    times, averaged over channels."""
    t = truth.reshape(-1, truth.shape[-1])
    return float(np.mean(1 - ((est - truth) ** 2).reshape(t.shape).mean(0) / t.var(0)))


def l96_obs_system() -> Path:
    """EV of the benchmark schemes on the cells of the observing-system figure -> l96_obs_system.json."""
    here = _inputs.root("benchmark_extended", "here")
    import torch
    data = torch.load(_inputs.root("p1_benchmark", "here") / "l96_datasets_obsj2_int100_nwin200.pt", weights_only=False)
    da_dir = here / "obs_system_da"
    res = {}
    for case, key in (("s0", "test_s0"), ("s1", "test_s1")):
        res[case] = {}
        for _, cells, _, _ in OBS_CELLS.values():
            for _, c in cells:
                if c in res[case]:
                    continue
                pw = np.load(_one(da_dir / "l96_da_random_layout", f"per_window_{c}_etks-*{OBS_DA_INF}.npz"))
                truth = np.stack([data[key][int(w)]["true_state"].numpy()[:, IDX] for w in pw[f"{case}_window_index"]])
                truth = truth.astype(np.float64)
                z = np.load(_one(da_dir, f"l96_baselines_trajectories_dws500_{c}_etks-*{OBS_DA_INF}_obsj2_fw_dafw.npz"))
                row = {m: _ev(_sel(z[f"{case}_{m.replace('-', '_')}_trajectories"]).astype(np.float64), truth) for m in OBS_DA}
                for lab, (run, sub) in OBS_LEARNED.items():
                    e = np.load(here / "eval_obs_system" / c / run / sub / f"estimates_{case}.npz")
                    assert np.allclose(e["truth"], truth, atol=1e-4), (c, run, case)
                    row[lab] = _ev(e["trajectories"].astype(np.float64), truth)
                res[case][c] = row
    path = OUT / "l96_obs_system.json"
    path.write_text(json.dumps({"cells": {k: [[x, c] for x, c in v[1]] for k, v in OBS_CELLS.items()}, "ev": res},
                               indent=1) + "\n")
    return path


def obs_system_figure(out: Path) -> None:
    plt = _plt()
    ev = json.loads((OUT / "l96_obs_system.json").read_text())["ev"]
    style = {"ETKF": ("#2a78d6", "o", "-"), "EnKF": ("#eb6834", "s", "-"), "ETKS": ("#1baf7a", "^", "-"),
             "Strong-4DVar": ("#52514e", "x", ":"), "D-Unet": ("#52514e", "+", "--"), "CFM": ("#eda100", "D", "-"),
             "SDA": ("#e87ba4", "v", "-"), "D-Unet+SDA": ("#008300", "P", "-")}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    for i, case in enumerate(("s0", "s1")):
        for j, key in enumerate("abc"):
            label, cells, (lo, hi), logx = OBS_CELLS[key]
            ax = axes[i, j]
            _style_axes(ax)
            xs = [x for x, _ in cells]
            if key == "c":
                ax.axvline(0.5, color="#f0efec", lw=8, zorder=0)
            else:
                ax.axvspan(lo, hi, color="#f0efec", zorder=0)
            for m, (col, mk, ls) in style.items():
                ax.plot(xs, [ev[case][c][m] for _, c in cells], color=col, marker=mk, ls=ls, lw=1.4, ms=4, label=m)
            if logx:
                ax.set_xscale("log")
            ticks = [6, 10, 30, 100, 300, 1000] if key == "a" else xs
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{x:g}" for x in ticks], fontsize=6.5)
            ax.minorticks_off()
            ax.set_title(f"({key}) {'S0' if case == 's0' else 'S1'}", color=INK, fontsize=8.5, fontweight="bold", loc="left")
            if i == 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel("EV")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=8, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.02))
    _save(fig, out, "fig_obs_system")
    plt.close(fig)


def figures() -> Path:
    out = OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    series = {"L96": json.loads((OUT / "l96_tau_series.json").read_text()),
              "QG": json.loads((QG / "qg_p1_series.json").read_text())}
    tau_figures(series, out)
    missp_figure(series, out)
    budget_figures(out)
    obs_system_figure(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=("series", "extras", "obs", "figures", "all"), nargs="?", default="all")
    args = p.parse_args()
    if args.step in ("series", "all"):
        print(f"wrote {l96_series()}")
    if args.step in ("extras", "all"):
        print(f"wrote {l96_extras()}")
    if args.step in ("obs", "all"):
        print(f"wrote {l96_obs_system()}")
    if args.step in ("figures", "all"):
        print(f"figures in {figures()}")


if __name__ == "__main__":
    main()
