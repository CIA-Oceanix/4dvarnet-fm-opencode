"""Per-window RMSE / EV / CRPS summary of the current L96 benchmark rows.

One section, shared by ``l96_benchmark_extended.md`` and ``p1_l96_benchmark.md``. It reads the ``benchmark_extended`` input bundle (the
current benchmark: DA at the #295 inflation, the ETKS of #299, 1200-epoch
DirectUNet / CFM / SDA (gw 25), the best hybrid) whatever report embeds it, so the
two copies are identical.

Per window, on the 24 observed channels of the 200 P1 test windows:
- RMSE: per-channel RMSE over time, averaged over channels (the report convention);
- EV: per-channel 1 - SSE/SST over time (SST about the window's own temporal mean),
  averaged over channels;
- CRPS: the stored per-channel CRPS (analysis ensemble for DA, members for flows /
  SDA), averaged over channels.
Seeds are averaged per window; cells are mean +- sd over the 200 windows.
"""
from pathlib import Path

import numpy as np
import torch

import _inputs
from evaluation.estimate_metrics import _groups_from_per_window as G
from evaluation.run_l96 import make_obs_j_indices

REPORT = "benchmark_extended"
HERE = _inputs.root(REPORT, "here")
REGE = HERE / "eval_regular"
CAN = HERE / "eval_rlayout_n10-100_k4-16_w200"
RANDOM_OBS_TIMES = _inputs.root(REPORT, "random_obs_times")
DA243 = _inputs.root(REPORT, "da_random_layout") / "l96_da_random_layout"
DANEW = HERE / "da_inflation_2026-09-28"
NEWINF = "_infs0-1.2_s1-3.0_etkf_infs0-1.15_s1-2.5"
ETKS = "etks-correct-Lfull-infs0-1.15_s1-2.5"
REG_TEST = _inputs.shared() / "l96_datasets_obsj2_int100_nwin200.pt"
CASES = ("s0", "s1")
IDX = np.array(make_obs_j_indices(8, 4, 2))
SEEDS = (1, 2, 3)
METRICS = (("rmse", "RMSE", "lower"), ("ev", "EV", "higher"), ("crps", "CRPS", "lower"))


def _sel(a: np.ndarray) -> np.ndarray:
    return a[..., IDX] if a.shape[-1] > len(IDX) else a


def _rmse(traj: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return G(np.sqrt(((traj - truth) ** 2).mean(1)))["all_obs"]


def _ev(traj: np.ndarray, truth: np.ndarray) -> np.ndarray:
    sse = ((traj - truth) ** 2).sum(1)
    sst = ((truth - truth.mean(1, keepdims=True)) ** 2).sum(1)
    return (1 - sse / sst).mean(1)


def _learned(dirs: list[Path], truth: dict) -> dict | None:
    dirs = [d for d in dirs if all((d / f"estimates_{c}.npz").exists() for c in CASES)]
    if not dirs:
        return None
    out = {}
    for c in CASES:
        per = {"rmse": [], "ev": [], "crps": []}
        for d in dirs:
            z = np.load(d / f"estimates_{c}.npz")
            t, tr = z["trajectories"].astype(np.float64), z["truth"].astype(np.float64)
            if not np.allclose(tr, truth[c], atol=1e-4):
                raise ValueError(f"{d}: truth differs from the P1 test set")
            per["rmse"].append(_rmse(t, tr))
            per["ev"].append(_ev(t, tr))
            s = d / f"scores_{c}.npz"
            per["crps"].append(G(np.load(s)["crps"])["all_obs"] if s.exists() else None)
        out[c] = {k: None if any(v is None for v in vs) else np.mean(vs, axis=0) for k, vs in per.items()}
    out["n"] = len(dirs)
    return out


def _da_regular(path: Path, method: str, truth: dict) -> dict:
    z = np.load(path)
    out = {"n": None}
    for c in CASES:
        k = f"{c}_{method.replace('-', '_')}"
        t = _sel(z[f"{k}_trajectories"]).astype(np.float64)
        out[c] = {"rmse": _rmse(t, truth[c]), "ev": _ev(t, truth[c]),
                  "crps": G(_sel(z[f"{k}_crps"]))["all_obs"] if f"{k}_crps" in z.files else None}
    return out


def _da_random(traj: Path | None, per_window: Path, method: str, truth: dict) -> dict:
    z = np.load(traj) if traj is not None else None
    p = np.load(per_window)
    out = {"n": None}
    for c in CASES:
        k = f"{c}_{method.replace('-', '_')}"
        out[c] = {"rmse": p[f"{k}_rmse_all_obs"],
                  "ev": _ev(_sel(z[f"{k}_trajectories"]).astype(np.float64), truth[c]) if z is not None else None,
                  "crps": p[f"{k}_crps_all_obs"] if f"{k}_crps_all_obs" in p.files else None}
    return out


def rows() -> list[tuple[str, str, dict | None, dict | None]]:
    """(family, scheme, regular, random) for the current benchmark rows."""
    d = torch.load(REG_TEST, weights_only=False)
    truth = {c: np.stack([w["true_state"].numpy()[:, IDX] for w in d[f"test_{c}"]]).astype(np.float64) for c in CASES}
    reg_da = DANEW / f"l96_baselines_trajectories_dws500_s0c_crps{NEWINF}_obsj2_int100_fw_dafw.npz"
    can_da = DANEW / f"l96_baselines_trajectories_dws500_rlayout_n10-100_k4-16_w200_d1{NEWINF}_obsj2_fw_dafw.npz"
    can_da_pw = DANEW / "l96_da_random_layout" / f"per_window_rlayout_n10-100_k4-16_w200_d1{NEWINF}.npz"
    reg_etks = HERE / f"l96_baselines_trajectories_dws500_s0c_test_{ETKS}_infs0-1.2_s1-3.0_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz"
    can_etks = HERE / f"l96_baselines_trajectories_dws500_rlayout_n10-100_k4-16_w200_d1_{ETKS}_infs0-1.15_s1-2.5_etkf_infs0-1.15_s1-2.5_obsj2_fw_dafw.npz"
    can_etks_pw = HERE / "l96_da_random_layout" / f"per_window_rlayout_n10-100_k4-16_w200_d1_{ETKS}_infs0-1.15_s1-2.5.npz"
    reg_strong = RANDOM_OBS_TIMES / "l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw_dafw.npz"
    can_strong_pw = DA243 / "per_window_rlayout_n10-100_k4-16_w200_d1_inf2.0.npz"
    out = [("DA", "ETKF (inflation 1.15 / 2.5)", _da_regular(reg_da, "ETKF", truth), _da_random(can_da, can_da_pw, "ETKF", truth)),
           ("DA", "EnKF (inflation 1.2 / 3.0)", _da_regular(reg_da, "EnKF", truth), _da_random(can_da, can_da_pw, "EnKF", truth)),
           ("DA", "ETKS (inflation 1.15 / 2.5)", _da_regular(reg_etks, "ETKS", truth), _da_random(can_etks, can_etks_pw, "ETKS", truth)),
           ("DA", "Strong-4DVar", _da_regular(reg_strong, "Strong-4DVar", truth), _da_random(None, can_strong_pw, "Strong-4DVar", truth))]
    for lab, name in (("SDA1-M", "B4_sda1_monaiM_ep1200_l96"), ("SDA2-M", "A3_sda2_monaiM_ep1200_l96"),
                      ("SDA3-fix-M", "A3_sda3fix_monaiM_ep1200_l96")):
        out.append(("SDA (gw 25, 1200 ep)", lab, _learned([REGE / f"{name}_seed{s}" / "ens30_gw25" for s in SEEDS], truth),
                    _learned([CAN / f"{name}_seed{s}" / "ens30_gw25" for s in SEEDS], truth)))
    for fam, lab, name, sub in (("DirectUNet (1200 ep)", "DirectUNet-M", "directunet", "ens1_no1"),
                                ("CFM (1200 ep)", "PredictStateCFM-M", "predictstatecfm", "ens30_no20"),
                                ("CFM (1200 ep)", "VanillaCFM-M", "vanillacfm", "ens30_no20")):
        run = f"L96B_{name}_monaiM_ep1200_seed{{s}}"
        out.append((fam, lab, _learned([HERE / run.format(s=s) / sub for s in SEEDS], truth),
                    _learned([CAN / run.format(s=s) / sub for s in SEEDS], truth)))
    hyb = "hybrid_DU1200s{s}_A3_sda3fix_monaiM_ep1200_l96"
    out.append(("*reference*", "*Hybrid DirectUNet-M(1200 ep) -> SDA3-fix-M(1200 ep)*",
                _learned([REGE / hyb.format(s=s) / "tau0.1_gw2" for s in SEEDS], truth),
                _learned([CAN / hyb.format(s=s) / "tau0.1_gw2" for s in SEEDS], truth)))
    return out


def _ranks(values: list[float | None], higher: bool) -> dict[int, str]:
    ok = [(v, i) for i, v in enumerate(values) if v is not None]
    ok.sort(reverse=higher)
    return {i: mark for (_, i), mark in zip(ok[:2], ("**", "*"))}


def section(heading: str = "##") -> list[str]:
    """Markdown lines of the per-window RMSE / EV / CRPS section."""
    R = rows()
    A = [f"{heading} Per-window RMSE / EV / CRPS summary (current benchmark)\n",
         "Common to `l96_benchmark_extended.md` and `p1_l96_benchmark.md`, and generated from "
         "the `benchmark_extended` inputs by `reports/l96/per_window_summary.py`. The DA rows use the #295 inflation "
         "(#298), the ETKS is the #299 row, and DirectUNet / CFM / the SDA priors are at 1200 epochs (SDA retrained in #307; "
         "the hybrid uses the 1200-epoch SDA3-fix prior, seed 1). The hybrid is shown for reference.\n",
         "Per window, on the 24 observed channels of the 200 P1 test windows: **RMSE** is the per-channel RMSE over time; "
         "**EV** is per-channel 1 - SSE/SST over time, with SST about the window's own mean, so it is lower than the pooled "
         "EV of the benchmark JSONs; **CRPS** is taken on the analysis ensemble (DA) or the members (flows, SDA). Each is "
         "averaged over channels. Seeds are averaged per window; cells are mean ± sd over the 200 windows. Per column, "
         "the best value is in **bold** and the second best in *italics*. Deterministic schemes have no CRPS. The "
         "random-layout Strong-4DVar trajectories were not kept, so that cell has no EV.\n"]
    for key, name, better in METRICS:
        A += [f"**{name}** ({better} is better)\n",
              "| family | scheme | seeds | regular S0 | regular S1 | random S0 | random S1 |", "|---|---|---|---|---|---|---|"]
        cols = [(t, c) for t in (2, 3) for c in CASES]
        marks = []
        for t, c in cols:
            vals = [None if r[t] is None or r[t][c][key] is None else float(r[t][c][key].mean()) for r in R]
            marks.append(_ranks(vals, better == "higher"))
        for i, r in enumerate(R):
            cells = []
            for j, (t, c) in enumerate(cols):
                v = None if r[t] is None else r[t][c][key]
                if v is None:
                    cells.append("—")
                    continue
                m = marks[j].get(i, "")
                cells.append(f"{m}{v.mean():.3f}{m} ± {v.std(ddof=1):.3f}")
            n = r[2]["n"] if r[2] is not None and r[2]["n"] else "—"
            A.append(f"| {r[0]} | {r[1]} | {n} | " + " | ".join(cells) + " |")
        A.append("")
    return A
