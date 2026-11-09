"""Energy and variogram scores of the P1 L96 ensembles, next to the flow-matching skill score.

Two steps:

    python reports/l96/p1_energy_variogram.py score ROOT OUT_DIR [--label NAME]
        run in the job that wrote the member dumps (they live on node-local storage,
        evaluation/members_store.py): scores every ``*_members.npz`` (DA, ``--save-members``) and
        ``members_<case>.npz`` (learned, ``--members-file full``) under ROOT and writes per-window
        arrays to ``OUT_DIR/<label>_<case>.npz``. The members are never copied.
    python reports/l96/p1_energy_variogram.py report
        pools the per-window arrays of the bundle root ``benchmark_extended/here/p1_energy_variogram``
        into ``reports/l96/outputs/fm_score/p1_energy_variogram.json``.

Units: per-variable z-scores (``probe_fm_score.norm_stats``), the 24 observed variables.
Scores, ensemble of M members X_m and truth y (lower is better):
- energy score ES = mean_m ||X_m - y|| - 1/(2 M^2) sum_{m,m'} ||X_m - X_m'||;
- variogram score of order p = 0.5, unit weights, VS = sum_{i<j} (|y_i - y_j|^p - mean_m |X_mi - X_mj|^p)^2.
Two sample geometries per window of T = 3000 steps:
- snapshots: the 24 variables at every 10th time step (300 per window);
- space-time patches: 24 variables x 10 times spaced by 30 steps (240-dimensional, 10 disjoint
  patches per window, each spanning 300 steps, i.e. 3 observation intervals).
Climatological reference: one fixed set of 30 truth samples (snapshots or patches) drawn uniformly
from the test windows of the same case (seed 0), shared by every window and scheme; the skill
scores are ESS = 1 - ES / ES_clim and VSS = 1 - VS / VS_clim, from window-pooled means.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _inputs  # noqa: E402
from reports.l96.probe_fm_score import norm_stats  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "reports" / "l96" / "outputs" / "fm_score" / "p1_energy_variogram.json"
SNAP_STRIDE = 10
PATCH_TIMES, PATCH_STRIDE = 10, 30
N_CLIM, P = 30, 0.5
KEYS = ("es_snap", "vs_snap", "es_patch", "vs_patch")


def snapshots(a: torch.Tensor) -> torch.Tensor:
    """(T, D, ...) -> (S, D, ...): every SNAP_STRIDE-th step."""
    return a[::SNAP_STRIDE]


def patches(a: torch.Tensor) -> torch.Tensor:
    """(T, D, ...) -> (n, PATCH_TIMES * D, ...): disjoint patches of PATCH_TIMES times PATCH_STRIDE apart."""
    span = PATCH_TIMES * PATCH_STRIDE
    n = a.shape[0] // span
    x = a[: n * span].reshape(n, PATCH_TIMES, PATCH_STRIDE, *a.shape[1:])[:, :, 0]
    return x.reshape(n, PATCH_TIMES * a.shape[1], *a.shape[2:])


def energy(ens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ens (S, D, M), y (S, D) -> (S,)."""
    e = ens.transpose(1, 2)
    skill = torch.linalg.vector_norm(e - y[:, None, :], dim=-1).mean(-1)
    spread = torch.cdist(e, e).mean((-1, -2))
    return skill - 0.5 * spread


def variogram(ens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ens (S, D, M), y (S, D) -> (S,), unit weights, order P."""
    vy = (y[:, :, None] - y[:, None, :]).abs() ** P
    vx = ((ens[:, :, None, :] - ens[:, None, :, :]).abs() ** P).mean(-1)
    return 0.5 * ((vy - vx) ** 2).sum((-1, -2))


def climatology(truth_z: np.ndarray, geometry, rng: np.random.Generator) -> torch.Tensor:
    """N_CLIM truth samples of the given geometry drawn uniformly over windows -> (D, N_CLIM)."""
    pool = [geometry(torch.from_numpy(truth_z[w])) for w in range(truth_z.shape[0])]
    n = pool[0].shape[0]
    idx = rng.choice(truth_z.shape[0] * n, size=N_CLIM, replace=False)
    return torch.stack([pool[i // n][i % n] for i in idx], dim=-1)


def score_file(path: Path, dev: str) -> dict[str, np.ndarray]:
    mu, sd = norm_stats()
    sd24 = sd if sd.shape[-1] == 24 else sd[_inputs_idx()]
    mu24 = mu if mu.shape[-1] == 24 else mu[_inputs_idx()]
    z = np.load(path)
    members, truth = z["members"], z["truth"]
    assert members.shape[2] == 24 and truth.shape == members.shape[:3], (path, members.shape, truth.shape)
    tz = (truth.astype(np.float64) - mu24) / sd24
    rng = np.random.default_rng(0)
    clim = {"snap": climatology(tz, snapshots, rng).to(dev), "patch": climatology(tz, patches, rng).to(dev)}
    W = truth.shape[0]
    out = {k: np.zeros(W) for k in KEYS + tuple(f"{k}_clim" for k in KEYS)}
    out["rmse_phys"], out["spread_phys"] = np.zeros(W), np.zeros(W)
    for w in range(W):
        m = (torch.from_numpy(members[w].astype(np.float64)).to(dev) - torch.from_numpy(mu24)[:, None].to(dev)) \
            / torch.from_numpy(sd24)[:, None].to(dev)
        y = torch.from_numpy(tz[w]).to(dev)
        for g, geo in (("snap", snapshots), ("patch", patches)):
            e, t = geo(m), geo(y)
            c = clim[g][None].expand(t.shape[0], -1, -1)
            out[f"es_{g}"][w] = energy(e, t).mean().item()
            out[f"vs_{g}"][w] = variogram(e, t).mean().item()
            out[f"es_{g}_clim"][w] = energy(c, t).mean().item()
            out[f"vs_{g}_clim"][w] = variogram(c, t).mean().item()
        mean = members[w].astype(np.float64).mean(-1)
        out["rmse_phys"][w] = float(np.sqrt(((mean - truth[w]) ** 2).mean()))
        out["spread_phys"][w] = float(np.sqrt(members[w].astype(np.float64).var(-1, ddof=1).mean()))
    return out


def _inputs_idx() -> list[int]:
    return list(range(8)) + [8 + k * 4 + j for k in range(8) for j in range(2)]


def _name(path: Path, label: str | None) -> str:
    case = re.search(r"_(s[01])(?:_|\.npz|$)", path.name) or re.search(r"(s[01])", path.name)
    assert case, path
    if label:
        return f"{label}_{case.group(1)}"
    da = re.search(r"_(s[01])_(ETKF|EnKF|ETKS)_members\.npz$", path.name)
    assert da, path
    return f"{da.group(2)}_{da.group(1)}"


def score(root: Path, out: Path, label: str | None) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(set(root.rglob("*_members.npz")) | set(root.rglob("members_s[01].npz")))
    assert files, f"no members files under {root}"
    for f in files:
        name = _name(f, label)
        arr = score_file(f, dev)
        np.savez(out / f"{name}.npz", **{k: v.astype(np.float32) for k, v in arr.items()})
        print(f"EV-SCORE {name}: rmse {arr['rmse_phys'].mean():.4f} ES {arr['es_snap'].mean():.4f} "
              f"VS {arr['vs_snap'].mean():.4f} <- {f}", flush=True)


# row -> (score file label, FMSS source, FMSS key, note). Main rows follow the final P1 mapping
# (SDA = SDA2-2D, D-Unet+SDA = DirectUNet-2D -> SDA3-fix-2D); the SDA1-based rows of the first
# scoring run, filed under the generic labels "SDA" / "D-Unet+SDA", are kept as a sensitivity entry.
# Learned FMSS_0.75: the seed-1 value (per_seed[0]), as the scores are on seed 1.
ROWS = {"ETKF": ("ETKF", "series", "ETKF", "DA run"), "EnKF": ("EnKF", "series", "EnKF", "DA run"),
        "ETKS": ("ETKS", "series", "ETKS", "DA run"), "CFM": ("CFM", "learned2d", "CFM", "2D CFM, seed 1"),
        "SDA": ("SDA2", "learned2d", "SDA2", "2D SDA2, seed 1"),
        "D-Unet+SDA": ("D-Unet+SDA3", "learned2d", "D-Unet+SDA", "2D D-Unet -> SDA3-fix, seed 1")}
SENSITIVITY = {"SDA1": ("SDA", "learned2d", "SDA1", "2D SDA1, seed 1"),
               "D-Unet+SDA1": ("D-Unet+SDA", "learned2d", "D-Unet+SDA1", "2D D-Unet -> SDA1, seed 1")}


def _fmss(d: dict, key: str, i75: int, seed_one: bool) -> float:
    r = d[key]
    if seed_one and "per_seed" in r:
        return float(r["per_seed"]["fmss"][0][i75])
    return float(r["fmss"][i75])


def _rows(spec: dict, src: Path, case: str, series: dict, extras: dict, i75: int) -> dict:
    out = {}
    for name, (label, kind, key, note) in spec.items():
        f = src / f"{label}_{case}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        row = {k: float(z[k].mean()) for k in KEYS}
        row.update({f"{k}_skill": float(1 - z[k].mean() / z[f"{k}_clim"].mean()) for k in KEYS})
        row["rmse_phys"] = float(z["rmse_phys"].mean())
        d = series["cases"][case] if kind == "series" else extras["learned2d"][case]
        row["fmss_0.75"] = _fmss(d, key, i75, kind == "learned2d")
        row["fmss_source"] = note
        out[name] = row
    return out


def report() -> Path:
    src = _inputs.root("benchmark_extended", "here") / "p1_energy_variogram"
    series = json.loads((REPO / "reports" / "p1" / "outputs" / "l96_tau_series.json").read_text())
    extras = json.loads((REPO / "reports" / "p1" / "outputs" / "l96_extras.json").read_text())
    i75 = series["taus"].index(0.75)
    res = {c: _rows(ROWS, src, c, series, extras, i75) for c in ("s0", "s1")}
    sens = {c: _rows(SENSITIVITY, src, c, series, extras, i75) for c in ("s0", "s1")}
    OUT_JSON.write_text(json.dumps({"snap_stride": SNAP_STRIDE, "patch": [PATCH_TIMES, PATCH_STRIDE],
                                    "n_clim": N_CLIM, "p": P, "cases": res,
                                    "sensitivity_sda1": sens}, indent=1) + "\n")
    return OUT_JSON


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("root", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--label", default=None)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "score":
        score(args.root, args.out, args.label)
    else:
        print(f"wrote {report()}")


if __name__ == "__main__":
    main()
