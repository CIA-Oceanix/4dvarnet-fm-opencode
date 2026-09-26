"""Versioned QG datasets for the spectral wind drivers: independent splits,
batched GPU generation, manifests, leakage checks and diversity reports.

Implements G2 of `docs/plans/tech/qg_batched_generation_datasets.md`:

* every window owns a `SeedSequence` spawn key ``(split_id, index)`` under its
  split's entropy; test uses its own entropy, so the test set does not depend
  on the train or val sizes and train generation never touches test streams;
* factors come from a Latin-hypercube design per split (independent designs);
* each window has its own forced spin-up, ocean initial state and gyrostat
  burn-in, so no two windows share a trajectory;
* train/val keep the truth every ``keep_every`` steps, test at full
  resolution; streamfunctions and curl fields are recomputed on load;
* a test split is written read-only and refused by non-test loaders.
"""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import stat
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from models.qg_batched import BatchedQGDynamics, SpectralWindForcing
from models.qg_wind_batched import batched_spectral_wind, stream_seed
from models.qg_wind_modes import FourierWindBasis

GENERATOR_VERSION = 1
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
STREAM_OCEAN_IC = 10
DESIGN_KEY = 999_999
FACTORS = ("rd", "U1", "rek", "level", "calm", "cx", "cy", "x0", "y0", "time_unit_days")
REGIME_MODES = {"l63ring4": (0, 3, 6, 9)}


@dataclass(frozen=True)
class QGDatasetSpec:
    name: str = "qg_specwind_gyrostat_v1"
    driver: str = "gyrostat"
    n_train: int = 5000
    n_val: int = 500
    n_test: int = 500
    train_val_entropy: int = 0x5147_5750_2601
    test_entropy: int = 0x5147_5445_2602
    nx: int = 64
    L: float = 1e6
    dt: float = 7200.0
    beta: float = 1.5e-11
    delta: float = 0.25
    U2: float = 0.0
    rd: tuple = (12750.0, 17250.0)
    U1: tuple = (0.0425, 0.0575)
    rek: tuple = (4.919e-7, 6.655e-7)
    r_cf: float = 2.8e-8
    level: tuple = (0.0, 3e-11)
    calm_fraction: float = 0.2
    cx: tuple = (0.25, 0.75)
    cy: tuple = (-0.06, 0.06)
    time_unit_days: tuple = (30.0, 90.0)
    kmax: int = 2
    sigma: float = 2.5e5
    preset: str = "l63ring4"
    tau_days: float = 15.0
    burnin_units: float = 50.0
    spinup_days: float = 730.0
    lead_days: float = 10.0
    window_days: float = 30.0
    keep_every_trainval: int = 6
    keep_every_test: int = 1
    kappa_fb: float = 1.0
    version: int = GENERATOR_VERSION
    notes: dict = field(default_factory=dict)

    @property
    def steps_per_day(self) -> int:
        return round(86400.0 / self.dt)

    @property
    def n_spinup(self) -> int:
        return int(round(self.spinup_days * self.steps_per_day))

    @property
    def n_lead(self) -> int:
        return int(round(self.lead_days * self.steps_per_day))

    @property
    def n_window(self) -> int:
        return int(round(self.window_days * self.steps_per_day))

    @property
    def n_total(self) -> int:
        return self.n_spinup + self.n_lead + self.n_window

    def n_windows(self, split: str) -> int:
        return {"train": self.n_train, "val": self.n_val, "test": self.n_test}[split]

    def entropy(self, split: str) -> int:
        return self.test_entropy if split == "test" else self.train_val_entropy

    def keep_every(self, split: str) -> int:
        k = self.keep_every_test if split == "test" else self.keep_every_trainval
        if self.n_lead % k or self.n_window % k:
            raise ValueError(f"lead ({self.n_lead}) and window ({self.n_window}) steps must be "
                             f"multiples of keep_every={k}")
        return k

    def split_hash(self, split: str) -> str:
        d = asdict(self)
        if self.driver != "coupled":
            d.pop("kappa_fb")
        for other in ("n_train", "n_val", "n_test"):
            if other != f"n_{split}":
                d.pop(other)
        if split == "test":
            d.pop("train_val_entropy")
        else:
            d.pop("test_entropy")
        d["split"] = split
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def to_json(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=list))

    @staticmethod
    def from_json(d: dict) -> "QGDatasetSpec":
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
        return QGDatasetSpec(**kw)


SPECS = {
    "qg_specwind_gyrostat_v1": QGDatasetSpec(),
    "qg_specwind_gyrostat_demo": QGDatasetSpec(name="qg_specwind_gyrostat_demo",
                                               n_train=256, n_val=64, n_test=64),
    "qg_coupled_gyrostat_v1": QGDatasetSpec(
        name="qg_coupled_gyrostat_v1", driver="coupled", kappa_fb=1.0,
        notes={"paired_with": "qg_specwind_gyrostat_v1",
               "pairing": "same entropies, factor design and seeds: window i differs from the "
                          "forced dataset's window i only by the two-way coupling"}),
    "qg_coupled_gyrostat_demo": QGDatasetSpec(
        name="qg_coupled_gyrostat_demo", driver="coupled", kappa_fb=1.0,
        n_train=256, n_val=64, n_test=64),
}


def window_seed(spec: QGDatasetSpec, split: str, index: int) -> int:
    ss = np.random.SeedSequence(spec.entropy(split), spawn_key=(SPLIT_IDS[split], int(index)))
    a, b = ss.generate_state(2)
    return int(a) << 32 | int(b)


def design_unit(spec: QGDatasetSpec, split: str) -> np.ndarray:
    from scipy.stats import qmc

    seed = np.random.SeedSequence(spec.entropy(split), spawn_key=(SPLIT_IDS[split], DESIGN_KEY))
    lhs = qmc.LatinHypercube(d=len(FACTORS), seed=np.random.default_rng(seed))
    return lhs.random(spec.n_windows(split))


def design_factors(spec: QGDatasetSpec, split: str) -> dict[str, np.ndarray]:
    u = design_unit(spec, split)
    col = {name: u[:, i] for i, name in enumerate(FACTORS)}

    def span(rng, v):
        return rng[0] + v * (rng[1] - rng[0])

    level = span(spec.level, col["level"])
    level = np.where(col["calm"] < spec.calm_fraction, 0.0, level)
    return {
        "rd": span(spec.rd, col["rd"]),
        "U1": span(spec.U1, col["U1"]),
        "rek": span(spec.rek, col["rek"]),
        "level": level,
        "cx": span(spec.cx, col["cx"]),
        "cy": span(spec.cy, col["cy"]),
        "x0": col["x0"] * spec.L,
        "y0": col["y0"] * spec.L,
        "time_unit_days": span(spec.time_unit_days, col["time_unit_days"]),
        "unit": u,
    }


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def generate_windows(spec: QGDatasetSpec, split: str, indices: list[int],
                     device: torch.device | str = "cpu", batch_size: int = 256,
                     dtype: torch.dtype = torch.float32) -> dict:
    device = torch.device(device)
    fac = design_factors(spec, split)
    keep = spec.keep_every(split)
    basis64 = FourierWindBasis(nx=spec.nx, L=spec.L, kmax=spec.kmax, device=device)
    basis_f = FourierWindBasis(nx=spec.nx, L=spec.L, kmax=spec.kmax, dtype=dtype, device=device)
    t0_frame = spec.n_lead // keep
    regime_idx = REGIME_MODES.get(spec.preset) if spec.driver.startswith("gyrostat") else None
    out = {k: [] for k in ("true_state", "wind_amplitudes", "window_seed", "regime", "rms_curl",
                           "ke_upper", "state_hash", "gyro_hash")}
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        seeds = [window_seed(spec, split, i) for i in idx]
        pick = lambda name: fac[name][idx]  # noqa: E731
        amps, modes = batched_spectral_wind(
            basis64, spec.driver, seeds, spec.n_total, spec.dt, amp=pick("level"),
            cx=pick("cx"), cy=pick("cy"), x0=pick("x0"), y0=pick("y0"), sigma=spec.sigma,
            tau_days=spec.tau_days, preset=spec.preset, time_unit_days=pick("time_unit_days"),
            burnin_units=spec.burnin_units, device=device, return_modes=True)
        model = BatchedQGDynamics(len(idx), nx=spec.nx, L=spec.L, dt=spec.dt, beta=spec.beta,
                                  rd=pick("rd"), delta=spec.delta, U1=pick("U1"), U2=spec.U2,
                                  rek=pick("rek"), r_cf=spec.r_cf, dtype=dtype, device=device)
        q0 = model.initial_q([stream_seed(s, STREAM_OCEAN_IC) for s in seeds])
        traj, _ = model.rollout(q0, spec.n_total, forcing=SpectralWindForcing(basis_f, amps.to(dtype)),
                                keep_from=spec.n_spinup, keep_every=keep)
        win_amps = amps[:, spec.n_spinup:].to(torch.float32)
        psi = model.streamfunctions(traj[:, t0_frame:])
        psih = torch.fft.rfft2(psi[:, :, 0], dim=(-2, -1))
        k2 = model.K2.to(psih.real.dtype)
        ke = 0.5 * (k2 * psih.abs() ** 2).sum(dim=(-2, -1)) / float(spec.nx * spec.nx) ** 2
        field_ms = 0.5 * (win_amps[:, spec.n_lead:] ** 2).sum(-1)
        out["true_state"].append(traj.to(torch.float32).cpu())
        out["wind_amplitudes"].append(win_amps.cpu())
        out["window_seed"].extend(seeds)
        out["rms_curl"].append(field_ms.mean(1).sqrt().cpu().double())
        out["ke_upper"].append(ke.mean(1).cpu().double())
        if regime_idx is not None:
            signs = (modes[:, spec.n_spinup + spec.n_lead, list(regime_idx)] > 0).to(torch.int64)
            weights = 2 ** torch.arange(len(regime_idx), device=signs.device)
            out["regime"].append((signs * weights).sum(-1).cpu())
        else:
            out["regime"].append(torch.full((len(idx),), -1, dtype=torch.int64))
        for b in range(len(idx)):
            out["state_hash"].append(_hash_tensor(traj[b, t0_frame]))
            out["gyro_hash"].append(_hash_tensor(modes[b, 0]))
    result = {
        "indices": list(indices),
        "true_state": torch.cat(out["true_state"]),
        "wind_amplitudes": torch.cat(out["wind_amplitudes"]),
        "window_seed": out["window_seed"],
        "regime": torch.cat(out["regime"]),
        "rms_curl": torch.cat(out["rms_curl"]),
        "ke_upper": torch.cat(out["ke_upper"]),
        "state_hash": out["state_hash"],
        "gyro_hash": out["gyro_hash"],
        "factors": {k: np.asarray(v[indices]) for k, v in fac.items() if k != "unit"},
        "unit": np.asarray(fac["unit"][indices]),
        "keep_every": keep,
        "window_start_frame": t0_frame,
    }
    return result


def shard_indices(n: int, shard: int, n_shards: int) -> list[int]:
    bounds = np.linspace(0, n, n_shards + 1).round().astype(int)
    return list(range(bounds[shard], bounds[shard + 1]))


def split_dir(root: str, spec: QGDatasetSpec, split: str) -> str:
    return os.path.join(root, spec.name, split)


def write_shard(spec: QGDatasetSpec, split: str, root: str, shard: int, n_shards: int,
                device: torch.device | str = "cpu", batch_size: int = 256) -> str:
    d = split_dir(root, spec, split)
    os.makedirs(d, exist_ok=True)
    idx = shard_indices(spec.n_windows(split), shard, n_shards)
    t0 = time.time()
    if spec.driver == "coupled":
        from models.qg_coupled import generate_coupled_windows

        data = generate_coupled_windows(spec, split, idx, device=device, batch_size=batch_size)
    else:
        data = generate_windows(spec, split, idx, device=device, batch_size=batch_size)
    data["meta"] = {"shard": shard, "n_shards": n_shards, "device": str(device),
                    "batch_size": batch_size, "seconds": time.time() - t0,
                    "torch": torch.__version__}
    path = os.path.join(d, f"shard_{shard:04d}_of_{n_shards:04d}.pt")
    torch.save(data, path)
    return path


def assemble_split(spec: QGDatasetSpec, split: str, root: str) -> dict:
    d = split_dir(root, spec, split)
    shards = sorted(glob.glob(os.path.join(d, "shard_*.pt")))
    if not shards:
        raise FileNotFoundError(f"no shards in {d}")
    records, seconds, meta = [], 0.0, []
    for path in shards:
        s = torch.load(path, map_location="cpu", weights_only=False)
        seconds += s["meta"]["seconds"]
        meta.append({**s["meta"], "file": os.path.basename(path), "n": len(s["indices"])})
        for j, i in enumerate(s["indices"]):
            records.append({
                "index": int(i), "file": os.path.basename(path), "row": j,
                "window_seed": str(s["window_seed"][j]),
                "factors": {k: float(v[j]) for k, v in s["factors"].items()},
                "unit": [float(x) for x in s["unit"][j]],
                "regime": int(s["regime"][j]), "rms_curl": float(s["rms_curl"][j]),
                "ke_upper": float(s["ke_upper"][j]),
                "state_hash": s["state_hash"][j], "gyro_hash": s["gyro_hash"][j],
            })
            if s.get("feedback_ratio") is not None:
                records[-1]["feedback_ratio"] = float(s["feedback_ratio"][j])
    records.sort(key=lambda r: r["index"])
    if [r["index"] for r in records] != list(range(spec.n_windows(split))):
        raise ValueError(f"{split}: shards do not cover indices 0..{spec.n_windows(split) - 1}")
    manifest = {
        "split": split, "split_id": SPLIT_IDS[split], "spec": spec.to_json(),
        "split_hash": spec.split_hash(split), "generator_version": GENERATOR_VERSION,
        "entropy": str(spec.entropy(split)), "n": len(records),
        "keep_every": spec.keep_every(split), "window_start_frame": spec.n_lead // spec.keep_every(split),
        "generation_seconds": seconds, "shards": meta, "windows": records,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = os.path.join(d, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh)
    if split == "test":
        for p in shards + [path]:
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP)
    return manifest


class QGGeneratedSplit:
    def __init__(self, directory: str, purpose: str):
        if purpose not in ("train", "eval", "test"):
            raise ValueError("purpose must be 'train', 'eval' or 'test'")
        with open(os.path.join(directory, "manifest.json")) as fh:
            self.manifest = json.load(fh)
        split = self.manifest["split"]
        if split == "test" and purpose != "test":
            raise PermissionError("a test split can only be opened with purpose='test'")
        if purpose == "test" and split != "test":
            raise PermissionError(f"purpose='test' given for the {split!r} split")
        self.directory = directory
        self.spec = QGDatasetSpec.from_json(self.manifest["spec"])
        self.windows = self.manifest["windows"]
        self._cache: tuple[str, dict] | None = None
        self.basis = FourierWindBasis(nx=self.spec.nx, L=self.spec.L, kmax=self.spec.kmax)

    def __len__(self) -> int:
        return len(self.windows)

    def _shard(self, name: str) -> dict:
        if self._cache is None or self._cache[0] != name:
            self._cache = (name, torch.load(os.path.join(self.directory, name),
                                            map_location="cpu", weights_only=False))
        return self._cache[1]

    def __getitem__(self, i: int) -> dict:
        rec = self.windows[i]
        s = self._shard(rec["file"])
        return {"true_state": s["true_state"][rec["row"]],
                "wind_amplitudes": s["wind_amplitudes"][rec["row"]],
                "factors": rec["factors"], "regime": rec["regime"], "index": rec["index"]}

    def model_for(self, i: int) -> BatchedQGDynamics:
        f = self.windows[i]["factors"]
        sp = self.spec
        return BatchedQGDynamics(1, nx=sp.nx, L=sp.L, dt=sp.dt, beta=sp.beta, rd=f["rd"],
                                 delta=sp.delta, U1=f["U1"], U2=sp.U2, rek=f["rek"],
                                 r_cf=sp.r_cf, dtype=torch.float32)

    def streamfunction(self, i: int) -> torch.Tensor:
        return self.model_for(i).streamfunctions(self[i]["true_state"].unsqueeze(0))[0]

    def curl_field(self, i: int) -> torch.Tensor:
        return self.basis.curl_field(self[i]["wind_amplitudes"].double())


def _unit_factors(split: QGGeneratedSplit) -> np.ndarray:
    return np.array([w["unit"] for w in split.windows])


def _window_start_q1(split: QGGeneratedSplit, max_n: int | None = None) -> np.ndarray:
    f0 = split.manifest["window_start_frame"]
    n = len(split) if max_n is None else min(max_n, len(split))
    by_file: dict[str, list[tuple[int, int]]] = {}
    for i in range(n):
        rec = split.windows[i]
        by_file.setdefault(rec["file"], []).append((i, rec["row"]))
    rows = [None] * n
    for name, items in by_file.items():
        states = split._shard(name)["true_state"]
        for i, r in items:
            rows[i] = states[r, f0, 0].reshape(-1).numpy()
    x = np.asarray(rows, dtype=np.float64)
    x -= x.mean(1, keepdims=True)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def check_independence(reference: QGGeneratedSplit, other: QGGeneratedSplit,
                       max_state_windows: int | None = 2000) -> dict:
    ref_seeds = {w["window_seed"] for w in reference.windows}
    oth_seeds = {w["window_seed"] for w in other.windows}
    ref_hash = {w["state_hash"] for w in reference.windows} | {w["gyro_hash"] for w in reference.windows}
    oth_hash = {w["state_hash"] for w in other.windows} | {w["gyro_hash"] for w in other.windows}
    from scipy.spatial import cKDTree

    a, b = _unit_factors(reference), _unit_factors(other)
    tree = cKDTree(a)
    nn_ref = tree.query(a, k=2)[0][:, 1]
    nn_oth = tree.query(b, k=1)[0]
    q01 = float(np.quantile(nn_ref, 0.01))
    frac_close = float((nn_oth < q01).mean())
    qa = _window_start_q1(reference, max_state_windows)
    qb = _window_start_q1(other, max_state_windows)
    max_corr = float(np.abs(qb @ qa.T).max())
    checks = {
        "seeds_disjoint": not (ref_seeds & oth_seeds),
        "state_hashes_disjoint": not (ref_hash & oth_hash),
        "no_duplicate_factors": bool(nn_oth.min() > 1e-12),
        "factor_nn_not_closer_than_within_reference": frac_close <= 0.05,
        "window_start_q1_max_abs_corr_below_0.99": max_corr < 0.99,
    }
    return {"reference": reference.manifest["split"], "other": other.manifest["split"],
            "checks": checks, "passed": all(checks.values()),
            "stats": {"factor_nn_other_to_ref_median": float(np.median(nn_oth)),
                      "factor_nn_within_ref_median": float(np.median(nn_ref)),
                      "fraction_below_ref_q01": frac_close,
                      "window_start_q1_max_abs_corr": max_corr}}


def diversity_report(splits: dict[str, QGGeneratedSplit]) -> dict:
    from scipy.stats import ks_2samp, qmc

    rep = {}
    train = splits.get("train")
    for name, s in splits.items():
        w = s.windows
        regimes = [x["regime"] for x in w]
        stats = {
            "n": len(w),
            "unit_design_discrepancy": float(qmc.discrepancy(_unit_factors(s))),
            "calm_fraction": float(np.mean([x["factors"]["level"] == 0.0 for x in w])),
            "regime_counts": {str(r): regimes.count(r) for r in sorted(set(regimes))},
            "rms_curl": [x["rms_curl"] for x in w],
            "ke_upper": [x["ke_upper"] for x in w],
            "factors": {k: [x["factors"][k] for x in w] for k in w[0]["factors"]},
        }
        if train is not None and name != "train":
            tw = train.windows
            ks = {"rms_curl": ks_2samp([x["rms_curl"] for x in w], [x["rms_curl"] for x in tw]).pvalue,
                  "ke_upper": ks_2samp([x["ke_upper"] for x in w], [x["ke_upper"] for x in tw]).pvalue}
            for k in w[0]["factors"]:
                ks[k] = ks_2samp([x["factors"][k] for x in w], [x["factors"][k] for x in tw]).pvalue
            stats["ks_pvalue_vs_train"] = {k: float(v) for k, v in ks.items()}
        rep[name] = stats
    return rep


def spec_with(spec: QGDatasetSpec, **changes) -> QGDatasetSpec:
    return dataclasses.replace(spec, **changes)


def estimate_storage_gb(spec: QGDatasetSpec) -> dict:
    per_frame = 2 * spec.nx * spec.nx * 4
    n_amp = FourierWindBasis(nx=spec.nx, L=spec.L, kmax=spec.kmax).n_amp
    out = {}
    for split in ("train", "val", "test"):
        frames = (spec.n_lead + spec.n_window) // spec.keep_every(split) + 1
        out[split] = spec.n_windows(split) * (frames * per_frame
                                               + (spec.n_lead + spec.n_window) * n_amp * 4) / 1e9
    out["total"] = sum(out.values())
    return {k: round(v, 2) for k, v in out.items()}

