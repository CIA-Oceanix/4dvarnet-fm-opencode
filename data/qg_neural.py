"""QG neural supervision datasets (S0, daily resolution, streamfunction target).

Mirrors the L96 neural design on the QG case study. A window is a 30-day S0
truth window (`num_steps` native steps at `dt`; 12 steps/day). Two supervision
targets are produced as daily means (12-step means):

* ``psi``  -- the full 2-layer **streamfunction** (channels [0:ny*nx] = upper
  layer psi1, [ny*nx:2*ny*nx] = lower layer psi2, layer major). This is the
  **primary state/target** the DirectUNet / VanillaCFM estimate.
* ``q``    -- the full 2-layer PV (`true_state` channels, layer major), used as
  an auxiliary target so the training loss can also match PV.

Because PV<->psi is a linear spectral inversion, ``psi = q_to_psi(q)`` holds
per native step, so ``daily-mean(psi) = q_to_psi(daily-mean(q))``; we invert the
daily-binned PV once per window (30 inversions/window). The inversion operator
(and thus the psi<->q map) depends on the per-window resolved ``rd``, so each
window carries its own reconstructed inverter (`QGPsiDynamics`).

Observations are the upper-layer psi obs of the S0 window, grid-expanded then
aggregated to one value per day and NaN-masked where unobserved, padded out to
the full state width (zeros in the lower-layer channels) so the shared
``DirectUNet``/``VanillaCFM`` flow models (which assume ``obs_dim==state_dim``)
need no code change.

**Normalization (2026-09-08):** psi (streamfunction) is z-score normalized
per layer with a single *global* (mean, std) pair computed once over the
whole training split (see `precompute_qg_norm_stats.py` ->
`experiments/qg_psi_norm_stats.pt`, loaded via `data.normalization.
load_norm_stats`/`normalize`/`denormalize`) and applied identically to
train/val/test so eval maps back to physical units. Obs (upper-layer psi)
uses the same psi1 stats. PV (q) is **left in raw physical units** -- it is
only ever an auxiliary loss term (see `train_qg_neural.py`'s
`q_loss_weight`, derived as `1/Var(q)` by the precompute script so its raw-
unit MSE contributes comparably to the (unit-variance) normalized psi loss).

Per-window normalization (`WindowScale`/`window_scales`) was the original
design (each window divided by its own std, so windows spanning a wide
streamfunction-energy range -- ~24-83x at nx=64 across the 1000-window train
split, see `precompute_qg_norm_stats.py`'s diagnostics -- are each mapped to
O(1)); global normalization trades that per-window equal-weighting for a
single interpretable physical-unit scale, at the cost of under-weighting
low-energy windows in the loss. `window_scales` is retained for this
diagnostic and is no longer used to build training targets.

**On-the-fly obs (train/val diversity):** ``QGNeuralDataset(on_the_fly_obs=True)``
redraws the (noisy) obs + init-state from a *truth-only* window
(``data.qg.ensure_truth_only_cache``, no obs baked in) at every ``__getitem__``
call via ``QGS01Dataset._generate_obs_ic`` with a fresh random seed -- so the
same cached truth trajectory yields a different obs realization each epoch
instead of one fixed draw, increasing training diversity without re-paying the
truth rollout cost. The state/PV targets (``psi_daily``/``q_daily``) come from
``true_state`` and are unaffected. The ``test`` split keeps the original fixed,
reproducible obs (``on_the_fly_obs=False``, the default) for stable evaluation.
"""

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from data.normalization import denormalize, normalize
from data.qg import QGConfig, QGS01Dataset, expand_obs_to_grid

_INVERTER_CACHE: dict = {}


def steps_per_day(cfg: QGConfig) -> int:
    return round(86400.0 / cfg.dt)


def num_days(cfg: QGConfig) -> int:
    spd = steps_per_day(cfg)
    return cfg.num_steps // spd


def layer_split(cfg: QGConfig) -> int:
    return cfg.ny * cfg.nx


def _daily_mean_bin(x: torch.Tensor, spd: int) -> torch.Tensor:
    *lead, T, D = x.shape
    if T % spd != 0:
        raise ValueError(f"T={T} not divisible by steps_per_day={spd}")
    x = x.reshape(*lead, T // spd, spd, D)
    return x.mean(dim=-2)


def _daily_obs_psi(window: dict, cfg: QGConfig) -> tuple[torch.Tensor, torch.Tensor]:
    grid = expand_obs_to_grid(window, cfg)  # (T, ny*nx) NaN-padded upper psi
    spd = steps_per_day(cfg)
    T, N = grid.shape
    days = T // spd
    obs = torch.full((days, N), float("nan"))
    mask = torch.zeros(days, N, dtype=torch.bool)
    for d in range(days):
        day = grid[d * spd:(d + 1) * spd]
        obs_present = ~torch.isnan(day)
        count = obs_present.sum(dim=0)
        mask[d] = count > 0
        day_obs = torch.nan_to_num(day, nan=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            obs[d] = day_obs.sum(dim=0) / count.clamp(min=1).float()
    obs[~mask] = float("nan")
    return obs, mask


def _reconstruct_inverter(cfg: QGConfig, rd: float, device: torch.device | None = None):
    """Per-(cfg, rd, device) cached QGPsiDynamics inverter for psi<->q.

    A separate instance is cached per target device so a GPU q-loss never moves
    the CPU inverter (used by `psi_daily`/`window_scales`) — avoiding device
    pollution of the shared cache.
    """
    dev_key = str(device) if device is not None else "cpu"
    key = (cfg.nx, cfg.L, rd, cfg.delta, dev_key)
    if key in _INVERTER_CACHE:
        return _INVERTER_CACHE[key]
    from models.qg_dynamics import QGDynamics
    from models.qg_psi_dynamics import QGPsiDynamics
    dyn = QGDynamics(
        nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=cfg.beta, rd=rd, delta=cfg.delta,
        U1=cfg.U1, U2=cfg.U2, rek=cfg.rek, filterfac=cfg.filterfac,
        wind_amp=cfg.wind_amp, wind_tau_days=cfg.wind_tau_days,
        wind_sigma=cfg.wind_sigma, wind_cx=cfg.wind_cx, wind_cy=cfg.wind_cy,
        wind_drift_tau_days=cfg.wind_drift_tau_days,
        wind_drift_sigma=cfg.wind_drift_sigma, wind_seed=cfg.wind_seed,
    )
    inv = QGPsiDynamics(dyn)
    if device is not None:
        inv = inv.to(device)
    _INVERTER_CACHE[key] = inv
    return inv


def q_daily(window: dict, cfg: QGConfig) -> torch.Tensor:
    return _daily_mean_bin(window["true_state"], steps_per_day(cfg))


def psi_daily(window: dict, cfg: QGConfig) -> torch.Tensor:
    inv = _reconstruct_inverter(cfg, float(window["true_params"]["rd"]))
    qd = q_daily(window, cfg)
    psi = inv.inner.streamfunctions(qd)  # (days, 2, ny, nx)
    if psi.dim() == 4:
        psi = psi.reshape(psi.shape[0], -1)
    return psi


def psi_to_q(state: torch.Tensor, rd: float, cfg: QGConfig,
             device: torch.device | None = None) -> torch.Tensor:
    """Map a flattened (..., 2*ny*nx) physical psi state to its PV, layered."""
    inv = _reconstruct_inverter(cfg, rd, device=device)
    return inv.psi_to_q(state)


@dataclass
class WindowScale:
    """Per-window per-layer std (streamfunction psi + PV q).

    No longer used to build training targets (see the module docstring's
    2026-09-08 normalization note) -- retained as a diagnostic (e.g.
    `precompute_qg_norm_stats.py`'s window-energy-range check) and for tests
    that pin its contract.
    """

    psi1: float = 1.0
    psi2: float = 1.0
    q1: float = 1.0
    q2: float = 1.0

    @staticmethod
    def collate(scales: list) -> torch.Tensor:
        return torch.tensor(
            [[s.psi1, s.psi2, s.q1, s.q2] for s in scales], dtype=torch.float32
        )


def window_scales(w: dict, cfg: QGConfig) -> WindowScale:
    """Per-window scales so each sample's psi/q targets are O(1) per layer."""
    split = layer_split(cfg)
    ps = psi_daily(w, cfg)
    qs = q_daily(w, cfg)
    return WindowScale(
        psi1=float(ps[:, :split].std()) if ps[:, :split].numel() > 1 else 1.0,
        psi2=float(ps[:, split:].std()) if ps[:, split:].numel() > 1 else 1.0,
        q1=float(qs[:, :split].std()) if qs[:, :split].numel() > 1 else 1.0,
        q2=float(qs[:, split:].std()) if qs[:, split:].numel() > 1 else 1.0,
    )


class QGBatch:
    """Batch for QG neural training: (globally psi-normalized) state target +
    obs + auxiliary raw-PV target.

    `states_q` (the PV/q auxiliary target) is in **raw physical units** --
    unlike `states`/`obs`, it is not normalized (see the module docstring).
    """

    def __init__(self, states, obs, obs_mask, forcing, states_q, rd, params=None):
        self.states = states
        self.obs = obs
        self.obs_mask = obs_mask
        self.forcing = forcing
        self.states_q = states_q
        self.rd = rd
        self.params = params
        self.batch_size, self.T, self.dim = states.shape

    def to(self, device):
        self.states = self.states.to(device)
        self.obs = self.obs.to(device)
        self.obs_mask = self.obs_mask.to(device)
        self.forcing = self.forcing.to(device)
        self.states_q = self.states_q.to(device)
        self.rd = self.rd.to(device)
        if self.params is not None:
            self.params = self.params.to(device)
        return self


class QGNeuralDataset(Dataset):
    """Daily-mean-binned S0 QG windows with a 2-layer streamfunction target.

    Yields (psi_norm, obs_pad, mask, forcing, q_raw, rd) for the QGBatch
    collate. `psi_norm`/`obs_pad` are z-score normalized with the *global*
    per-layer `psi_norm_stats` (mean/std dict, see `data.normalization` and
    `precompute_qg_norm_stats.py`); `q_raw` (the auxiliary PV target) is left
    in raw physical units. `psi_norm_stats=None` reproduces raw (unnormalized)
    psi/obs, e.g. for tests that don't care about normalization.

    `on_the_fly_obs=True` treats `windows` as truth-only (no baked-in obs) and
    redraws the obs/init-state fresh on every `__getitem__` call (see module
    docstring) -- use for train/val to increase obs diversity across epochs.
    Keep the default `False` (fixed, reproducible obs) for test/eval.
    """

    def __init__(self, windows: list, cfg: QGConfig, psi_norm_stats: dict | None = None,
                 on_the_fly_obs: bool = False):
        self.windows = windows
        self.cfg = cfg
        self.psi_norm_stats = psi_norm_stats
        self.on_the_fly_obs = on_the_fly_obs

    def __len__(self) -> int:
        return len(self.windows)

    def _resolved_window(self, idx: int) -> dict:
        w = self.windows[idx]
        if not self.on_the_fly_obs:
            return w
        # `_generate_obs_ic` turns `i` into np/torch seeds via `+ i*101`/`+
        # i*17` on top of `cfg.seed`/`cfg.init_seed`; keep the draw small so
        # the resulting seed stays a valid (< 2**32) RNG seed. Batch-of-1 call
        # (master's `_generate_obs_ic` takes lists of windows/indices).
        draw = random.randrange(1, 1_000_000)
        ic = QGS01Dataset._generate_obs_ic(self.cfg, [w], [draw])[0]
        w = dict(w)
        w.update(ic)
        return w

    def __getitem__(self, idx: int) -> tuple:
        w = self._resolved_window(idx)
        split = layer_split(self.cfg)
        days = num_days(self.cfg)

        psi = psi_daily(w, self.cfg)
        qs = q_daily(w, self.cfg)  # left raw -- see module docstring
        psi_n = psi.clone()
        obs_d, mask_d = _daily_obs_psi(w, self.cfg)
        if self.psi_norm_stats is not None:
            stats = self.psi_norm_stats
            psi_n[:, :split] = normalize(psi_n[:, :split],
                                         {"mean": stats["mean"][0], "std": stats["std"][0]})
            psi_n[:, split:] = normalize(psi_n[:, split:],
                                         {"mean": stats["mean"][1], "std": stats["std"][1]})
            obs_d = normalize(obs_d, {"mean": stats["mean"][0], "std": stats["std"][0]})

        obs_pad = torch.zeros(days, 2 * split)
        obs_pad[:, :split] = torch.nan_to_num(obs_d, nan=0.0)
        mask_full = mask_d.any(dim=-1)
        forcing = torch.zeros(days, dtype=obs_pad.dtype)
        rd = torch.tensor([float(w["true_params"]["rd"])], dtype=torch.float32)
        return psi_n, obs_pad, mask_full, forcing, qs, rd

    def raw_psi(self, idx: int) -> torch.Tensor:
        return psi_daily(self.windows[idx], self.cfg)

    def raw_q(self, idx: int) -> torch.Tensor:
        return q_daily(self.windows[idx], self.cfg)

    def rd(self, idx: int) -> float:
        return float(self.windows[idx]["true_params"]["rd"])

    def scale(self, idx: int) -> WindowScale:
        return window_scales(self.windows[idx], self.cfg)


def qg_collate(batch: list) -> QGBatch:
    states = torch.stack([b[0] for b in batch])
    obs = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    forcing = torch.stack([b[3] for b in batch])
    states_q = torch.stack([b[4] for b in batch])
    rd = torch.stack([b[5] for b in batch]).squeeze(-1)
    return QGBatch(states, obs, masks, forcing, states_q, rd)


def denorm_psi(x: torch.Tensor, cfg: QGConfig, psi_norm_stats: dict | None) -> torch.Tensor:
    """Map a globally-z-score-normalized daily-mean 2-layer psi estimate back
    to physical units (`psi_norm_stats=None` is the identity)."""
    if psi_norm_stats is None:
        return x
    split = layer_split(cfg)
    stats = psi_norm_stats
    x = x.clone()
    x[..., :split] = denormalize(x[..., :split], {"mean": stats["mean"][0], "std": stats["std"][0]})
    x[..., split:] = denormalize(x[..., split:], {"mean": stats["mean"][1], "std": stats["std"][1]})
    return x


def q_from_psi_norm(pred_psi_norm: torch.Tensor, rd: float, cfg: QGConfig,
                    psi_norm_stats: dict | None, device: torch.device) -> torch.Tensor:
    """Physical-units predicted PV from a globally-normalized psi estimate.

    Denormalizes the psi estimate to physical units then spectrally inverts
    it to PV; the result is compared directly against the raw-unit `states_q`
    target (PV is never normalized, see the module docstring).
    """
    psi_phys = denorm_psi(pred_psi_norm, cfg, psi_norm_stats)
    return psi_to_q(psi_phys, rd, cfg, device=device)


def ensure_truth_cache(cfg: QGConfig, num_windows: int, cache_dir: str) -> list:
    """Fixed, reproducible obs windows (test/eval split)."""
    from data.qg import make_qg_s0_s1_datasets
    datasets = make_qg_s0_s1_datasets(cfg, num_test_windows=num_windows, cache_dir=cache_dir)
    return list(datasets["test_s0"])


def ensure_truth_only_cache(cfg: QGConfig, num_windows: int, cache_dir: str) -> list:
    """Truth-only windows (no baked-in obs) for `QGNeuralDataset(on_the_fly_obs=True)`."""
    from data.qg import ensure_truth_only_cache as _ensure
    return _ensure(cfg, num_windows, cache_dir)
