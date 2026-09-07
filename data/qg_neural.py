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

Normalization is fixed per-layer scalars (psi per layer, q per layer, obs per
field) computed on the training split and applied identically everywhere so eval
maps back to physical units.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from data.qg import QGConfig, expand_obs_to_grid

_INVERTER_CACHE: dict = {}


@dataclass
class QGNorm:
    psi1: float = 1.0
    psi2: float = 1.0
    q1: float = 1.0
    q2: float = 1.0
    obs: float = 1.0

    @property
    def psi_scale(self):
        return [self.psi1, self.psi2]

    @property
    def q_scale(self):
        return [self.q1, self.q2]


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


def compute_norm(windows: list, cfg: QGConfig) -> QGNorm:
    """Global per-layer normalization scales (legacy; see `window_scales`).

    Retained for report/inspection only. Training uses per-window scales
    (`WindowScale`) so samples with widely varying streamfunction energy are
    each made O(1); this global scalar form is kept for backward compatibility
    and informational display.
    """
    split = layer_split(cfg)
    psi1, psi2, q1, q2, obs_vals = [], [], [], [], []
    for w in windows:
        ps = psi_daily(w, cfg)
        qs = q_daily(w, cfg)
        psi1.append(ps[:, :split].reshape(-1))
        psi2.append(ps[:, split:].reshape(-1))
        q1.append(qs[:, :split].reshape(-1))
        q2.append(qs[:, split:].reshape(-1))
        od, md = _daily_obs_psi(w, cfg)
        obs_vals.append(od[md].reshape(-1))
    return QGNorm(
        psi1=float(torch.cat(psi1).std()) if psi1 and torch.cat(psi1).numel() > 1 else 1.0,
        psi2=float(torch.cat(psi2).std()) if psi2 and torch.cat(psi2).numel() > 1 else 1.0,
        q1=float(torch.cat(q1).std()) if q1 and torch.cat(q1).numel() > 1 else 1.0,
        q2=float(torch.cat(q2).std()) if q2 and torch.cat(q2).numel() > 1 else 1.0,
        obs=float(torch.cat(obs_vals).std()) if obs_vals and torch.cat(obs_vals).numel() > 1 else 1.0,
    )


@dataclass
class WindowScale:
    """Per-window per-layer normalization scales (streamfunction psi + PV q)."""

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
    """Batch for QG neural training: psi target + obs + auxiliary q target + per-window scales.

    `scale` is a (B, 4) tensor [psi1, psi2, q1, q2] used to (de)normalize each
    sample's psi estimate and the auxiliary q-loss.
    """

    def __init__(self, states, obs, obs_mask, forcing, states_q, rd, scale, params=None):
        self.states = states
        self.obs = obs
        self.obs_mask = obs_mask
        self.forcing = forcing
        self.states_q = states_q
        self.rd = rd
        self.scale = scale
        self.params = params
        self.batch_size, self.T, self.dim = states.shape

    def to(self, device):
        self.states = self.states.to(device)
        self.obs = self.obs.to(device)
        self.obs_mask = self.obs_mask.to(device)
        self.forcing = self.forcing.to(device)
        self.states_q = self.states_q.to(device)
        self.rd = self.rd.to(device)
        self.scale = self.scale.to(device)
        if self.params is not None:
            self.params = self.params.to(device)
        return self


class QGNeuralDataset(Dataset):
    """Daily-mean-binned S0 QG windows with a 2-layer streamfunction target.

    Yields (psi_norm, obs_pad, mask, forcing, q_norm, rd, WindowScale) for the
    QGBatch collate. Each window is normalized by its own per-layer scales
    (`window_scales`) so samples with very different streamfunction energy are
    each mapped to O(1), avoiding a single global scalar being dominated by the
    highest-energy windows.
    """

    def __init__(self, windows: list, cfg: QGConfig, norm: QGNorm | None = None):
        self.windows = windows
        self.cfg = cfg
        self.norm = norm

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple:
        w = self.windows[idx]
        split = layer_split(self.cfg)
        days = num_days(self.cfg)
        sc = window_scales(w, self.cfg)

        psi = psi_daily(w, self.cfg)
        qs = q_daily(w, self.cfg)
        psi_n = psi.clone()
        qs_n = qs.clone()
        psi_n[:, :split] = psi_n[:, :split] / sc.psi1
        psi_n[:, split:] = psi_n[:, split:] / sc.psi2
        qs_n[:, :split] = qs_n[:, :split] / sc.q1
        qs_n[:, split:] = qs_n[:, split:] / sc.q2

        obs_d, mask_d = _daily_obs_psi(w, self.cfg)
        obs_pad = torch.zeros(days, 2 * split)
        obs_d_n = torch.nan_to_num(obs_d / sc.psi1, nan=0.0)
        obs_pad[:, :split] = obs_d_n
        mask_full = mask_d.any(dim=-1)
        forcing = torch.zeros(days, dtype=obs_pad.dtype)
        rd = torch.tensor([float(w["true_params"]["rd"])], dtype=torch.float32)
        return psi_n, obs_pad, mask_full, forcing, qs_n, rd, sc

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
    scale = WindowScale.collate([b[6] for b in batch])
    return QGBatch(states, obs, masks, forcing, states_q, rd, scale)


def denorm_state(x: torch.Tensor, cfg: QGConfig, scale) -> torch.Tensor:
    """Map a normalized daily-mean 2-layer psi estimate back to physical units.

    `scale` is a (2,) or (..., 2) psi-scale tensor ([psi1, psi2]); when `x` has
    a batch/time leading dimension of size equal to `scale.shape[-2]` the scales
    broadcast per-sample, otherwise a single (2,) scale applies to all.
    """
    split = layer_split(cfg)
    s = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    psi1 = s[..., 0]
    psi2 = s[..., 1]
    x = x.clone()
    if x.dim() >= 2 and s.dim() >= 2 and x.shape[-3] == s.shape[-2] \
            and s.dim() == x.dim() - 1:
        x[..., :split] = x[..., :split] * psi1[..., None]
        x[..., split:] = x[..., split:] * psi2[..., None]
    else:
        x[..., :split] = x[..., :split] * psi1
        x[..., split:] = x[..., split:] * psi2
    return x


def norm_q_from_psi(pred_psi_norm: torch.Tensor, rd: float, cfg: QGConfig,
                    scale, device: torch.device) -> torch.Tensor:
    """Normalized predicted PV from a normalized flattened psi estimate (per window).

    `scale` is a (4,) tensor [psi1, psi2, q1, q2] used to denormalize the psi
    estimate then re-normalize the resulting PV.
    """
    split = layer_split(cfg)
    s = torch.as_tensor(scale, dtype=pred_psi_norm.dtype, device=device)
    psi_phys = denorm_state(pred_psi_norm, cfg, s[:2])
    q_phys = psi_to_q(psi_phys, rd, cfg, device=device)
    q_n = q_phys.clone()
    q_n[..., :split] = q_n[..., :split] / s[2]
    q_n[..., split:] = q_n[..., split:] / s[3]
    return q_n


def ensure_truth_cache(cfg: QGConfig, num_windows: int, cache_dir: str) -> list:
    from data.qg import make_qg_s0_s1_datasets
    datasets = make_qg_s0_s1_datasets(cfg, num_test_windows=num_windows, cache_dir=cache_dir)
    return list(datasets["test_s0"])
