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

**Forcing + params conditioning (Q3/Q4, 2026-09-10):** ``QGNeuralDataset``
can additionally condition the estimator on the wind-forcing field and the
physical params (``rd``/``rek``/``U1`` -- **not** ``beta``, see
``PARAM_KEYS``'s comment: it's an exact constant across the train split,
so z-scoring it divides by zero), controlled by ``cond_mode``:

* ``"none"`` (default, Q1/Q2 behavior) -- ``forcing`` is an all-zero
  ``(days, ny, nx)`` field, ``params`` is ``None``.
* ``"true"`` (Q3, oracle) -- ``forcing`` is the window's real ``wind_curl``
  spatial field (from the *true* trajectory), daily-mean binned; ``params``
  is the window's exact ``true_params`` vector ``[U1, rd, rek]``. Both
  are deterministic per window (no resampling).
* ``"noisy"`` (Q4) -- mirrors the L96 SDA3 CFM study's per-step resampled
  corruption (see PLAN.md's 2026-09-10 QG Q3/Q4 section) rather than one
  fixed S1 bias: every ``__getitem__`` draws a fresh random severity
  fraction in ``[0, noisy_max]`` of the full S1 corruption (``s1_amp_bias``/
  ``s1_loc_sigma_frac``/``s1_sigma_eta_frac`` for the wind, ``s1_param_bias``
  for ``rd``/``rek``), so training sees a distribution of corruption
  severity rather than a single fixed operating point.
* ``"scenario"`` (cross-scenario S0/S1 **eval only**, not used for training)
  -- deterministic, no resampling: reads whatever the window's own S0/S1
  scenario wrapper (``QGS01Dataset._scenario_window``) designates as
  believed (``wind_state_corrupted``/``da_params``, which equal
  ``wind_state_true``/``true_params`` exactly for an S0-scenario window) --
  the same biased forcing/params a DA method's dynamical model sees under
  S1, unlike ``"true"``/``"noisy"`` which always ignore the scenario label.

``forcing``/``params`` are always returned in **physical units** from the
dataset; z-score normalization (mirroring the psi/obs normalization above)
is applied via explicit stats dicts -- required whenever ``cond_mode !=
"none"``:

* ``param_norm_stats`` (``{"mean", "std"}`` over ``[U1, rd, rek]``,
  produced by ``precompute_qg_norm_stats.py``'s ``--output-params``) --
  the 3 physical params span ~9 orders of magnitude raw.
* ``forcing_norm_stats`` (a single global scalar ``{"mean", "std"}`` over
  the pooled ``wind_curl`` field, produced by ``precompute_qg_norm_stats.py``'s
  ``--output-forcing``). **Added 2026-09-10 after a real training failure**:
  the forcing field was originally left unnormalized on the (unverified)
  assumption its per-grid-cell scale was "already comparable" to the
  z-scored obs/psi channels -- it is not: raw ``wind_curl`` is
  ``O(1e-13)-O(1e-12)`` (`wind_amp` ranges 0-3e-11 over a Witch-of-Agnesi
  profile peaking at ~1), roughly 12-13 orders of magnitude smaller than
  the unit-variance psi/obs/param channels it is concatenated with. A full
  200-epoch Q3 training run collapsed at epoch 28 (loss frozen thereafter
  at exactly the "predict-the-zero-mean" baseline, ``loss_psi≈1 +
  q_loss_weight·loss_q≈1``, i.e. the model died) -- the forcing channel's
  negligible raw scale is the leading suspect (a conv layer needing large
  weights to extract any signal from a ~1e-12-scale channel is a plausible
  destabilization mechanism). Spatial structure (storm location) is
  preserved by z-scoring with one global scalar, not per-grid-cell, stats.

**Initial-condition conditioning (Q5, 2026-09-11):** ``QGNeuralDataset``
can additionally condition on the raw initial-condition snapshot via
``include_ic=True`` -- a **third, distinct conditioning class** from
``forcing``/``params``: unlike ``forcing`` (one field *per day*) or
``params`` (one scalar *vector*), the IC is one static spatial field for
the *whole window*, unaffected by ``cond_mode`` (always the true
``window["init_state"]``, like obs -- never scenario-corrupted, since
physically it represents a recent analysis/observation, not the DA model's
own internal forecast). Inverted to psi via the per-window spectral
inverter (``_ic_field``) and z-scored with the same global
``psi_norm_stats`` already used for state/obs -- no new stats file. Neither
``init_state`` nor any IC-derived quantity was referenced anywhere in this
module before Q5 -- Q1/Q3/Q4 have zero equivalent of the background/IC
skill DA baselines get from rolling forward a sampled init state.
"""

import random
from dataclasses import dataclass, replace as _dc_replace

import numpy as np
import torch
from torch.utils.data import Dataset

from data.normalization import denormalize, normalize
from data.qg import (
    QGConfig,
    QGS01Dataset,
    _make_corrupted_wind_state,
    _make_qg_dynamics,
    expand_obs_to_grid,
)

_INVERTER_CACHE: dict = {}
_DYN_CACHE: dict = {}

PARAM_KEYS = ("U1", "rd", "rek")


def _cached_qg_dynamics(cfg: QGConfig):
    """Per-cfg cached `QGDynamics`, mirroring `_reconstruct_inverter`'s cache
    pattern. `_make_qg_dynamics(cfg)` rebuilds the full spectral PV-inversion
    machinery (wavenumber grids, filters) from scratch, none of which
    `wind_curl_field` actually needs (only the (x,y) grid + wind_sigma/L/W)
    -- but the real fix is not recomputing it at all: this was being called
    once per training draw (~1000/epoch at batch_size=2), a measured ~2x
    epoch-time bottleneck in Q4 that a training run caught (see PLAN.md's
    2026-09-10 QG Q3/Q4 section). Deterministic given `cfg`, so caching by
    the exact fields `_make_qg_dynamics` reads is safe (no result change).
    """
    key = (cfg.nx, cfg.L, cfg.dt, cfg.beta, cfg.rd, cfg.delta, cfg.U1, cfg.U2,
           cfg.rek, cfg.filterfac, cfg.wind_amp, cfg.wind_tau_days, cfg.wind_sigma,
           cfg.wind_cx, cfg.wind_cy, cfg.wind_drift_tau_days, cfg.wind_drift_sigma,
           cfg.wind_seed)
    if key not in _DYN_CACHE:
        _DYN_CACHE[key] = _make_qg_dynamics(cfg)
    return _DYN_CACHE[key]
# `beta` is deliberately excluded: `QGS01Dataset._generate_truth_only` never
# jitters it (only U1/rd/rek get a per-window `u,r,k` random draw, see
# data/qg.py), so it is an exact constant across the whole train split --
# z-score normalizing a zero-variance channel divides by std=0 (confirmed by
# a training smoke test: param_dim=4 with beta included produced NaN loss
# within the first epoch). A constant channel also carries zero information
# for the network to condition on regardless of normalization.


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


def _daily_mean_field(field: torch.Tensor, spd: int) -> torch.Tensor:
    """Daily-mean bin a spatial (T, ny, nx) field to (days, ny, nx)."""
    T, ny, nx = field.shape
    if T % spd != 0:
        raise ValueError(f"T={T} not divisible by steps_per_day={spd}")
    return field.reshape(T // spd, spd, ny, nx).mean(dim=1)


def _true_params_vector(window: dict) -> torch.Tensor:
    tp = window["true_params"]
    return torch.tensor([float(tp[k]) for k in PARAM_KEYS], dtype=torch.float32)


def _true_forcing_and_params(window: dict, cfg: QGConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Q3 (oracle): the real true wind_curl field + exact true params, no jitter."""
    forcing = _daily_mean_field(window["wind_curl"], steps_per_day(cfg))
    return forcing, _true_params_vector(window)


def _scenario_forcing_and_params(window: dict, cfg: QGConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-scenario eval (Q3/Q4 evaluated on S0 vs S1 test windows,
    see PLAN.md's 2026-09-10 QG Q3/Q4 evaluation-plan section): deterministic,
    no resampling -- reads whatever the window's *own* scenario wrapper
    (`data.qg.QGS01Dataset._scenario_window`) designates as the "believed"
    model: `wind_state_corrupted`/`da_params` for an S1-scenario window (the
    exact same biased forcing/params a DA method's dynamical model sees
    under S1), which fall back to `wind_state_true`/`true_params` for an
    S0-scenario window since `_scenario_window`'s "test_s0" branch sets
    `wind_state_corrupted = wind_state_true` and `da_params = true_params`
    exactly -- so this one code path is correct for both scenarios uniformly.
    Distinct from `cond_mode="true"`/`"noisy"` (training-time modes, which
    always use the true/freshly-resampled-synthetic values regardless of
    scenario label -- appropriate for training diversity, but NOT a fair S1
    apples-to-apples test since they never actually consume the scenario's
    own defined bias).
    """
    ws = window.get("wind_state_corrupted", window["wind_state_true"])
    dyn = _cached_qg_dynamics(cfg)
    wind_curl = dyn.wind_curl_field(ws)
    forcing = _daily_mean_field(wind_curl, steps_per_day(cfg))
    params_src = window.get("da_params", window["true_params"])
    params = torch.tensor([float(params_src[k]) for k in PARAM_KEYS], dtype=torch.float32)
    return forcing, params


def _noisy_forcing_and_params(window: dict, cfg: QGConfig,
                              noisy_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Q4: resample a fresh random severity fraction of the full S1
    corruption (wind + rd/rek bias) every call, mirroring the L96 SDA3 CFM
    study's per-step resampled noisy-params conditioning (see module
    docstring) instead of one fixed S1 operating point.
    """
    draw = random.randrange(1, 1_000_000)
    rng = np.random.RandomState(draw)
    frac = float(rng.uniform(0.0, noisy_max))
    scaled_cfg = _dc_replace(cfg, s1_amp_bias=cfg.s1_amp_bias * frac,
                             s1_loc_sigma_frac=cfg.s1_loc_sigma_frac * frac,
                             s1_sigma_eta_frac=cfg.s1_sigma_eta_frac * frac)
    ws_corrupt = _make_corrupted_wind_state(scaled_cfg, window["wind_state_true"], draw)
    dyn = _cached_qg_dynamics(cfg)
    wind_curl_corrupt = dyn.wind_curl_field(ws_corrupt)
    forcing = _daily_mean_field(wind_curl_corrupt, steps_per_day(cfg))
    tp = window["true_params"]
    b = cfg.s1_param_bias * frac
    params = torch.tensor(
        [float(tp["U1"]), float(tp["rd"]) * (1.0 - b), float(tp["rek"]) * (1.0 - b)],
        dtype=torch.float32)
    return forcing, params


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


def _ic_field(window: dict, cfg: QGConfig, psi_norm_stats: dict) -> torch.Tensor:
    """Q5: the raw initial-condition snapshot (`window["init_state"]`,
    physical PV/q, always from the *true* trajectory -- like obs, never
    scenario-corrupted, since physically it represents a recent analysis/
    observation, not the DA model's own internal forecast). Inverted to psi
    and z-scored with the same global `psi_norm_stats` already used for the
    state/obs channels (a third conditioning class, distinct from forcing/
    params: one static field per window, not per-day/scalar -- see
    `QGNeuralDataset`'s `include_ic` docstring)."""
    rd = float(window["true_params"]["rd"])
    inv = _reconstruct_inverter(cfg, rd)
    ic_psi = inv.inner.streamfunctions(window["init_state"].reshape(1, -1))
    if ic_psi.dim() == 4:
        ic_psi = ic_psi.reshape(ic_psi.shape[0], -1)
    ic_psi = ic_psi.squeeze(0)
    split = layer_split(cfg)
    ic_n = ic_psi.clone()
    stats = psi_norm_stats
    ic_n[:split] = normalize(ic_n[:split], {"mean": stats["mean"][0], "std": stats["std"][0]})
    ic_n[split:] = normalize(ic_n[split:], {"mean": stats["mean"][1], "std": stats["std"][1]})
    return ic_n


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

    def __init__(self, states, obs, obs_mask, forcing, states_q, rd, params=None, ic=None):
        self.states = states
        self.obs = obs
        self.obs_mask = obs_mask
        self.forcing = forcing
        self.states_q = states_q
        self.rd = rd
        self.params = params
        self.ic = ic
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
        if self.ic is not None:
            self.ic = self.ic.to(device)
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

    `cols_per_day_range=(lo, hi)` additionally resamples `cfg.cols_per_day`
    itself uniformly in `[lo, hi]` (inclusive) on every draw (requires
    `on_the_fly_obs=True`) -- obs-density-augmented training, so a single
    checkpoint generalizes across observing-network densities instead of
    only the one fixed value it happened to train at. Mirrors L96's
    obs-density-augmented training (`data/obs_density.py`), adapted to QG's
    per-window (not per-timestep) `cols_per_day` granularity: the same
    resampled count applies to every day within one window-draw, only the
    count itself varies draw-to-draw. Leave `None` (default) for the
    original fixed-density behavior; keep `None` for val/test so evaluation
    stays at one stable reference density.
    """

    def __init__(self, windows: list, cfg: QGConfig, psi_norm_stats: dict | None = None,
                 on_the_fly_obs: bool = False, cond_mode: str = "none",
                 param_norm_stats: dict | None = None, noisy_max: float = 1.5,
                 forcing_norm_stats: dict | None = None, include_ic: bool = False,
                 cols_per_day_range: tuple[int, int] | None = None):
        if cols_per_day_range is not None and not on_the_fly_obs:
            raise ValueError(
                "cols_per_day_range requires on_the_fly_obs=True -- otherwise "
                "each window's obs is drawn once (at whatever cols_per_day the "
                "truth cache/cfg carries) and never redrawn, so the range would "
                "have no effect")
        if cols_per_day_range is not None:
            lo, hi = cols_per_day_range
            if not (1 <= lo <= hi):
                raise ValueError(
                    f"cols_per_day_range must satisfy 1 <= min <= max, got "
                    f"{cols_per_day_range!r}")
            max_cols = steps_per_day(cfg)
            if cfg.obs_geometry == "random_columns" and hi > max_cols:
                raise ValueError(
                    f"cols_per_day_range max ({hi}) exceeds cfg.dt's "
                    f"steps_per_day ({max_cols}) -- "
                    "_generate_random_column_observations assigns each of "
                    "cols_per_day distinct columns to its own distinct "
                    "intra-day time slot, so a value above steps_per_day "
                    "can never be satisfied and its collision-avoidance loop "
                    "spins forever (confirmed: hung a real GPU job, see "
                    "PLAN.md's 2026-09-13 Q1-obsdensity note). Lower the "
                    "range max to at most steps_per_day, or use a smaller dt.")
        if cond_mode not in ("none", "true", "noisy", "scenario"):
            hint = (" (YAML `cond_mode: true` parses as the boolean True, not "
                    "this string -- quote it as `cond_mode: \"true\"`)"
                    if isinstance(cond_mode, bool) else "")
            raise ValueError(f"unknown cond_mode {cond_mode!r}{hint}")
        if cond_mode != "none" and param_norm_stats is None:
            raise ValueError(
                "param_norm_stats is required when cond_mode != 'none' -- the "
                "3 physical params span ~9 orders of magnitude raw, see "
                "precompute_qg_norm_stats.py --output-params")
        if cond_mode != "none" and forcing_norm_stats is None:
            raise ValueError(
                "forcing_norm_stats is required when cond_mode != 'none' -- raw "
                "wind_curl is ~1e-13-1e-12, ~12 orders of magnitude smaller than "
                "the unit-variance psi/obs/param channels it's concatenated with; "
                "leaving it unnormalized destabilized a real training run (see "
                "module docstring), see precompute_qg_norm_stats.py --output-forcing")
        if include_ic and psi_norm_stats is None:
            raise ValueError(
                "psi_norm_stats is required when include_ic=True -- the IC is "
                "inverted to psi and z-scored with the same global psi stats "
                "used for state/obs (Q5, see data.qg_neural._ic_field)")
        self.windows = windows
        self.cfg = cfg
        self.psi_norm_stats = psi_norm_stats
        self.on_the_fly_obs = on_the_fly_obs
        self.cond_mode = cond_mode
        self.param_norm_stats = param_norm_stats
        self.noisy_max = noisy_max
        self.forcing_norm_stats = forcing_norm_stats
        self.include_ic = include_ic
        self.cols_per_day_range = cols_per_day_range

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
        cfg = self.cfg
        if self.cols_per_day_range is not None:
            lo, hi = self.cols_per_day_range
            # Same per-draw-cfg-override pattern as `_noisy_forcing_and_params`'s
            # `_dc_replace` call below -- a fresh `cols_per_day` this draw only,
            # the shared `self.cfg` (and any other dataset instance using it,
            # e.g. a fixed-density val/test split) is untouched.
            cfg = _dc_replace(cfg, cols_per_day=random.randint(lo, hi))
        ic = QGS01Dataset._generate_obs_ic(cfg, [w], [draw])[0]
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
        rd = torch.tensor([float(w["true_params"]["rd"])], dtype=torch.float32)

        if self.cond_mode == "none":
            forcing = torch.zeros(days, self.cfg.ny, self.cfg.nx, dtype=obs_pad.dtype)
            params = None
        else:
            if self.cond_mode == "true":
                forcing, params = _true_forcing_and_params(w, self.cfg)
            elif self.cond_mode == "noisy":
                forcing, params = _noisy_forcing_and_params(w, self.cfg, self.noisy_max)
            else:
                forcing, params = _scenario_forcing_and_params(w, self.cfg)
            forcing = forcing.to(obs_pad.dtype)
            forcing = normalize(forcing, self.forcing_norm_stats)
            params = normalize(params, self.param_norm_stats)

        ic = _ic_field(w, self.cfg, self.psi_norm_stats) if self.include_ic else None

        return psi_n, obs_pad, mask_full, forcing, qs, rd, params, ic

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
    params = None if batch[0][6] is None else torch.stack([b[6] for b in batch])
    ic = None if batch[0][7] is None else torch.stack([b[7] for b in batch])
    return QGBatch(states, obs, masks, forcing, states_q, rd, params=params, ic=ic)


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


def ensure_truth_cache_redrawn(cache_cfg: QGConfig, eval_cfg: QGConfig, num_windows: int,
                               cache_dir: str, scenario: str = "test_s0") -> list:
    """Like `ensure_truth_cache`, but for when `eval_cfg`'s obs/IC protocol
    (`obs_noise_std_frac`/`init_lag_days`/`s1_param_bias`/`s1_amp_bias`/...)
    differs from the cached truth's own key. Loads the cached truth at
    `cache_cfg`'s key (a cache HIT -- `cache_cfg` should share `nx`/`seed`/
    `num_windows` with the production cache and leave every other field at
    the `QGConfig` default), then cheaply redraws obs/init-state at
    `eval_cfg`'s actual settings via `QGS01Dataset._generate_obs_ic` (truth
    generation doesn't depend on the obs/IC protocol, only sampling does).

    Calling `ensure_truth_cache(eval_cfg, ...)` directly would silently MISS
    the cache (`_truth_cache_path` hashes the *whole* `QGConfig`) and trigger
    a full from-scratch truth rollout -- caught the hard way when Q5/Q3-lag5
    smoke tests were first launched with that naive call and had to be
    killed after ~10 minutes of silent, expensive regeneration (see PLAN.md's
    2026-09-12 note). Mirrors `eval_qg_neural_s0_s1.py`'s identical fix.
    """
    from data.qg import QGS01Dataset, _truth_cache_path
    path = _truth_cache_path(cache_cfg, num_windows, cache_dir)
    base = torch.load(path, map_location="cpu")[:num_windows]
    raw = QGS01Dataset(eval_cfg, scenario, base_windows=base).windows
    ic = QGS01Dataset._generate_obs_ic(eval_cfg, raw, list(range(len(raw))))
    return [dict(w, **entry) for w, entry in zip(raw, ic)]


def ensure_truth_only_cache(cfg: QGConfig, num_windows: int, cache_dir: str) -> list:
    """Truth-only windows (no baked-in obs) for `QGNeuralDataset(on_the_fly_obs=True)`."""
    from data.qg import ensure_truth_only_cache as _ensure
    return _ensure(cfg, num_windows, cache_dir)
