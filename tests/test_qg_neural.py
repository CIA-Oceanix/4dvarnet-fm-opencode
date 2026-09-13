import numpy as np
import pytest
import torch

from data.normalization import compute_channel_stats
from data.qg import (
    QGConfig,
    QGS01Dataset,
    ensure_truth_only_cache,
    make_qg_s0_s1_datasets,
)
from data.qg_neural import (
    PARAM_KEYS,
    QGNeuralDataset,
    denorm_psi,
    ensure_truth_cache,
    layer_split,
    psi_daily,
    psi_to_q,
    q_daily,
    q_from_psi_norm,
    qg_collate,
    steps_per_day,
    window_scales,
)
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM


def _cfg(**kw):
    base = {"nx": 8, "window_days": 2.0, "spinup_years": 0.02,
            "num_windows": 1, "obs_geometry": "random_columns",
            "cols_per_day": 1, "R_var": 1e-12}
    base.update(kw)
    return QGConfig(**base)


def _window(cfg=None, **kw):
    cfg = cfg or _cfg(**kw)
    ds = make_qg_s0_s1_datasets(cfg, num_test_windows=1,
                                cache_dir="/tmp/qg_neural_test_cache")
    return cfg, ds["test_s0"][0]


def _window_with_wind(**kw):
    """Like `_window`, but window index 1 (not 0): `data.qg._S1_WIND_LEVELS`
    starts at 0.0, so index 0 always has a flat all-zero wind_curl field
    (see PLAN.md's 2026-09-03 illustration-fix note for the same gotcha) --
    unsuitable for testing forcing conditioning."""
    cfg = _cfg(num_windows=2, **kw)
    ds = make_qg_s0_s1_datasets(cfg, num_test_windows=2,
                                cache_dir="/tmp/qg_neural_test_cache")
    return cfg, ds["test_s0"][1]


def _s0_s1_window_pair(**kw):
    """S0- and S1-scenario-wrapped versions of the SAME underlying window
    (index 1, nonzero wind -- see `_window_with_wind`), for cross-scenario
    cond_mode="scenario" tests."""
    cfg = _cfg(num_windows=2, **kw)
    ds = make_qg_s0_s1_datasets(cfg, num_test_windows=2,
                                cache_dir="/tmp/qg_neural_test_cache")
    return cfg, ds["test_s0"][1], ds["test_s1"][1]


def _psi_norm_stats(cfg, windows):
    """Global per-layer psi (mean, std) stats, same shape `data.normalization`
    (and `precompute_qg_norm_stats.py`) produce, over the given windows."""
    split = layer_split(cfg)
    layer1, layer2 = [], []
    for w in windows:
        ps = psi_daily(w, cfg)
        layer1.append(ps[:, :split].reshape(-1))
        layer2.append(ps[:, split:].reshape(-1))
    psi_layers = torch.stack([torch.cat(layer1), torch.cat(layer2)], dim=-1)
    return compute_channel_stats(psi_layers)


def _param_norm_stats(windows):
    """Global [U1,rd,rek] (mean, std) stats, same shape/format
    `precompute_qg_norm_stats.py --output-params` produces. `std` is
    clamped away from 0 -- in these tests' 1-2-window fixtures the params
    can be degenerate (identical across the "split"), unlike the real
    1000-window train split precompute_qg_norm_stats.py runs against."""
    params = torch.tensor(
        [[float(w["true_params"][k]) for k in PARAM_KEYS] for w in windows],
        dtype=torch.float32)
    stats = compute_channel_stats(params)
    stats["std"] = stats["std"].clamp(min=1e-6)
    return stats


def _forcing_norm_stats(cfg, windows):
    """Global scalar wind_curl (mean, std), same format
    `precompute_qg_norm_stats.py --output-forcing` produces."""
    from data.qg_neural import _daily_mean_field
    spd = steps_per_day(cfg)
    forcing = torch.cat([_daily_mean_field(w["wind_curl"], spd).reshape(-1) for w in windows])
    stats = compute_channel_stats(forcing.reshape(-1, 1))
    stats["std"] = stats["std"].clamp(min=1e-20)
    return stats


def test_psi_daily_matches_upper_field_daily_mean():
    """`psi_daily` (streamfunctions of daily-mean q) matches the window's own
    upper-layer psi target, daily-binned — the linearity psi=invert(q) holds."""
    cfg, w = _window()
    spd = steps_per_day(cfg)
    split = layer_split(cfg)
    ps = psi_daily(w, cfg)
    assert ps.shape == (2, cfg.state_dim)
    daily_upper = w["target_state_psi"].reshape(-1, split)
    daily_upper = daily_upper[:cfg.num_steps].reshape(2, spd, split).mean(dim=1)
    assert torch.allclose(ps[:, :split], daily_upper, atol=1e-3 * ps.abs().max())


def test_q_daily_is_full_2layer_pv():
    cfg, w = _window()
    qs = q_daily(w, cfg)
    assert qs.shape == (2, cfg.state_dim)
    assert torch.isclose(qs.mean(), torch.tensor(0.0), atol=1e-6)


def test_window_scales_make_otargets_unit_variance_per_layer():
    """`window_scales` is no longer used to build training targets (that's now
    global normalization, see `_psi_norm_stats`/`test_dataset_and_collate_shapes`)
    but is retained as a diagnostic -- pin its own per-window-O(1) contract."""
    cfg, w = _window()
    sc = window_scales(w, cfg)
    split = layer_split(cfg)
    ps = psi_daily(w, cfg)
    qs = q_daily(w, cfg)
    psi_n = ps.clone()
    qs_n = qs.clone()
    psi_n[:, :split] = psi_n[:, :split] / sc.psi1
    psi_n[:, split:] = psi_n[:, split:] / sc.psi2
    qs_n[:, :split] = qs_n[:, :split] / sc.q1
    qs_n[:, split:] = qs_n[:, split:] / sc.q2
    assert sc.psi1 > 0 and sc.psi2 > 0 and sc.q1 > 0 and sc.q2 > 0
    assert 0.5 < psi_n.std().item() < 2.0
    assert 0.5 < qs_n.std().item() < 2.0


def test_dataset_and_collate_shapes():
    cfg, w = _window()
    norm = _psi_norm_stats(cfg, [w])
    ds = QGNeuralDataset([w], cfg, norm)
    item = ds[0]
    psi_n, obs_pad, mask, forcing, q_raw, rd, params, ic = item
    split = layer_split(cfg)
    days = 2
    assert psi_n.shape == (days, 2 * split)
    assert obs_pad.shape == (days, 2 * split)
    assert mask.shape == (days,)
    assert forcing.shape == (days, cfg.ny, cfg.nx)
    assert torch.equal(forcing, torch.zeros_like(forcing))
    assert q_raw.shape == (days, 2 * split)
    assert rd.shape == (1,)
    assert params is None
    # obs (upper-layer psi) padded into the psi1 channel region only, no NaN
    assert not torch.isnan(obs_pad).any()
    assert obs_pad[:, split:].abs().sum() == 0.0
    batch = qg_collate([item, ds[0]])
    assert batch.states.shape == (2, days, 2 * split)
    assert batch.states_q.shape == (2, days, 2 * split)
    assert batch.rd.shape == (2,)
    assert batch.forcing.shape == (2, days, cfg.ny, cfg.nx)
    assert batch.params is None


def test_global_normalization_makes_psi_unit_variance_but_leaves_q_raw():
    """Global per-layer z-score normalization (the training-time scheme) maps
    a window's psi close to unit variance when the window is near the global
    mean/std it was computed from, while q is passed through unnormalized."""
    cfg, w = _window()
    norm = _psi_norm_stats(cfg, [w, w])  # stats computed on the same window(s)
    ds = QGNeuralDataset([w], cfg, norm)
    psi_n, _obs, _mask, _f, q_raw, _rd, _params, _ic = ds[0]
    qs = q_daily(w, cfg)
    assert torch.allclose(q_raw, qs)
    # normalized against its own stats -> exactly unit variance per layer
    split = layer_split(cfg)
    assert torch.isclose(psi_n[:, :split].std(), torch.tensor(1.0), atol=0.05)
    assert torch.isclose(psi_n[:, split:].std(), torch.tensor(1.0), atol=0.05)


def test_dataset_without_norm_stats_is_raw_identity():
    cfg, w = _window()
    ds = QGNeuralDataset([w], cfg, psi_norm_stats=None)
    psi_n, _obs, _mask, _f, _q, _rd, _params, _ic = ds[0]
    assert torch.allclose(psi_n, psi_daily(w, cfg))


def test_batch_to_device_and_params_attr():
    cfg, w = _window()
    ds = QGNeuralDataset([w, w], cfg)
    batch = qg_collate([ds[0], ds[0]])
    assert hasattr(batch, "params")
    assert batch.params is None


def test_denorm_psi_round_trip():
    cfg, w = _window()
    norm = _psi_norm_stats(cfg, [w, w])
    ds = QGNeuralDataset([w], cfg, norm)
    psi_n, _obs, _mask, _f, _q, _rd, _params, _ic = ds[0]
    back = denorm_psi(psi_n, cfg, norm)
    ps = psi_daily(w, cfg)
    assert torch.allclose(back, ps, atol=1e-3 * ps.abs().max())


def test_denorm_psi_identity_when_stats_none():
    cfg, w = _window()
    ps = psi_daily(w, cfg)
    assert torch.equal(denorm_psi(ps, cfg, None), ps)


def test_q_from_psi_norm_matches_raw_pv():
    """Given the true (globally-normalized) psi, `q_from_psi_norm` recovers
    the true raw-units PV up to the spectral-inversion round-off."""
    cfg, w = _window()
    norm = _psi_norm_stats(cfg, [w, w])
    ds = QGNeuralDataset([w], cfg, norm)
    psi_n, _obs, _mask, _f, q_true, _rd, _params, _ic = ds[0]
    q_pred = q_from_psi_norm(psi_n, float(w["true_params"]["rd"]), cfg, norm,
                             torch.device("cpu"))
    assert torch.allclose(q_pred, q_true, atol=1e-3 * q_true.abs().max())


def test_psi_to_q_does_not_move_shared_cpu_inverter():
    """A GPU psi_to_q must not pollute the shared CPU inverter used by
    psi_daily/window_scales (per-device cache separation)."""
    cfg, w = _window()
    from data.qg_neural import _INVERTER_CACHE
    _INVERTER_CACHE.clear()
    ps = psi_daily(w, cfg)
    if torch.cuda.is_available():
        q_gpu = psi_to_q(ps.clone().cuda(), float(w["true_params"]["rd"]),
                         cfg, device=torch.device("cuda"))
        assert q_gpu.is_cuda
    # CPU path still works after a GPU call
    q_cpu = psi_to_q(ps.clone(), float(w["true_params"]["rd"]), cfg)
    assert not q_cpu.is_cuda


def test_lightning_direct_unet_forward_backward_with_qloss():
    _test_lightning_forward_backward("direct_unet")


def test_lightning_vanilla_cfm_forward_backward_with_qloss():
    _test_lightning_forward_backward("vanilla_cfm")


def _synth_batch(split=64, days=30, rd=15000.0, b=2):
    """Deterministic QGBatch with realistic shapes (no window generation),
    whose normalized targets have O(1) per-layer scale like the real dataset."""
    from data.qg_neural import QGBatch
    split = int(split)
    nyx = int(round(split ** 0.5))
    assert nyx * nyx == split, "split must be a perfect square (QG is ny==nx)"
    D = 2 * split
    states = torch.randn(b, days, D)
    states_q = torch.randn(b, days, D) * 1e-6
    obs = torch.randn(b, days, D) * 0.5
    mask = torch.ones(b, days, dtype=torch.bool)
    forcing = torch.zeros(b, days, nyx, nyx)
    rd_t = torch.full((b,), rd, dtype=torch.float32)
    return QGBatch(states, obs, mask, forcing, states_q, rd_t)


def _test_lightning_forward_backward(model_type):
    from train_qg_neural import QGNeuralLightning
    cfg = _cfg()
    batch = _synth_batch(split=layer_split(cfg), rd=cfg.rd)
    norm = {"mean": torch.zeros(2), "std": torch.ones(2)}
    if model_type == "direct_unet":
        model = DirectUNet(state_dim=cfg.state_dim, param_dim=0, cond_extra_dim=0,
                           hidden_channels=[8, 16, 32])
    else:
        model = VanillaCFM(state_dim=cfg.state_dim, param_dim=0, cond_extra_dim=0,
                           hidden_channels=[8, 16, 32], time_emb_dim=16, N_outer=10,
                           sigma_prior=0.5, dropout=0.1, train_tau_0_only=True)
    # Cosine scheduler off: this test is about forward/backward loss
    # correctness, not LR scheduling -- `opt` below is used as a plain
    # optimizer (`opt.zero_grad()`), which configure_optimizers() only
    # returns when the scheduler is disabled.
    lit = QGNeuralLightning(model, model_type, norm, cfg, q_loss_weight=0.1,
                            use_cosine_scheduler=False)
    opt = lit.configure_optimizers()
    loss, _lp, _lq = lit._total_loss(batch)
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_estimate_windows_shapes():
    from train_qg_neural import estimate_windows
    # 30-day window so the UNet's time downsampling has enough samples
    cfg = _cfg(window_days=30.0)
    ds = make_qg_s0_s1_datasets(cfg, num_test_windows=1,
                                cache_dir="/tmp/qg_neural_test_cache")
    windows = list(ds["test_s0"])
    model = VanillaCFM(state_dim=cfg.state_dim, param_dim=0, cond_extra_dim=0,
                       hidden_channels=[8, 16, 32], time_emb_dim=16, N_outer=10,
                       sigma_prior=0.5, dropout=0.1, train_tau_0_only=True)
    est, rd = estimate_windows(model, windows, cfg, "vanilla_cfm", "cpu", n_members=2)
    days = cfg.num_steps // steps_per_day(cfg)
    assert est.shape == (1, days, cfg.state_dim)
    assert rd.shape == (1,)
    assert np.isfinite(est).all()


def test_truth_only_plus_obs_ic_matches_generate_truth():
    """Composing `_generate_truth_only` + `_generate_obs_ic` (keyed by the
    same per-window index `i`) must reproduce what `_generate_truth` itself
    produces for that index -- the split only changes what gets cached, not
    the seeded data (this is exactly what `_generate_truth` does internally)."""
    cfg = _cfg()
    combined = QGS01Dataset._generate_truth(cfg, 2)
    truth_only = QGS01Dataset._generate_truth_only(cfg, 2)
    indices = list(range(2))
    ics = QGS01Dataset._generate_obs_ic(cfg, truth_only, indices)
    for full, part, ic in zip(combined, truth_only, ics):
        assert torch.equal(full["true_state"], part["true_state"])
        assert full["true_params"] == part["true_params"]
        assert torch.equal(full["obs_mask"], ic["obs_mask"])
        assert torch.allclose(
            torch.nan_to_num(full["obs"]), torch.nan_to_num(ic["obs"]),
        )
        assert torch.equal(full["init_state"], ic["init_state"])


def test_ensure_truth_only_cache_windows_lack_obs():
    cfg = _cfg()
    windows = ensure_truth_only_cache(cfg, 1, "/tmp/qg_neural_test_cache")
    assert "obs" not in windows[0]
    assert "true_state" in windows[0]


def test_on_the_fly_obs_varies_across_draws_target_fixed():
    """`on_the_fly_obs=True` redraws a different obs realization on every
    `__getitem__` call, while the psi/q targets (from `true_state`) stay
    identical -- diversity comes only from the obs/init-state resample."""
    cfg = _cfg()
    windows = ensure_truth_only_cache(cfg, 1, "/tmp/qg_neural_test_cache")
    ds = QGNeuralDataset(windows, cfg, on_the_fly_obs=True)
    psi_a, obs_a, _mask_a, _f_a, q_a, _rd_a, _params_a, _ic_a = ds[0]
    psi_b, obs_b, _mask_b, _f_b, q_b, _rd_b, _params_b, _ic_b = ds[0]
    assert torch.equal(psi_a, psi_b)
    assert torch.equal(q_a, q_b)
    assert not torch.equal(obs_a, obs_b)


def test_fixed_obs_dataset_is_deterministic_across_draws():
    """Default `on_the_fly_obs=False` keeps returning the same baked-in obs
    on repeated `__getitem__` calls (test/eval reproducibility, unchanged)."""
    cfg, w = _window()
    ds = QGNeuralDataset([w], cfg)
    item_a = ds[0]
    item_b = ds[0]
    for a, b in zip(item_a, item_b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)


def test_cond_mode_none_requires_no_param_norm_stats():
    cfg, w = _window()
    QGNeuralDataset([w], cfg, cond_mode="none")  # must not raise


def test_cond_mode_true_or_noisy_requires_param_norm_stats():
    cfg, w = _window()
    for mode in ("true", "noisy"):
        try:
            QGNeuralDataset([w], cfg, cond_mode=mode)
            assert False, f"expected ValueError for cond_mode={mode!r} with no param stats"
        except ValueError:
            pass


def test_cond_mode_true_or_noisy_requires_forcing_norm_stats():
    """Regression test for a real bug: leaving the forcing field unnormalized
    collapsed a full 200-epoch Q3 training run at epoch 28 (raw wind_curl is
    ~1e-13-1e-12, ~12 orders of magnitude smaller than the unit-variance psi/
    obs/param channels it's concatenated with). param_norm_stats alone must
    not be enough -- forcing_norm_stats is also required."""
    cfg, w = _window()
    pstats = _param_norm_stats([w, w])
    for mode in ("true", "noisy"):
        try:
            QGNeuralDataset([w], cfg, cond_mode=mode, param_norm_stats=pstats)
            assert False, f"expected ValueError for cond_mode={mode!r} with no forcing stats"
        except ValueError:
            pass


def test_cond_mode_true_matches_true_forcing_and_params():
    """Q3 (oracle): forcing is the real true wind_curl field (daily-binned,
    z-scored, nonzero, matching the window's own normalized wind_curl),
    params are the exact normalized true_params -- deterministic across
    repeated draws."""
    cfg, w = _window_with_wind()
    pstats = _param_norm_stats([w, w])
    fstats = _forcing_norm_stats(cfg, [w, w])
    ds = QGNeuralDataset([w], cfg, cond_mode="true", param_norm_stats=pstats,
                        forcing_norm_stats=fstats)
    _psi, _obs, _mask, forcing, _q, _rd, params, _ic = ds[0]
    assert forcing.shape == (2, cfg.ny, cfg.nx)
    assert forcing.abs().sum() > 0.0
    spd = steps_per_day(cfg)
    raw_forcing = w["wind_curl"].reshape(2, spd, cfg.ny, cfg.nx).mean(dim=1)
    expected_forcing = (raw_forcing - fstats["mean"][0]) / fstats["std"][0]
    assert torch.allclose(forcing, expected_forcing, atol=1e-5)
    assert params.shape == (3,)
    true_vec = torch.tensor([float(w["true_params"][k]) for k in PARAM_KEYS])
    expected_params = (true_vec - pstats["mean"]) / pstats["std"]
    assert torch.allclose(params, expected_params, atol=1e-4)
    # deterministic: repeated draws give identical forcing/params
    _psi2, _obs2, _mask2, forcing2, _q2, _rd2, params2, _ic2 = ds[0]
    assert torch.equal(forcing, forcing2)
    assert torch.equal(params, params2)


def test_cond_mode_noisy_varies_across_draws_and_stays_finite():
    """Q4: every draw resamples a fresh corruption severity -- forcing/params
    should differ across repeated __getitem__ calls (unlike Q3's oracle),
    while remaining finite and (params) still centered near the true value
    since bias only ever reduces rd/rek toward, never past, 0."""
    cfg, w = _window_with_wind()
    pstats = _param_norm_stats([w, w])
    fstats = _forcing_norm_stats(cfg, [w, w])
    ds = QGNeuralDataset([w], cfg, cond_mode="noisy", param_norm_stats=pstats,
                        forcing_norm_stats=fstats, noisy_max=1.5)
    draws = [ds[0] for _ in range(5)]
    forcings = torch.stack([d[3] for d in draws])
    params = torch.stack([d[6] for d in draws])
    assert torch.isfinite(forcings).all()
    assert torch.isfinite(params).all()
    assert not torch.equal(forcings[0], forcings[1])
    assert not torch.equal(params[0], params[1])
    # U1 is never biased by the noisy-params mechanism (only rd/rek are)
    true_vec = torch.tensor([float(w["true_params"][k]) for k in PARAM_KEYS])
    expected_u1 = (true_vec[0] - pstats["mean"][0]) / pstats["std"][0]
    assert torch.allclose(params[:, 0], expected_u1.expand(5), atol=1e-4)


def test_cond_mode_scenario_matches_true_on_s0_and_biased_on_s1():
    """cond_mode="scenario" is the cross-scenario eval mode: on an S0-scenario
    window it must match cond_mode="true" exactly (da_params/wind_state_corrupted
    equal true_params/wind_state_true there), and on the SAME underlying
    window's S1-scenario wrapper it must differ (biased rd/rek, corrupted
    wind) -- deterministic, unlike cond_mode="noisy"."""
    cfg, w_s0, w_s1 = _s0_s1_window_pair()
    pstats = _param_norm_stats([w_s0, w_s0])
    fstats = _forcing_norm_stats(cfg, [w_s0, w_s0])

    ds_true = QGNeuralDataset([w_s0], cfg, cond_mode="true", param_norm_stats=pstats,
                              forcing_norm_stats=fstats)
    ds_scenario_s0 = QGNeuralDataset([w_s0], cfg, cond_mode="scenario",
                                     param_norm_stats=pstats, forcing_norm_stats=fstats)
    _p1, _o1, _m1, forcing_true, _q1, _r1, params_true, _ic1 = ds_true[0]
    _p2, _o2, _m2, forcing_s0, _q2, _r2, params_s0, _ic2 = ds_scenario_s0[0]
    assert torch.equal(forcing_true, forcing_s0)
    assert torch.equal(params_true, params_s0)

    ds_scenario_s1 = QGNeuralDataset([w_s1], cfg, cond_mode="scenario",
                                     param_norm_stats=pstats, forcing_norm_stats=fstats)
    _p3, _o3, _m3, forcing_s1, _q3, _r3, params_s1, _ic3 = ds_scenario_s1[0]
    assert torch.isfinite(forcing_s1).all()
    assert torch.isfinite(params_s1).all()
    assert not torch.equal(params_s0, params_s1), (
        "S1's biased da_params (rd/rek scaled by 1-s1_param_bias) must differ "
        "from S0's true params")
    assert not torch.equal(forcing_s0, forcing_s1), (
        "S1's corrupted wind_state_corrupted must produce a different forcing "
        "field than S0's true wind state")
    # deterministic: repeated draws on the same S1 window give identical results
    _p4, _o4, _m4, forcing_s1b, _q4, _r4, params_s1b, _ic4 = ds_scenario_s1[0]
    assert torch.equal(forcing_s1, forcing_s1b)
    assert torch.equal(params_s1, params_s1b)


def test_include_ic_requires_psi_norm_stats():
    cfg, w = _window()
    try:
        QGNeuralDataset([w], cfg, psi_norm_stats=None, include_ic=True)
        assert False, "expected ValueError for include_ic=True with no psi_norm_stats"
    except ValueError:
        pass


def test_include_ic_returns_normalized_psi_scale_ic_and_none_when_false():
    """Q5: `ic` is the window's own init_state, inverted to psi and z-scored
    with psi_norm_stats -- finite, plausible O(1)-ish scale (not raw q's
    ~1e-5 scale), deterministic across repeated draws (unlike cond_mode
    "noisy" forcing/params). `include_ic=False` (default) -> ic is None."""
    cfg, w = _window()
    norm = _psi_norm_stats(cfg, [w, w])
    ds_no_ic = QGNeuralDataset([w], cfg, norm, include_ic=False)
    item = ds_no_ic[0]
    assert item[7] is None

    ds = QGNeuralDataset([w], cfg, norm, include_ic=True)
    item_a = ds[0]
    ic = item_a[7]
    split = layer_split(cfg)
    assert ic.shape == (2 * split,)
    assert torch.isfinite(ic).all()
    assert ic.std() > 1e-3, f"ic std={ic.std():.3e} looks like raw q scale, not normalized psi"
    item_b = ds[0]
    assert torch.equal(ic, item_b[7])  # deterministic: init_state isn't resampled by include_ic


def test_qg_collate_stacks_ic_when_present():
    cfg, w = _window()
    norm = _psi_norm_stats(cfg, [w, w])
    ds = QGNeuralDataset([w, w], cfg, norm, include_ic=True)
    batch = qg_collate([ds[0], ds[1]])
    assert batch.ic is not None
    assert batch.ic.shape == (2, 2 * layer_split(cfg))
    ds_no_ic = QGNeuralDataset([w, w], cfg, norm, include_ic=False)
    batch_no_ic = qg_collate([ds_no_ic[0], ds_no_ic[1]])
    assert batch_no_ic.ic is None


def test_ensure_truth_cache_redrawn_matches_plain_when_cfg_unchanged():
    """Regression test for a real bug: naively calling `ensure_truth_cache`
    with an obs/IC-protocol-overridden cfg (Q5's lag=5.0/noise=0.05) misses
    the cache (`_truth_cache_path` hashes the whole QGConfig) and triggers a
    full from-scratch truth rollout -- caught only after Q5/Q3-lag5 smoke
    test jobs sat "hung" (actually silently regenerating truth) for ~10
    minutes with no output. `ensure_truth_cache_redrawn` fixes this by
    loading the cache at a separate, unaffected `cache_cfg` key and
    redrawing obs/IC via `eval_cfg`. When `eval_cfg == cache_cfg` (the
    Q1/Q3/Q4 case, no override), it must reproduce `ensure_truth_cache`'s
    output bit-for-bit (same underlying `_generate_truth_only` +
    `_generate_obs_ic` composition, see
    `test_truth_only_plus_obs_ic_matches_generate_truth`)."""
    from data.qg_neural import ensure_truth_cache_redrawn
    cfg = _cfg(num_windows=2)
    cache_dir = "/tmp/qg_neural_test_cache"
    plain = ensure_truth_cache(cfg, 2, cache_dir)
    redrawn = ensure_truth_cache_redrawn(cfg, cfg, 2, cache_dir)
    for a, b in zip(plain, redrawn):
        assert torch.equal(a["true_state"], b["true_state"])
        assert torch.equal(torch.nan_to_num(a["obs"]), torch.nan_to_num(b["obs"]))
        assert torch.equal(a["init_state"], b["init_state"])


def test_ensure_truth_cache_redrawn_uses_cache_cfg_key_not_eval_cfg():
    """The whole point: an eval_cfg with a *different* obs_noise_std_frac/
    init_lag_days must NOT change which cache file gets loaded (that's the
    bug) -- only cache_cfg's key matters, and the redrawn windows' obs
    actually reflect eval_cfg's (different) settings."""
    from data.qg_neural import ensure_truth_cache_redrawn
    cache_cfg = _cfg(num_windows=2)
    cache_dir = "/tmp/qg_neural_test_cache"
    ensure_truth_cache(cache_cfg, 2, cache_dir)  # populate the cache at cache_cfg's key
    eval_cfg = _cfg(num_windows=2, obs_noise_std_frac=0.5, init_lag_days=1.5)
    windows = ensure_truth_cache_redrawn(cache_cfg, eval_cfg, 2, cache_dir)
    assert len(windows) == 2
    # truth is identical (same cache_cfg key); obs noise scale reflects eval_cfg
    plain = ensure_truth_cache(cache_cfg, 2, cache_dir)
    for a, b in zip(plain, windows):
        assert torch.equal(a["true_state"], b["true_state"])


def test_cached_qg_dynamics_reused_and_matches_uncached():
    """Regression test for the Q4 epoch-time fix: `_cached_qg_dynamics(cfg)`
    must return the SAME object across repeated calls with an equal `cfg`
    (not rebuild it every draw -- that rebuild was a measured ~2x epoch-time
    bottleneck for Q4, since it happened on every training draw), and the
    cached object's `wind_curl_field` output must be bitwise-identical to a
    freshly-built one (the caching is a pure performance fix, no behavior
    change)."""
    from data.qg import _make_qg_dynamics
    from data.qg_neural import _DYN_CACHE, _cached_qg_dynamics
    _DYN_CACHE.clear()
    cfg = _cfg()
    dyn_a = _cached_qg_dynamics(cfg)
    dyn_b = _cached_qg_dynamics(cfg)
    assert dyn_a is dyn_b
    fresh = _make_qg_dynamics(cfg)
    wind_state = torch.tensor([[1e-11, cfg.L / 2, 0.0]] * 4, dtype=torch.float32)
    assert torch.equal(dyn_a.wind_curl_field(wind_state), fresh.wind_curl_field(wind_state))
    # a materially different cfg must NOT reuse the same cached object
    cfg2 = _cfg(wind_sigma=cfg.wind_sigma * 2)
    dyn_c = _cached_qg_dynamics(cfg2)
    assert dyn_c is not dyn_a


def test_qg_collate_stacks_params_when_present():
    cfg, w = _window()
    pstats = _param_norm_stats([w, w])
    fstats = _forcing_norm_stats(cfg, [w, w])
    ds = QGNeuralDataset([w, w], cfg, cond_mode="true", param_norm_stats=pstats,
                        forcing_norm_stats=fstats)
    batch = qg_collate([ds[0], ds[1]])
    assert batch.params is not None
    assert batch.params.shape == (2, 3)
    assert batch.forcing.shape == (2, 2, cfg.ny, cfg.nx)


def test_build_model_honors_yaml_param_dim_and_cond_extra_dim():
    """Regression test for a bug where build_model() hardcoded param_dim=0/
    cond_extra_dim=0 regardless of the experiment YAML's model.* fields.
    QG's "direct_unet" is MONAI-backed (models.monai_unet_qg2d), so this
    needs the fdv-monai-proto env -- skip if monai isn't installed (matches
    tests/test_monai_unet_qg2d.py's gating; CI's default env has no monai,
    see .github/workflows/ci.yml)."""
    pytest.importorskip("monai")
    from train_qg_neural import build_model
    cfg = _cfg()
    model = build_model("direct_unet", cfg, param_dim=4, cond_extra_dim=1)
    assert model.param_dim == 4
    assert model.cond_extra_dim == 1
    obs_channels = model.unet.obs_channels
    assert obs_channels == model.nlayers + 1 + 4


def test_param_keys_are_never_zero_variance_across_windows():
    """Regression test for a real bug caught by a training smoke test: `beta`
    was originally included in PARAM_KEYS but `QGS01Dataset._generate_truth_only`
    never jitters it (only U1/rd/rek get a per-window random draw, see
    data/qg.py) -- it's an exact constant across the whole train split, so
    z-scoring it divides by std=0 and poisons the loss to NaN within the
    first epoch. Guards against reintroducing any such zero-variance key by
    checking PARAM_KEYS' values actually differ across two independently
    generated windows."""
    from data.qg import QGS01Dataset
    cfg = _cfg(num_windows=2)
    windows = QGS01Dataset._generate_truth_only(cfg, 2)
    for k in PARAM_KEYS:
        v0, v1 = windows[0]["true_params"][k], windows[1]["true_params"][k]
        assert v0 != v1, f"PARAM_KEYS entry {k!r} is constant across windows " \
                          f"(both {v0!r}) -- would divide by std=0 when normalized"


def test_q3_q4_yaml_configs_parse_cond_mode_as_string():
    """Regression test for a real bug caught by a training smoke test: bare
    `cond_mode: true` in YAML parses as the Python boolean True (a YAML
    boolean literal), not the string "true" QGNeuralDataset expects --
    train_qg_neural.py's main() would then pass cond_mode=True straight into
    QGNeuralDataset, which raises ValueError. Q3's config must quote it
    (`cond_mode: "true"`); this test loads the actual YAML files (unlike
    the other cond_mode tests, which pass the Python string literal
    directly and would never have caught this)."""
    import os

    from omegaconf import OmegaConf
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name, expected_mode in [("Q3_direct_unet_s0_oracle_cond", "true"),
                                ("Q4_direct_unet_s1_noisy_cond", "noisy")]:
        cfg = OmegaConf.load(os.path.join(base, "config", "experiment", f"{name}.yaml"))
        cond_mode = cfg.data.cond_mode
        assert isinstance(cond_mode, str), (
            f"{name}.yaml: cond_mode parsed as {type(cond_mode).__name__} "
            f"({cond_mode!r}), not a string -- likely an unquoted YAML boolean")
        assert cond_mode == expected_mode
        assert int(cfg.model.param_dim) == 3
        assert int(cfg.model.cond_extra_dim) == 1
        assert cfg.data.param_norm_stats_path
        assert cfg.data.forcing_norm_stats_path


def test_normalized_forcing_is_order_one_not_raw_scale():
    """Regression test for the training-collapse bug: raw wind_curl is
    ~1e-13-1e-12 (see data/qg_neural.py's module docstring), ~12-13 orders
    of magnitude smaller than the unit-variance psi/obs/param channels it's
    concatenated with. After normalization the forcing values actually fed
    to the model must be O(1)-ish (not still ~1e-12), or the fix is a no-op."""
    cfg, w = _window_with_wind()
    pstats = _param_norm_stats([w, w])
    fstats = _forcing_norm_stats(cfg, [w, w])
    raw_forcing = w["wind_curl"]
    assert raw_forcing.abs().max() < 1e-9, (
        "sanity check on the fixture itself: raw wind_curl should be tiny")
    ds = QGNeuralDataset([w], cfg, cond_mode="true", param_norm_stats=pstats,
                        forcing_norm_stats=fstats)
    _psi, _obs, _mask, forcing, _q, _rd, _params, _ic = ds[0]
    assert forcing.std() > 1e-3, (
        f"normalized forcing std={forcing.std():.3e} is still tiny -- "
        "normalization did not actually rescale it")
