import numpy as np
import torch

from data.normalization import compute_channel_stats
from data.qg import (
    QGConfig,
    QGS01Dataset,
    ensure_truth_only_cache,
    make_qg_s0_s1_datasets,
)
from data.qg_neural import (
    QGNeuralDataset,
    denorm_psi,
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
    psi_n, obs_pad, mask, forcing, q_raw, rd = item
    split = layer_split(cfg)
    days = 2
    assert psi_n.shape == (days, 2 * split)
    assert obs_pad.shape == (days, 2 * split)
    assert mask.shape == (days,)
    assert forcing.shape == (days,)
    assert q_raw.shape == (days, 2 * split)
    assert rd.shape == (1,)
    # obs (upper-layer psi) padded into the psi1 channel region only, no NaN
    assert not torch.isnan(obs_pad).any()
    assert obs_pad[:, split:].abs().sum() == 0.0
    batch = qg_collate([item, ds[0]])
    assert batch.states.shape == (2, days, 2 * split)
    assert batch.states_q.shape == (2, days, 2 * split)
    assert batch.rd.shape == (2,)


def test_global_normalization_makes_psi_unit_variance_but_leaves_q_raw():
    """Global per-layer z-score normalization (the training-time scheme) maps
    a window's psi close to unit variance when the window is near the global
    mean/std it was computed from, while q is passed through unnormalized."""
    cfg, w = _window()
    norm = _psi_norm_stats(cfg, [w, w])  # stats computed on the same window(s)
    ds = QGNeuralDataset([w], cfg, norm)
    psi_n, _obs, _mask, _f, q_raw, _rd = ds[0]
    qs = q_daily(w, cfg)
    assert torch.allclose(q_raw, qs)
    # normalized against its own stats -> exactly unit variance per layer
    split = layer_split(cfg)
    assert torch.isclose(psi_n[:, :split].std(), torch.tensor(1.0), atol=0.05)
    assert torch.isclose(psi_n[:, split:].std(), torch.tensor(1.0), atol=0.05)


def test_dataset_without_norm_stats_is_raw_identity():
    cfg, w = _window()
    ds = QGNeuralDataset([w], cfg, psi_norm_stats=None)
    psi_n, _obs, _mask, _f, _q, _rd = ds[0]
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
    psi_n, _obs, _mask, _f, _q, _rd = ds[0]
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
    psi_n, _obs, _mask, _f, q_true, _rd = ds[0]
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
    D = 2 * split
    states = torch.randn(b, days, D)
    states_q = torch.randn(b, days, D) * 1e-6
    obs = torch.randn(b, days, D) * 0.5
    mask = torch.ones(b, days, dtype=torch.bool)
    forcing = torch.zeros(b, days)
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
    psi_a, obs_a, _mask_a, _f_a, q_a, _rd_a = ds[0]
    psi_b, obs_b, _mask_b, _f_b, q_b, _rd_b = ds[0]
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
