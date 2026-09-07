import numpy as np
import torch

from data.qg import QGConfig, make_qg_s0_s1_datasets
from data.qg_neural import (
    QGNeuralDataset,
    QGNorm,
    WindowScale,
    compute_norm,
    denorm_state,
    layer_split,
    norm_q_from_psi,
    psi_daily,
    psi_to_q,
    q_daily,
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
    cfg, w = _window()
    sc = window_scales(w, cfg)
    split = layer_split(cfg)
    ps = psi_daily(w, cfg)
    qs = q_daily(w, cfg)
    psi_n = ps.clone()
    psi_n[:, :split] = psi_n[:, :split] / sc.psi1
    psi_n[:, split:] = psi_n[:, split:] / sc.psi2
    q_n = qs.clone()
    q_n[:, :split] = q_n[:, :split] / sc.q1
    q_n[:, split:] = q_n[:, split:] / sc.q2
    assert sc.psi1 > 0 and sc.psi2 > 0 and sc.q1 > 0 and sc.q2 > 0
    assert 0.5 < psi_n.std().item() < 2.0
    assert 0.5 < q_n.std().item() < 2.0


def test_dataset_and_collate_shapes():
    cfg, w = _window()
    ds = QGNeuralDataset([w], cfg)
    item = ds[0]
    psi_n, obs_pad, mask, forcing, q_n, rd, _sc = item
    split = layer_split(cfg)
    days = 2
    assert psi_n.shape == (days, 2 * split)
    assert obs_pad.shape == (days, 2 * split)
    assert mask.shape == (days,)
    assert forcing.shape == (days,)
    assert q_n.shape == (days, 2 * split)
    assert rd.shape == (1,)
    assert isinstance(_sc, WindowScale)
    # obs (upper-layer psi) padded into the psi1 channel region only, no NaN
    assert not torch.isnan(obs_pad).any()
    assert obs_pad[:, split:].abs().sum() == 0.0
    batch = qg_collate([item, ds[0]])
    assert batch.states.shape == (2, days, 2 * split)
    assert batch.scale.shape == (2, 4)
    assert batch.rd.shape == (2,)


def test_batch_to_device_and_params_attr():
    cfg, w = _window()
    ds = QGNeuralDataset([w, w], cfg)
    batch = qg_collate([ds[0], ds[0]])
    assert hasattr(batch, "params")
    assert batch.params is None


def test_denorm_state_round_trip():
    cfg, w = _window()
    sc = window_scales(w, cfg)
    split = layer_split(cfg)
    ps = psi_daily(w, cfg)
    psi_n = ps.clone()
    psi_n[:, :split] = psi_n[:, :split] / sc.psi1
    psi_n[:, split:] = psi_n[:, split:] / sc.psi2
    s = torch.tensor([sc.psi1, sc.psi2])
    back = denorm_state(psi_n, cfg, s)
    assert torch.allclose(back, ps, atol=1e-5 * ps.abs().max())


def test_norm_q_from_psi_round_trip():
    """Given the true normalized psi, `norm_q_from_psi` recovers the true
    normalized PV up to the spectral-inversion round-off."""
    cfg, w = _window()
    sc = window_scales(w, cfg)
    split = layer_split(cfg)
    ps = psi_daily(w, cfg)
    qs = q_daily(w, cfg)
    psi_n = ps.clone()
    psi_n[:, :split] = psi_n[:, :split] / sc.psi1
    psi_n[:, split:] = psi_n[:, split:] / sc.psi2
    scale = torch.tensor([sc.psi1, sc.psi2, sc.q1, sc.q2])
    q_pred = norm_q_from_psi(psi_n, float(w["true_params"]["rd"]), cfg, scale,
                             torch.device("cpu"))
    q_true = qs.clone()
    q_true[:, :split] = q_true[:, :split] / sc.q1
    q_true[:, split:] = q_true[:, split:] / sc.q2
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
    s1, s2 = 3.0, 5.0
    q1, q2 = 1e-6, 1e-6
    states = torch.randn(b, days, D)
    states_q = torch.randn(b, days, D)
    obs = torch.randn(b, days, D) * 0.5
    mask = torch.ones(b, days, dtype=torch.bool)
    forcing = torch.zeros(b, days)
    rd_t = torch.full((b,), rd, dtype=torch.float32)
    scale = torch.tensor([[s1, s2, q1, q2]] * b, dtype=torch.float32)
    return QGBatch(states, obs, mask, forcing, states_q, rd_t, scale)


def _test_lightning_forward_backward(model_type):
    from train_qg_neural import QGNeuralLightning
    cfg = _cfg()
    batch = _synth_batch(split=layer_split(cfg), rd=cfg.rd)
    if model_type == "direct_unet":
        model = DirectUNet(state_dim=cfg.state_dim, param_dim=0, cond_extra_dim=0,
                           hidden_channels=[8, 16, 32])
    else:
        model = VanillaCFM(state_dim=cfg.state_dim, param_dim=0, cond_extra_dim=0,
                           hidden_channels=[8, 16, 32], time_emb_dim=16, N_outer=10,
                           sigma_prior=0.5, dropout=0.1, train_tau_0_only=True)
    lit = QGNeuralLightning(model, model_type, QGNorm(), cfg, q_loss_weight=0.1)
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


def test_compute_norm_returns_positive_scales():
    cfg, w = _window()
    norm = compute_norm([w, w], cfg)
    assert norm.psi1 > 0 and norm.psi2 > 0
    assert norm.q1 > 0 and norm.q2 > 0
