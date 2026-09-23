import numpy as np
import pytest
import torch

from data.obs_times import resample_obs_variable
from evaluation.baselines import ETKF, EnKF, ObsOperator, Strong4DVar, _channel_obs_mask, _obs0_for_init
from evaluation.run_l96 import make_fast_ring_fill, make_obs_j_indices
from models.lorenz96_dynamics import Lorenz96Dynamics

IDX = make_obs_j_indices(8, 4, 2)
PARAMS = dict(F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1)


def _kw(B):
    return {k: torch.full((B,), v) for k, v in PARAMS.items()}


def test_resample_obs_variable_first_step_step0_has_window_k():
    g = torch.Generator().manual_seed(1)
    states = torch.randn(16, 3000, 24)
    obs, mask = resample_obs_variable(states, torch.zeros(16, 3000, dtype=torch.bool), 0.5,
                                      (5, 50), (4, 16), generator=g, first_step=True)
    assert mask[:, 0].all()
    for b in range(16):
        rows = torch.isfinite(obs[b, mask[b]])
        assert rows[:, :8].all()
        k = rows[:, 8:].sum(dim=1)
        assert (k == k[0]).all() and 4 <= int(k[0]) <= 16
        assert 5 <= int(mask[b].sum()) <= 50


def test_fast_ring_fill_interpolates_along_periodic_ring():
    fill = make_fast_ring_fill(8, 4, 2)
    pos = np.array([k * 4 + j for k in range(8) for j in range(2)], dtype=float)
    v = torch.cat([torch.arange(8.0) * 10, torch.tensor(pos)]).float()
    v[8 + np.array([3, 4, 5])] = float("nan")
    v[8 + 15] = float("nan")
    out = fill(v.unsqueeze(0))[0]
    assert torch.equal(out[:8], torch.arange(8.0) * 10)
    assert torch.allclose(out[8 + np.array([3, 4, 5])], torch.tensor(pos[[3, 4, 5]]).float())
    assert out[8 + 15].item() == pytest.approx(28.0 + (29.0 - 28.0) / (32.0 - 28.0) * (0.0 - 28.0))
    assert torch.isfinite(out).all()


def test_obs0_for_init_rejects_missing_channels_without_fill():
    v = torch.randn(2, 24)
    v[0, 12] = float("nan")
    with pytest.raises(ValueError):
        _obs0_for_init(v, None)
    assert torch.isfinite(_obs0_for_init(v, make_fast_ring_fill())).all()


def test_channel_obs_mask_requires_time_and_finite_value():
    obs = torch.randn(1, 3, 4)
    obs[0, 1, 2] = float("nan")
    m = _channel_obs_mask(obs, torch.tensor([[True, True, False]]))
    assert m[0, 0].all() and not m[0, 2].any()
    assert m[0, 1].tolist() == [True, True, False, True]


def _s1_setup(T=60, times=(0, 20, 40)):
    torch.manual_seed(0)
    dyn = Lorenz96Dynamics(dt=0.001, NO=8, J=2, h=1.0, hx=1.0, eps=0.1, coupling_exponent=1.0)
    truth = torch.randn(1, T, 24)
    mask = torch.zeros(1, T, dtype=torch.bool)
    mask[:, list(times)] = True
    obs = torch.full((1, T, 24), float("nan"))
    obs[:, list(times)] = truth[:, list(times)] + 0.3 * torch.randn(1, len(times), 24)
    return dyn, truth, obs, mask


@pytest.mark.parametrize("cls", [ETKF, EnKF])
def test_missing_channels_equal_reduced_obs_operator(cls, monkeypatch):
    """Channels NaN in the obs rows give exactly the analysis of an operator
    that does not observe them (EnKF: obs perturbations zeroed so both runs
    are deterministic)."""
    dyn, truth, obs, mask = _s1_setup()
    drop = [9, 14, 20]
    keep = [c for c in range(24) if c not in drop]
    obs_nan = obs.clone()
    obs_nan[:, 1:, drop] = float("nan")
    ens0 = truth[0, 0] + 0.5 * torch.randn(12, 24)
    if cls is EnKF:
        monkeypatch.setattr(torch, "randn", lambda *s, device=None, dtype=None, **k: torch.zeros(
            *(s[0] if len(s) == 1 and isinstance(s[0], tuple) else s), device=device, dtype=dtype))
    common = dict(N_ensemble=12, dt=0.001, inflation=1.2, dynamics=dyn, NO=8, J=2, init_ensemble=ens0)
    full = cls(obs_operator=ObsOperator(24, list(range(24))), **common)
    red = cls(obs_operator=ObsOperator(24, keep), **common)
    a = full.assimilate_batch(obs_nan, mask, torch.zeros(1, 60), truth, **_kw(1))[0].trajectory
    b = red.assimilate_batch(obs[..., keep], mask, torch.zeros(1, 60), truth, **_kw(1))[0].trajectory
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_batched_enkf_missing_channels_do_not_leak_across_windows():
    dyn, truth, obs, mask = _s1_setup()
    truth2 = torch.cat([truth, truth + 0.1])
    obs2 = torch.cat([obs, obs + 0.1])
    obs2[0, 20:, 12:18] = float("nan")
    mask2 = mask.repeat(2, 1)
    ens0 = truth[0, 0] + 0.5 * torch.randn(12, 24)
    enkf = EnKF(N_ensemble=12, dt=0.001, inflation=1.2, dynamics=dyn, NO=8, J=2, init_ensemble=ens0,
                obs_operator=ObsOperator(24, list(range(24))))
    res = enkf.assimilate_batch(obs2, mask2, torch.zeros(2, 60), truth2, **_kw(2))
    for r in res:
        assert np.isfinite(r.trajectory).all()
        assert r.ensemble_variance[-1].mean() > 1e-3


def test_strong4dvar_missing_channels_are_not_fit_to_zero():
    dyn, truth, obs, mask = _s1_setup()
    truth = truth + 5.0
    obs = obs + 5.0
    obs[:, 1:, 8:] = float("nan")
    sv = Strong4DVar(da_window_steps=60, dt=0.001, max_iter=10, lr=0.2, dynamics=dyn,
                     obs_operator=ObsOperator(24, list(range(24))))
    traj = sv.assimilate_batch(obs, mask, torch.zeros(1, 60), truth, **_kw(1))[0].trajectory
    assert np.isfinite(traj).all()
    assert traj[20:, 8:].mean() > 4.0


def test_draw_layout_covers_every_da_window():
    from eval_da_random_layout_l96 import covers_da_windows, draw_layout
    truth = torch.randn(3000, 24)
    for seed in range(40):
        obs, mask = draw_layout(truth, (6, 11), (4, 16), seed, 500)
        assert mask[0] and covers_da_windows(mask, 500)
        assert 6 <= int(mask.sum()) <= 11
    assert not covers_da_windows(torch.tensor([True] + [False] * 999), 500)


def test_draw_layout_refuses_counts_below_the_number_of_da_windows():
    from eval_da_random_layout_l96 import draw_layout
    with pytest.raises(ValueError):
        draw_layout(torch.randn(3000, 24), (5, 50), (4, 16), 0, 500)


def test_build_cell_restricts_cases_and_keeps_layouts_independent_of_case_selection():
    from eval_da_random_layout_l96 import build_cell
    T = 3000
    datasets = {key: [{"true_state": torch.randn(T, 40)} for _ in range(2)] for key in ("test_s0", "test_s1")}
    both, lay_both = build_cell(datasets, [0, 1], 1, (10, 20), (4, 16), IDX, 500)
    s0, lay_s0 = build_cell(datasets, [0, 1], 1, (10, 20), (4, 16), IDX, 500, cases=("s0",))
    assert set(s0) == {"test_s0"} and set(lay_s0) == {"s0"}
    assert torch.equal(lay_s0["s0"]["obs_mask"], lay_both["s0"]["obs_mask"])
    assert torch.equal(torch.nan_to_num(lay_s0["s0"]["obs"]), torch.nan_to_num(lay_both["s0"]["obs"]))


def test_parse_case_inflation_accepts_scalar_and_per_case():
    from evaluation.run_l96 import L96_DA_INFLATION, parse_case_inflation
    assert parse_case_inflation("2.0") == 2.0 and parse_case_inflation(1.5) == 1.5
    assert parse_case_inflation("s0=1.5,s1=2.0") == {"s0": 1.5, "s1": 2.0} == L96_DA_INFLATION
    with pytest.raises(ValueError):
        parse_case_inflation("s0=1.5")


def test_per_case_inflation_reaches_each_case_and_keeps_scalar_cache_names():
    from evaluation.run_l96 import _case_cfg, inflation_tag
    cfg = {"inflation": {"s0": 1.5, "s1": 2.0}, "init_fill": None}
    assert _case_cfg(cfg, "s0")["inflation"] == 1.5 and _case_cfg(cfg, "s1")["inflation"] == 2.0
    assert _case_cfg({"inflation": 2.0}, "s0") == {"inflation": 2.0}
    assert inflation_tag(2.0) == "2.0"
    assert inflation_tag({"s1": 2.0, "s0": 1.5}) == "s0-1.5_s1-2.0"
