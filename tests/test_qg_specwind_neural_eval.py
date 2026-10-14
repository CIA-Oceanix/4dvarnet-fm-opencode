import numpy as np
import pytest
import torch

from data.qg import QGConfig
from data.qg_neural import psi_daily, steps_per_day
from evaluation.run_qg_specwind_da import FIELDS
from evaluation.run_qg_specwind_neural import bin_centres, daily_to_steps, score_window

SPD = 12


def test_bin_centres_are_the_mean_step_of_each_day():
    assert np.allclose(bin_centres(3, SPD), [5.5, 17.5, 29.5])


def test_cubic_reproduces_a_cubic_in_time_exactly():
    t = np.arange(30 * SPD, dtype=float)
    poly = lambda x: 2e-6 * x ** 3 - 1e-3 * x ** 2 + 0.5 * x - 3.0  # noqa: E731
    daily = poly(bin_centres(30, SPD))[:, None] * np.ones((1, 5))
    out = daily_to_steps(daily, SPD, len(t), "cubic")
    assert np.allclose(out, poly(t)[:, None], rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("kind", ["cubic", "linear"])
def test_constant_and_linear_series_are_kept(kind):
    n = 30 * SPD
    const = np.full((2, 30, 4), 7.0)
    assert np.allclose(daily_to_steps(const, SPD, n, kind), 7.0)
    lin = (0.3 * bin_centres(30, SPD) - 1.0)[:, None] * np.ones((1, 4))
    assert np.allclose(daily_to_steps(lin, SPD, n, kind), (0.3 * np.arange(n) - 1.0)[:, None])


def test_unknown_interpolation_is_refused():
    with pytest.raises(ValueError):
        daily_to_steps(np.zeros((30, 2)), SPD, 360, "nearest")


def _window(cfg: QGConfig, n_days: int = 3) -> dict:
    g = torch.Generator().manual_seed(0)
    q0 = torch.randn(2, cfg.ny * cfg.nx, generator=g) * 1e-5
    q0 = (q0 - q0.mean(1, keepdim=True)).reshape(-1)
    tp = {"U1": 0.0, "U2": 0.0, "rd": 4.0e4, "rek": 1e-7, "beta": cfg.beta}
    return {"true_state": q0.repeat(n_days * steps_per_day(cfg), 1), "true_params": tp}


def test_a_perfect_daily_estimate_of_a_steady_field_scores_one():
    cfg = QGConfig(nx=16)
    w = _window(cfg)
    psi = psi_daily(w, cfg).numpy()
    m = score_window(psi[None], w, cfg, torch.device("cpu"))
    for f in FIELDS:
        assert m[f"ev_da_{f}"] == pytest.approx(1.0, abs=1e-6)
        assert m[f"daily_ev_da_{f}"] == pytest.approx(1.0, abs=1e-6)
    assert m["crps"] < 1e-5 * float(w["true_state"].std())
    assert "spread_ratio_q1" not in m
    assert not any(k.startswith("ev_free") for k in m)


def test_ensemble_scores_carry_crps_and_spread():
    cfg = QGConfig(nx=16)
    w = _window(cfg)
    psi = psi_daily(w, cfg).numpy()
    rng = np.random.default_rng(0)
    members = psi[None] + rng.normal(scale=0.1 * psi.std(), size=(5, *psi.shape))
    m = score_window(members, w, cfg, torch.device("cpu"))
    assert m["crps"] > 0.0
    assert m["spread_ratio_q1"] > 0.0 and m["spread_ratio_q2"] > 0.0
    assert m["score"] < 1.0


def _sda_prior_and_batch(nx: int = 16, days: int = 4, b: int = 2):
    from types import SimpleNamespace

    from models.monai_unet_qg2d import MonaiSDAPriorQGChannelTime
    torch.manual_seed(0)
    model = MonaiSDAPriorQGChannelTime(ny=nx, nx=nx, T=days, hidden_channels=[8, 16, 32],
                                       norm_num_groups=4).eval()
    d = 2 * nx * nx
    mask = torch.zeros(b, days, d, dtype=torch.bool)
    mask[:, :, : nx * nx : 7] = True
    obs = torch.where(mask, torch.full((b, days, d), 2.0), torch.zeros(b, days, d))
    return model, SimpleNamespace(obs=obs, obs_mask=mask)


def test_sda_members_without_guidance_reproduce_the_prior_sample():
    from evaluation.run_qg_specwind_neural import sda_members
    model, batch = _sda_prior_and_batch()
    torch.manual_seed(1)
    guided = sda_members(model, batch, n_members=1, guidance_weight=0.0, n_steps=5)
    torch.manual_seed(1)
    x = torch.randn(batch.obs.shape) * model.sigma_prior
    with torch.no_grad():
        for k in range(5):
            x = x + 0.2 * model(x, batch, torch.full((2,), k / 5))
    assert guided.shape == (1, *batch.obs.shape)
    assert torch.allclose(guided[0], x, atol=1e-6)


def test_sda_guidance_pulls_observed_cells_towards_the_obs():
    from evaluation.run_qg_specwind_neural import sda_members
    model, batch = _sda_prior_and_batch()
    misfit = {}
    for w in (0.0, 5.0):
        torch.manual_seed(2)
        x = sda_members(model, batch, n_members=3, guidance_weight=w, n_steps=5)
        assert x.shape == (3, *batch.obs.shape)
        misfit[w] = float(((x - batch.obs) ** 2)[:, batch.obs_mask].mean())
    assert misfit[5.0] < misfit[0.0]


class _DayLocalFlow(torch.nn.Module):
    """A 'network' whose velocity at each day depends on that day only: composing it over
    sub-windows must reproduce it applied to the whole window."""

    def __init__(self, T: int, d: int):
        super().__init__()
        self.T, self.state_dim, self.sigma_prior, self.N_outer = T, d, 1.0, 4
        self.train_tau_0_only = False
        from models.interpolant import LinearInterpolant
        self.interpolant = LinearInterpolant(nu=1.0)

    def forward(self, x, batch, tau):
        if x.shape[1] != self.T:
            raise ValueError("wrong window")
        return torch.tanh(x) * (1.0 + tau[:, None, None])


def test_composed_flow_reproduces_a_day_local_velocity_on_the_full_window():
    from evaluation.run_qg_specwind_neural import ComposedFlow
    _, batch = _sda_prior_and_batch(nx=4, days=12)
    inner = _DayLocalFlow(T=5, d=batch.obs.shape[-1])
    flow = ComposedFlow(inner, n_days=12, stride=2)
    x = torch.randn(batch.obs.shape)
    tau = torch.full((2,), 0.3)
    assert torch.allclose(flow(x, batch, tau), torch.tanh(x) * 1.3, atol=1e-6)
    assert flow.sample(batch).shape == batch.obs.shape


def test_composed_flow_samples_keep_the_spread_of_independent_members():
    from evaluation.run_qg_specwind_neural import ComposedFlow
    _, batch = _sda_prior_and_batch(nx=4, days=12)

    class Zero(_DayLocalFlow):
        def forward(self, x, batch, tau):
            return torch.zeros_like(x)

    flow = ComposedFlow(Zero(T=5, d=batch.obs.shape[-1]), n_days=12, stride=2)
    torch.manual_seed(0)
    members = torch.stack([flow.sample(batch) for _ in range(200)])
    assert members.std(0).mean() == pytest.approx(1.0, abs=0.05)


def test_sda_members_with_a_composed_prior_cover_the_long_window():
    from evaluation.run_qg_specwind_neural import ComposedFlow, sda_members
    _, batch = _sda_prior_and_batch(days=6)
    from models.monai_unet_qg2d import MonaiSDAPriorQGChannelTime
    prior = MonaiSDAPriorQGChannelTime(ny=16, nx=16, T=4, hidden_channels=[8, 16, 32],
                                       norm_num_groups=4).eval()
    flow = ComposedFlow(prior, n_days=6, stride=1)
    x = sda_members(flow, batch, n_members=2, guidance_weight=5.0, n_steps=3)
    assert x.shape == (2, *batch.obs.shape) and torch.isfinite(x).all()


def test_warm_start_at_tau0_one_returns_the_starting_estimate():
    from evaluation.run_qg_specwind_neural import sda_members
    model, batch = _sda_prior_and_batch()
    start = torch.randn(batch.obs.shape)
    x = sda_members(model, batch, n_members=1, guidance_weight=5.0, n_steps=5,
                    mean_estimate=start, tau0=1.0)
    assert torch.allclose(x[0], start, atol=1e-6)


def test_renorm_inverts_denorm():
    from data.qg_neural import denorm_psi
    from evaluation.run_qg_specwind_neural import renorm_psi
    cfg = QGConfig(nx=16)
    norm = {"mean": torch.tensor([1.0, -2.0]), "std": torch.tensor([3.0, 0.5])}
    x = torch.randn(2, 4, 2 * 16 * 16)
    assert torch.allclose(renorm_psi(denorm_psi(x, cfg, norm), cfg, norm), x, atol=1e-5)


def test_guidance_schedules():
    from evaluation.run_qg_specwind_neural import guidance_schedule
    assert guidance_schedule(5.0) == 5.0
    decay = guidance_schedule(5.0, "linear-decay")
    assert decay(0.0) == 10.0 and decay(1.0) == 0.0 and decay(0.5) == 5.0
    with pytest.raises(ValueError):
        guidance_schedule(5.0, "cosine")


def test_sda_members_accept_a_scheduled_guidance_weight():
    from evaluation.run_qg_specwind_neural import guidance_schedule, sda_members
    model, batch = _sda_prior_and_batch()
    torch.manual_seed(3)
    x = sda_members(model, batch, n_members=2, guidance_weight=guidance_schedule(5.0, "linear-decay"),
                    n_steps=4)
    assert x.shape == (2, *batch.obs.shape) and torch.isfinite(x).all()


def test_published_sda_without_information_reproduces_the_prior_sample():
    from evaluation.run_qg_specwind_neural import PublishedSDA, published_sda_members
    model, batch = _sda_prior_and_batch()
    torch.manual_seed(1)
    guided = published_sda_members(model, batch, 1, 5, PublishedSDA(obs_sigma=1e8, n_corrector=0))
    torch.manual_seed(1)
    x = torch.randn(batch.obs.shape) * model.sigma_prior
    with torch.no_grad():
        for k in range(5):
            x = x + 0.2 * model(x, batch, torch.full((2,), k / 5))
    assert torch.allclose(guided[0], x, atol=1e-5)


def test_published_sda_guidance_pulls_towards_the_obs_and_corrector_runs():
    from evaluation.run_qg_specwind_neural import PublishedSDA, published_sda_members
    model, batch = _sda_prior_and_batch()
    misfit = {}
    for name, opts in (("off", PublishedSDA(obs_sigma=1e8, n_corrector=0)),
                       ("on", PublishedSDA(obs_sigma=0.1, gamma=0.1, n_corrector=1))):
        torch.manual_seed(2)
        x = published_sda_members(model, batch, 3, 5, opts)
        assert x.shape == (3, *batch.obs.shape) and torch.isfinite(x).all()
        misfit[name] = float(((x - batch.obs) ** 2)[:, batch.obs_mask].mean())
    assert misfit["on"] < misfit["off"]


def test_published_sda_refuses_a_warm_start():
    from evaluation.run_qg_specwind_neural import PublishedSDA, estimate_members
    model, batch = _sda_prior_and_batch()
    with pytest.raises(ValueError):
        estimate_members(model, "sda_prior_tchannels", batch, 1, sda_steps=3,
                         mean_estimate=torch.zeros(batch.obs.shape), published=PublishedSDA())


def test_fm_posterior_members_on_a_composed_prior_and_inflation():
    from evaluation.run_qg_specwind_neural import ComposedFlow, FMPosterior, fm_posterior_members
    from models.monai_unet_qg2d import MonaiSDAPriorQGChannelTime
    _, batch = _sda_prior_and_batch(days=6)
    torch.manual_seed(0)
    prior = MonaiSDAPriorQGChannelTime(ny=16, nx=16, T=4, hidden_channels=[8, 16, 32],
                                       norm_num_groups=4).eval()
    flow = ComposedFlow(prior, n_days=6, stride=1)
    start = torch.randn(batch.obs.shape)
    out = {}
    for infl in (1.0, 2.0):
        torch.manual_seed(5)
        out[infl] = fm_posterior_members(flow, batch, 3, 3, FMPosterior(inflation=infl, member_chunk=2),
                                         tau0=0.5, mean_estimate=start)
        assert out[infl].shape == (3, *batch.obs.shape) and torch.isfinite(out[infl]).all()
    assert torch.allclose(out[1.0].mean(0), out[2.0].mean(0), atol=1e-5)
    assert torch.allclose(out[2.0].std(0), 2.0 * out[1.0].std(0), atol=1e-5)


def test_fm_posterior_guidance_pulls_towards_the_obs():
    from evaluation.run_qg_specwind_neural import FMPosterior, fm_posterior_members
    model, batch = _sda_prior_and_batch()
    misfit = {}
    start = torch.zeros(batch.obs.shape)
    for w in (0.0, 1.0):
        torch.manual_seed(4)
        x = fm_posterior_members(model, batch, 3, 5, FMPosterior(guidance_weight=w, gamma=0.0,
                                                                 r_norm=1.0),
                                 tau0=0.5, mean_estimate=start)
        misfit[w] = float(((x - batch.obs) ** 2)[:, batch.obs_mask].mean())
    assert misfit[1.0] < misfit[0.0]


def test_perturbed_obs_starts_differ_per_member_and_only_through_the_obs():
    from data.qg_neural import QGBatch
    from evaluation.run_qg_specwind_neural import perturbed_obs_starts
    _, batch = _sda_prior_and_batch(days=4)
    b, t, d = batch.obs.shape
    qb = QGBatch(torch.zeros(b, t, d), batch.obs, batch.obs_mask, torch.zeros(b, t, 16, 16),
                 torch.zeros(b, t, d), torch.full((b,), 1.5e4))

    class ObsEcho(torch.nn.Module):
        T = 4

        def forward(self, bt):
            return bt.obs

    cfg = QGConfig(nx=16)
    starts = perturbed_obs_starts((ObsEcho(), "direct_unet_tchannels", None), qb, cfg, None, 3, 0.1, 2)
    assert starts.shape == (b, t, d, 3)
    assert not torch.allclose(starts[..., 0], starts[..., 1])
    unobserved = ~batch.obs_mask
    assert torch.equal(starts[..., 0][unobserved], batch.obs[unobserved])
