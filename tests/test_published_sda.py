import pytest
import torch

from evaluation.published_sda import nfe_per_member, published_sda_sample


class _ToyPrior(torch.nn.Module):
    sigma_prior = 0.5

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, batch, tau: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x)) * (1.0 + tau.view(-1, 1, 1)) - x


def _obs():
    torch.manual_seed(1)
    obs = torch.randn(2, 6, 4)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, ::2] = True
    return obs, mask


def _sample(obs, mask, var, seed=3, **kw):
    torch.manual_seed(seed)
    return published_sda_sample(_ToyPrior(), None, obs, mask, var, 2, 5, **kw)


def test_per_channel_variance_equal_to_a_scalar_matches_the_scalar_path():
    obs, mask = _obs()
    a = _sample(obs, mask.unsqueeze(-1), 0.2)
    b = _sample(obs, mask.unsqueeze(-1), torch.full((4,), 0.2))
    assert a.shape == (2, *obs.shape) and torch.allclose(a, b, atol=1e-6)


def test_nan_obs_inside_an_observed_time_are_ignored_not_read_as_zero():
    obs, mask = _obs()
    holed = obs.clone()
    holed[:, 0, 1] = float("nan")
    cell = mask.unsqueeze(-1).expand_as(obs).clone()
    cell[:, 0, 1] = False
    zeroed = torch.where(cell, obs, torch.zeros_like(obs))
    a = _sample(holed, mask.unsqueeze(-1), 0.2)
    b = _sample(zeroed, cell, 0.2)
    c = _sample(torch.nan_to_num(holed), mask.unsqueeze(-1), 0.2)
    assert torch.isfinite(a).all() and torch.allclose(a, b, atol=1e-6)
    assert not torch.allclose(a, c, atol=1e-4)


def test_guidance_pulls_towards_the_obs():
    obs, mask = _obs()
    m = mask.unsqueeze(-1).expand_as(obs)
    off = _sample(obs, m, 1e8, n_corrector=0)
    on = _sample(obs, m, 0.05, gamma=0.01, n_corrector=1)
    assert ((on - obs) ** 2)[:, m].mean() < ((off - obs) ** 2)[:, m].mean()


@pytest.mark.parametrize("steps,ncorr,nfe", [(100, 1, 199), (10, 0, 10), (1, 2, 1)])
def test_nfe_per_member(steps, ncorr, nfe):
    assert nfe_per_member(steps, ncorr) == nfe
