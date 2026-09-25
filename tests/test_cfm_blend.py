from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.cfm_blend import TauBlendedFlow, parse_blend_schedule
from models.monai_unet_adapter import MonaiPredictStateCFM, MonaiVanillaCFM


def _batch(B: int = 3, T: int = 32, D: int = 4) -> SimpleNamespace:
    return SimpleNamespace(obs=torch.randn(B, T, D), forcing=torch.randn(B, T), params=None)


def _flows():
    torch.manual_seed(0)
    kw = dict(state_dim=4, hidden_channels=[16, 32], N_outer=4, dropout=0.0, param_dim=0,
              cond_extra_dim=0, num_res_blocks=1, norm_num_groups=8)
    return MonaiVanillaCFM(**kw).eval(), MonaiPredictStateCFM(**kw).eval()


def _sample(model, batch, seed: int = 7) -> torch.Tensor:
    torch.manual_seed(seed)
    with torch.no_grad():
        return model.sample(batch, N_outer=5, step_power=0.5)


@pytest.mark.parametrize("lam,which", [("const:0", 0), ("const:1", 1)])
def test_endpoint_schedules_reproduce_each_flow_exactly(lam, which):
    vc, psc, batch = *_flows(), _batch()
    blend = TauBlendedFlow(vc, psc, lam)
    assert torch.equal(_sample(blend, batch), _sample((vc, psc)[which], batch))


def test_blended_velocity_is_the_convex_combination():
    vc, psc, batch = *_flows(), _batch()
    blend = TauBlendedFlow(vc, psc, "const:0.3")
    x, tau = torch.randn(3, 32, 4), torch.full((3,), 0.4)
    with torch.no_grad():
        v_vc = vc.forward(x, batch, tau)
        v_psc = (psc.forward(x, batch, tau) - x) / 0.6
        assert torch.allclose(blend.velocity(x, batch, tau, 0.3), 0.3 * v_psc + 0.7 * v_vc, atol=1e-5)


def test_schedules():
    assert parse_blend_schedule("const:0.25")(0.9) == 0.25
    sw = parse_blend_schedule("switch:0.3")
    assert sw(0.29) == 1.0 and sw(0.3) == 0.0
    assert parse_blend_schedule("power:2")(0.5) == 0.25
    assert parse_blend_schedule("rpower:1")(0.5) == 0.5
    for bad in ("const:1.5", "linear:1", "power:-1"):
        with pytest.raises(ValueError):
            parse_blend_schedule(bad)


def test_same_family_blend_is_the_average_of_two_seeds():
    _, psc, batch = *_flows(), _batch()
    torch.manual_seed(1)
    psc2 = MonaiPredictStateCFM(state_dim=4, hidden_channels=[16, 32], N_outer=4, dropout=0.0, param_dim=0,
                                cond_extra_dim=0, num_res_blocks=1, norm_num_groups=8).eval()
    blend = TauBlendedFlow(psc, psc2, "const:0.5")
    x, tau = torch.randn(3, 32, 4), torch.full((3,), 0.4)
    with torch.no_grad():
        mean_d = 0.5 * (psc.forward(x, batch, tau) + psc2.forward(x, batch, tau))
        assert torch.allclose(blend.velocity(x, batch, tau, 0.5), (mean_d - x) / 0.6, atol=1e-5)


def test_rejects_non_flows_or_mismatched_flows():
    vc, psc = _flows()
    with pytest.raises(TypeError):
        TauBlendedFlow(vc, torch.nn.Linear(2, 2), "const:0.5")
    psc.sigma_prior = 1.0
    with pytest.raises(ValueError):
        TauBlendedFlow(vc, psc, "const:0.5")


def test_mean_velocity_flow_matches_the_two_flow_blend_and_averages_three():
    from models.cfm_blend import MeanVelocityFlow
    vc, psc, batch = *_flows(), _batch()
    two = MeanVelocityFlow([vc, psc])
    assert torch.allclose(_sample(two, batch), _sample(TauBlendedFlow(vc, psc, "const:0.5"), batch), atol=1e-5)
    torch.manual_seed(2)
    psc3 = MonaiPredictStateCFM(state_dim=4, hidden_channels=[16, 32], N_outer=4, dropout=0.0, param_dim=0,
                                cond_extra_dim=0, num_res_blocks=1, norm_num_groups=8).eval()
    three = MeanVelocityFlow([vc, psc, psc3])
    x, tau = torch.randn(3, 32, 4), torch.full((3,), 0.4)
    with torch.no_grad():
        expected = (vc.forward(x, batch, tau) + (psc.forward(x, batch, tau) - x) / 0.6
                    + (psc3.forward(x, batch, tau) - x) / 0.6) / 3
        assert torch.allclose(three.velocity(x, batch, tau), expected, atol=1e-5)
    with pytest.raises(TypeError):
        MeanVelocityFlow([vc])


def test_random_weight_schedule_is_seeded_and_on_the_simplex():
    from models.cfm_blend import random_weight_schedule
    w, knots = random_weight_schedule(3, seed=4)
    w2, knots2 = random_weight_schedule(3, seed=4)
    assert np.array_equal(knots, knots2) and knots.shape == (5, 3)
    for tau in (0.0, 0.1, 0.37, 0.5, 0.99, 1.0):
        assert abs(sum(w(tau)) - 1) < 1e-12 and min(w(tau)) >= 0 and w(tau) == w2(tau)
    assert np.allclose(w(0.25), knots[1])


def test_weighted_mean_velocity_flow_uses_the_tau_weights():
    from models.cfm_blend import MeanVelocityFlow
    vc, psc, batch = *_flows(), _batch()
    flow = MeanVelocityFlow([vc, psc], weights=lambda tau: [tau, 1 - tau])
    x, tau = torch.randn(3, 32, 4), torch.full((3,), 0.4)
    with torch.no_grad():
        expected = 0.4 * vc.forward(x, batch, tau) + 0.6 * (psc.forward(x, batch, tau) - x) / 0.6
        assert torch.allclose(flow.velocity(x, batch, tau), expected, atol=1e-5)
