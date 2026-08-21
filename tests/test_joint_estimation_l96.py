import numpy as np
import pytest

from evaluation.baselines import (
    JointEnKFL96, JointETKFL96, JointStrong4DVarL96,
)
from evaluation.metrics import param_rmse


@pytest.fixture
def l96_ctx(device):
    import torch
    from models.lorenz96_dynamics import Lorenz96Dynamics
    from evaluation.baselines import ObsOperator
    torch.manual_seed(0)
    dyn = Lorenz96Dynamics(dt=0.001, coupling_exponent=1.6)
    op = ObsOperator(40, list(range(24)))
    T = 50
    obs = torch.randn(T, 24, device=device)
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    mask[::5] = True
    force = torch.zeros(T, device=device)
    truth = torch.randn(T, 40, device=device)
    return {
        "dyn": dyn, "op": op, "obs": obs, "mask": mask,
        "force": force, "truth": truth, "device": device,
    }


class TestJointEnKFL96:
    def test_shapes(self, l96_ctx, device):
        m = JointEnKFL96(N_ensemble=6, dt=0.001, device=device,
                         dynamics=l96_ctx["dyn"], obs_operator=l96_ctx["op"], NO=8, J=4)
        r = m.assimilate(l96_ctx["obs"], l96_ctx["mask"], l96_ctx["force"],
                         true_state=l96_ctx["truth"], F=8.0, c1=1.0,
                         hx=1.0, eps=0.1, fast_weights=[1, 1, 0.1, 0.1])
        assert r.trajectory.shape == (50, 40)
        assert r.params.shape == (50, 8)
        assert r.rmse.shape == (40,)
        assert np.all(np.isfinite(r.params))
        assert np.all(r.params >= 0)

    def test_param_rmse_finite(self, l96_ctx, device):
        m = JointEnKFL96(N_ensemble=6, dt=0.001, device=device,
                         dynamics=l96_ctx["dyn"], obs_operator=l96_ctx["op"], NO=8, J=4)
        r = m.assimilate(l96_ctx["obs"], l96_ctx["mask"], l96_ctx["force"],
                         true_state=l96_ctx["truth"], F=8.0, c1=1.0,
                         hx=1.0, eps=0.1, fast_weights=[1, 1, 0.1, 0.1])
        true_params = np.tile([8.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 0.1], (50, 1))
        prmse = param_rmse(r.params.reshape(-1, 8), true_params.reshape(-1, 8))
        assert prmse.shape == (8,)
        assert np.all(np.isfinite(prmse))


class TestJointETKFL96:
    def test_shapes(self, l96_ctx, device):
        m = JointETKFL96(N_ensemble=6, dt=0.001, device=device,
                         dynamics=l96_ctx["dyn"], obs_operator=l96_ctx["op"], NO=8, J=4)
        r = m.assimilate(l96_ctx["obs"], l96_ctx["mask"], l96_ctx["force"],
                         true_state=l96_ctx["truth"], F=8.0, c1=1.0,
                         hx=1.0, eps=0.1, fast_weights=[1, 1, 0.1, 0.1])
        assert r.trajectory.shape == (50, 40)
        assert r.params.shape == (50, 8)
        assert r.rmse.shape == (40,)
        assert np.all(np.isfinite(r.params))


class TestJointStrong4DVarL96:
    def test_shapes_and_obs_projection(self, l96_ctx, device):
        m = JointStrong4DVarL96(dt=0.001, da_window_steps=25, device=device,
                                dynamics=l96_ctx["dyn"], obs_operator=l96_ctx["op"],
                                max_iter=1, J=4)
        r = m.assimilate(l96_ctx["obs"], l96_ctx["mask"], l96_ctx["force"],
                         true_state=l96_ctx["truth"], F=8.0, c1=1.0,
                         hx=1.0, eps=0.1, fast_weights=[1, 1, 0.1, 0.1])
        assert r.trajectory.shape == (50, 40)
        assert r.params.shape == (50, 8)
        assert r.rmse.shape == (40,)
        assert np.all(np.isfinite(r.trajectory))
        assert np.all(np.isfinite(r.params))
