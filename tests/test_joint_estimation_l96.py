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

    def test_es_finite_shape(self, l96_ctx, device):
        m = JointETKFL96(N_ensemble=6, dt=0.001, device=device,
                         dynamics=l96_ctx["dyn"], obs_operator=l96_ctx["op"], NO=8, J=4)
        r = m.assimilate(l96_ctx["obs"], l96_ctx["mask"], l96_ctx["force"],
                         true_state=l96_ctx["truth"], F=8.0, c1=1.0,
                         hx=1.0, eps=0.1, fast_weights=[1, 1, 0.1, 0.1])
        assert r.es is not None
        assert r.es.shape == (40,)
        assert np.all(np.isfinite(r.es))

    def test_s1_shape_w3w4_default(self, device):
        import torch
        from models.lorenz96_dynamics import Lorenz96Dynamics
        from evaluation.baselines import ObsOperator
        torch.manual_seed(0)
        J = 2
        dyn = Lorenz96Dynamics(dt=0.001, NO=8, J=J, h=1.0, hx=1.0, eps=0.1,
                               coupling_exponent=1.0)
        op = ObsOperator(24, list(range(24)))
        m = JointETKFL96(N_ensemble=6, dt=0.001, device=device,
                         dynamics=dyn, obs_operator=op, NO=8, J=J)
        T = 60
        obs = torch.randn(T, 24, device=device)
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        mask[::5] = True
        force = torch.zeros(T, device=device)
        truth = torch.randn(T, 24, device=device)
        r = m.assimilate(obs, mask, force, true_state=truth, F=8.0, c1=1.0,
                         h=1.0, hx=1.0, eps=0.1, fast_weights=[1.0, 1.0, 0.1, 0.1])
        assert r.trajectory.shape == (T, 24)
        assert r.params.shape == (T, 8)
        assert np.all(np.isfinite(r.params))
        assert r.es is not None and r.es.shape == (24,)
        np.testing.assert_allclose(r.params[:, 6], 0.1, rtol=1e-9)
        np.testing.assert_allclose(r.params[:, 7], 0.1, rtol=1e-9)

    def _seq_vs_batch(self, device, J):
        import torch
        from models.lorenz96_dynamics import Lorenz96Dynamics
        from evaluation.baselines import ObsOperator
        torch.manual_seed(7)
        dyn = Lorenz96Dynamics(dt=0.001, NO=8, J=J, h=1.0, hx=1.0, eps=0.1,
                               coupling_exponent=1.0)
        op = ObsOperator(24, list(range(24)))
        m = JointETKFL96(N_ensemble=6, dt=0.001, device=device,
                         dynamics=dyn, obs_operator=op, NO=8, J=J)
        T = 40
        obs = torch.randn(T, 24, device=device)
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        mask[::5] = True
        force = torch.zeros(T, device=device)
        truth = torch.randn(T, 24, device=device)

        def run_seq():
            torch.manual_seed(11)
            return m.assimilate(obs, mask, force, true_state=truth,
                                F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1,
                                fast_weights=[1.0, 1.0, 0.1, 0.1])

        def run_batch():
            torch.manual_seed(11)
            F = torch.full((1,), 8.0, device=device)
            c1 = torch.full((1,), 1.0, device=device)
            hx = torch.full((1,), 1.0, device=device)
            eps = torch.full((1,), 0.1, device=device)
            fw = torch.tensor([[1.0, 1.0]], device=device)
            return m.assimilate_batch(
                obs.unsqueeze(0), mask.unsqueeze(0), force.unsqueeze(0),
                true_state=truth.unsqueeze(0), F=F, c1=c1, hx=hx, eps=eps,
                fast_weights=fw)[0]

        r1 = run_seq()
        r2 = run_batch()
        np.testing.assert_allclose(r2.trajectory, r1.trajectory, atol=1e-6)
        np.testing.assert_allclose(r2.params, r1.params, atol=1e-6)
        np.testing.assert_allclose(r2.es, r1.es, atol=1e-6)

    def test_batch_matches_sequential_s1(self, device):
        self._seq_vs_batch(device, J=2)

    def test_batch_matches_sequential_s0(self, device):
        import torch
        from models.lorenz96_dynamics import Lorenz96Dynamics
        from evaluation.baselines import ObsOperator
        from evaluation.run_l96 import make_obs_j_indices
        torch.manual_seed(7)
        ovi = make_obs_j_indices(8, 4, 2)
        dyn = Lorenz96Dynamics(dt=0.001, coupling_exponent=1.6)
        op = ObsOperator(40, ovi)
        m = JointETKFL96(N_ensemble=6, dt=0.001, device=device,
                         dynamics=dyn, obs_operator=op, NO=8, J=4)
        T = 40
        obs = torch.randn(T, 24, device=device)
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        mask[::5] = True
        force = torch.zeros(T, device=device)
        truth = torch.randn(T, 40, device=device)

        def run_seq():
            torch.manual_seed(11)
            return m.assimilate(obs, mask, force, true_state=truth,
                                F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1,
                                fast_weights=[1.0, 1.0, 0.1, 0.1])

        def run_batch():
            torch.manual_seed(11)
            F = torch.full((1,), 8.0, device=device)
            c1 = torch.full((1,), 1.0, device=device)
            hx = torch.full((1,), 1.0, device=device)
            eps = torch.full((1,), 0.1, device=device)
            fw = torch.tensor([[1.0, 1.0, 0.1, 0.1]], device=device)
            return m.assimilate_batch(
                obs.unsqueeze(0), mask.unsqueeze(0), force.unsqueeze(0),
                true_state=truth.unsqueeze(0), F=F, c1=c1, hx=hx, eps=eps,
                fast_weights=fw)[0]

        r1 = run_seq()
        r2 = run_batch()
        np.testing.assert_allclose(r2.trajectory, r1.trajectory, atol=1e-6)
        np.testing.assert_allclose(r2.params, r1.params, atol=1e-6)
        np.testing.assert_allclose(r2.es, r1.es, atol=1e-6)

    def test_etkf_ridge_applied(self, device):
        import torch
        from models.lorenz96_dynamics import Lorenz96Dynamics
        from evaluation.baselines import ObsOperator
        torch.manual_seed(0)
        dyn = Lorenz96Dynamics(dt=0.001, coupling_exponent=1.6)
        op = ObsOperator(40, list(range(24)))
        obs = torch.randn(50, 24, device=device)
        mask = torch.zeros(50, dtype=torch.bool, device=device)
        mask[::5] = True
        force = torch.zeros(50, device=device)
        truth = torch.randn(50, 40, device=device)

        def run(ridge):
            m = JointETKFL96(N_ensemble=6, dt=0.001, device=device,
                             dynamics=dyn, obs_operator=op, NO=8, J=4,
                             etkf_ridge=ridge)
            return m.assimilate(obs, mask, force, true_state=truth,
                                F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1,
                                fast_weights=[1, 1, 0.1, 0.1]).trajectory

        t0 = run(0.0)
        t1 = run(10.0)
        assert np.mean(np.abs(t0 - t1)) > 1e-3

    def test_nan_safety(self, device, monkeypatch, l96_ctx):
        import torch
        from models.lorenz96_dynamics import Lorenz96Dynamics
        from evaluation.baselines import ObsOperator
        torch.manual_seed(0)
        dyn = Lorenz96Dynamics(dt=0.001, coupling_exponent=1.6)
        op = ObsOperator(40, list(range(24)))
        m = JointETKFL96(N_ensemble=6, dt=0.001, device=device,
                         dynamics=dyn, obs_operator=op, NO=8, J=4)

        def bad_analysis(self, ensemble, y_t, idx, H):
            mu = torch.mean(ensemble, dim=0)
            blob = torch.full_like(ensemble, float("nan"))
            infl = self.inflation
            return mu + infl * (blob - mu)

        monkeypatch.setattr(m, "_analysis", bad_analysis.__get__(m, type(m)))
        r = m.assimilate(l96_ctx["obs"], l96_ctx["mask"], l96_ctx["force"],
                         true_state=l96_ctx["truth"], F=8.0, c1=1.0, h=1.0,
                         hx=1.0, eps=0.1, fast_weights=[1, 1, 0.1, 0.1])
        assert np.all(np.isfinite(r.trajectory))
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
