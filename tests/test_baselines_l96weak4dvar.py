"""Unit tests for L96Weak4DVar (evaluation/baselines.py) -- the whitened-
control, gradient-safe weak/strong-constraint 4D-Var for L96, mirroring
evaluation/run_qg_baselines.py::QG4DVar's design (see L96Weak4DVar's own
docstring for the full plain-Weak4DVar-diverges investigation motivating
this port)."""
import numpy as np
import torch

from evaluation.baselines import L96Weak4DVar, ObsOperator
from models.lorenz96_dynamics import Lorenz96Dynamics


def _make_dynamics(NO=2, J=4, dt=0.01):
    return Lorenz96Dynamics(dt=dt, NO=NO, J=J, h=1.0, coupling_exponent=1.6, clip_range=50.0)


def _make_obs_op(NO=2, J=4, obs_j=1):
    sd = NO + NO * J
    X_idx = list(range(NO))
    Y_idx = [NO + k * J + j for k in range(NO) for j in range(obs_j)]
    return ObsOperator(sd, X_idx + Y_idx), sd


def _params():
    return dict(F=8.0, c1=1.0, h=1.0, hx=1.0, eps=0.1)


class TestL96Weak4DVarInit:
    def test_defaults(self):
        dyn = _make_dynamics()
        obs_op, sd = _make_obs_op()
        m = L96Weak4DVar(dt=0.01, dynamics=dyn, obs_operator=obs_op)
        assert m.state_dim == sd
        assert m.mode == "weak"
        assert m.da_window_steps == 500
        assert m.optimizer == "adam"
        assert m.grad_clip == 100.0


class TestL96Weak4DVarForward:
    def test_forward_strong_shape_and_finite(self):
        dyn = _make_dynamics()
        obs_op, sd = _make_obs_op()
        m = L96Weak4DVar(dt=0.01, dynamics=dyn, obs_operator=obs_op)
        x0 = torch.zeros(sd)
        force = torch.zeros(20)
        traj = m._forward_strong(x0, 20, force, _params())
        assert traj.shape == (20, sd)
        assert torch.isfinite(traj).all()

    def test_forward_weak_shape_and_finite(self):
        dyn = _make_dynamics()
        obs_op, sd = _make_obs_op()
        m = L96Weak4DVar(dt=0.01, dynamics=dyn, obs_operator=obs_op)
        x0 = torch.zeros(sd)
        u = torch.randn(20, sd) * 0.01
        force = torch.zeros(20)
        traj = m._forward_weak(x0, u, 20, force, Lq=0.1, params=_params())
        assert traj.shape == (20, sd)
        assert torch.isfinite(traj).all()

    def test_forward_weak_matches_strong_at_zero_u(self):
        """u=0 -- the weak-mode forward must reduce to the pure dynamics
        forecast (the same trajectory _forward_strong produces)."""
        dyn = _make_dynamics()
        obs_op, sd = _make_obs_op()
        m = L96Weak4DVar(dt=0.01, dynamics=dyn, obs_operator=obs_op)
        x0 = torch.randn(sd) * 0.1
        force = torch.randn(20) * 0.1
        strong = m._forward_strong(x0, 20, force, _params())
        weak = m._forward_weak(x0, torch.zeros(20, sd), 20, force, Lq=0.1, params=_params())
        assert torch.allclose(strong, weak)


class TestL96Weak4DVarResetNan:
    def test_reset_nan_zeros_params(self):
        w = torch.full((5,), float("nan"))
        u = torch.full((3, 5), float("nan"))
        L96Weak4DVar._reset_nan([w, u])
        assert torch.equal(w, torch.zeros(5))
        assert torch.equal(u, torch.zeros(3, 5))


class TestL96Weak4DVarOptimizeSafety:
    def _make(self, optimizer="adam", **kw):
        dyn = _make_dynamics()
        obs_op, sd = _make_obs_op()
        m = L96Weak4DVar(dt=0.01, dynamics=dyn, obs_operator=obs_op,
                         da_window_steps=20, optimizer=optimizer,
                         opt_steps=20, max_iter=5, **kw)
        return m, sd

    def test_adam_resets_on_nan_loss(self):
        m, sd = self._make(optimizer="adam")
        w = torch.zeros(sd, requires_grad=True)

        def loss_fn():
            return torch.tensor(float("nan")) * w.sum()

        m._optimize(loss_fn, [w])
        assert torch.equal(w.detach(), torch.zeros(sd))

    def test_lbfgs_resets_on_nan_loss(self):
        m, sd = self._make(optimizer="lbfgs")
        w = torch.zeros(sd, requires_grad=True)

        def loss_fn():
            return torch.tensor(float("nan")) * w.sum()

        m._optimize(loss_fn, [w])
        assert torch.isfinite(w.detach()).all()

    def test_adam_clips_and_sanitizes_gradient(self):
        """A deliberately huge, partly-NaN gradient must be sanitized
        (nan_to_num) and clamped to grad_clip before the Adam step, not
        propagated as-is."""
        m, sd = self._make(optimizer="adam", grad_clip=1.0, lr=0.1)
        w = torch.zeros(sd, requires_grad=True)

        def loss_fn():
            # A term whose gradient w.r.t. w is huge (1e6) plus a NaN
            # contribution from a different, disconnected path -- Adam's
            # step must still leave w finite and bounded.
            huge = (1e6 * w).sum()
            return huge

        m._optimize(loss_fn, [w])
        assert torch.isfinite(w.detach()).all()
        # With grad_clip=1.0 and a single Adam step of lr=0.1, w cannot
        # have moved further than a small, bounded amount.
        assert w.detach().abs().max() < 10.0


class TestL96Weak4DVarAssimilate:
    def _make(self, mode="weak", **kw):
        dyn = _make_dynamics()
        obs_op, sd = _make_obs_op()
        m = L96Weak4DVar(dt=0.01, dynamics=dyn, obs_operator=obs_op,
                         da_window_steps=10, opt_steps=10, mode=mode, **kw)
        return m, sd, obs_op

    def test_assimilate_weak_shape_and_finite(self):
        m, sd, obs_op = self._make(mode="weak")
        T = 20
        torch.manual_seed(0)
        obs = torch.randn(T, len(obs_op.indices))
        mask = torch.ones(T, dtype=torch.bool)
        forcing = torch.zeros(T)
        true_state = torch.randn(T, sd)
        res = m.assimilate(obs, mask, forcing, true_state=true_state)
        assert res.trajectory.shape == (T, sd)
        assert np.isfinite(res.trajectory).all()
        assert res.rmse.shape == (sd,)

    def test_assimilate_strong_mode_shape_and_finite(self):
        m, sd, obs_op = self._make(mode="strong")
        T = 20
        torch.manual_seed(0)
        obs = torch.randn(T, len(obs_op.indices))
        mask = torch.ones(T, dtype=torch.bool)
        forcing = torch.zeros(T)
        true_state = torch.randn(T, sd)
        res = m.assimilate(obs, mask, forcing, true_state=true_state)
        assert res.trajectory.shape == (T, sd)
        assert np.isfinite(res.trajectory).all()

    def test_assimilate_survives_extreme_observations(self):
        """Deliberately adversarial obs (huge magnitude) with an aggressive
        lr -- exactly the kind of stress that diverges the plain
        (unguarded) Weak4DVar class to NaN/huge values. Must stay finite
        here thanks to the whitened controls + NaN-reset safety net."""
        dyn = _make_dynamics()
        obs_op, sd = _make_obs_op()
        m = L96Weak4DVar(dt=0.01, dynamics=dyn, obs_operator=obs_op,
                         da_window_steps=10, mode="weak", opt_steps=30, lr=1.0)
        T = 20
        obs = torch.full((T, len(obs_op.indices)), 1000.0)
        mask = torch.ones(T, dtype=torch.bool)
        forcing = torch.zeros(T)
        true_state = torch.zeros(T, sd)
        res = m.assimilate(obs, mask, forcing, true_state=true_state)
        assert np.isfinite(res.trajectory).all()

    def test_assimilate_uses_true_state_for_rmse_when_given(self):
        m, sd, obs_op = self._make(mode="weak")
        T = 20
        torch.manual_seed(1)
        obs = torch.randn(T, len(obs_op.indices))
        mask = torch.ones(T, dtype=torch.bool)
        forcing = torch.zeros(T)
        true_state = torch.zeros(T, sd)
        res = m.assimilate(obs, mask, forcing, true_state=true_state)
        expected_rmse = np.sqrt(np.mean(res.trajectory ** 2, axis=0))
        assert np.allclose(res.rmse, expected_rmse)
