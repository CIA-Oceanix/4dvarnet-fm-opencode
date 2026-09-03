import torch

from models.dynamics import DynamicsBase


class _PsiMixin:
    """Shared forward PV operator (psi -> q) + psi<->q conversion helpers.

    The QG dynamics natively evolve PV anomaly `q`; the streamfunction `psi`
    is related to it by a linear invertible spectral operator. This mixin
    exposes `forward_pv` (the spectral matrix that maps psi-hat -> q-hat,
    consistent with the wrapped q-model's `_invert`) and `psi_to_q`, so a
    wrapper can hold `psi` as its state and convert to `q` only around the
    q-space RK4 integration. The K2 == 0 (mean) mode is zeroed identically to
    `_invert`, making `forward_pv(_invert(qh)) == qh` on mean-removed fields.
    """
    @property
    def ny(self) -> int:
        return self.inner.ny

    @property
    def nx(self) -> int:
        return self.inner.nx

    def psi_to_q(self, state: torch.Tensor) -> torch.Tensor:
        """Convert a physical psi-state (flattened layer-major) to q-state."""
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        psi = self.inner._grid(state_b)
        psh = torch.fft.rfft2(psi, dim=(-2, -1))
        qh = self.forward_pv(psh)
        q = torch.fft.irfft2(qh, s=(self.ny, self.nx), dim=(-2, -1))
        out = self.inner._flatten(q)
        return out.squeeze(0) if single else out

    def q_to_psi(self, state: torch.Tensor) -> torch.Tensor:
        """Convert a physical q-state (flattened layer-major) to psi-state."""
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self.inner._grid(state_b)
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        ph = self.inner._invert(qh)
        psi = torch.fft.irfft2(ph, s=(self.ny, self.nx), dim=(-2, -1))
        out = self.inner._flatten(psi)
        return out.squeeze(0) if single else out

    def streamfunctions(self, state: torch.Tensor) -> torch.Tensor:
        """Return the grid-shaped psi (identity for a psi-state wrapper).

        `_field_layer_metrics` calls `inner.streamfunctions(traj)` to compute
        psi diagnostics; for a psi-state the trajectory already is psi, so we
        reshape to (..., nlayer, ny, nx).
        """
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        out = self.inner._grid(state_b)
        return out.squeeze(0) if single else out


class QGPsiDynamics(_PsiMixin, DynamicsBase):
    """Two-layer QG with the streamfunction `psi` as the state variable.

    Wraps a `QGDynamics` (which integrates PV `q`) but stores/returns `psi`
    so the observation operator over streamfunctions reduces to a trivial
    index lookup. The q-space physics is bit-identical to the wrapped model:
    at each step the wrapper converts psi->q (linear spectral `forward_pv`),
    applies the exact `_rk4_step` (including the pyqg spectral filter and the
    `clip_range` q-clamp), then converts q->psi. The only difference is the
    linear round-trip per step (~1e-12 float roundoff), so free forecasts and
    DA analyses match the q-state scheme up to system chaoticity.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.state_dim = inner.state_dim
        self.param_dim = inner.param_dim
        self.param_names = inner.param_names
        self.forcing_dim = inner.forcing_dim
        self.dtype = inner.dtype

        K2 = inner.K2.to(torch.float64)
        F1 = float(inner.F1)
        F2 = float(inner.F2)
        zero = K2 == 0
        b11 = -(K2 + F1)
        b12 = torch.full_like(K2, F1)
        b21 = torch.full_like(K2, F2)
        b22 = -(K2 + F2)
        for m in (b11, b12, b21, b22):
            m[zero] = 0.0
        self.register_buffer("b11", b11.to(inner.dtype))
        self.register_buffer("b12", b12.to(inner.dtype))
        self.register_buffer("b21", b21.to(inner.dtype))
        self.register_buffer("b22", b22.to(inner.dtype))

    @property
    def device(self) -> torch.device:
        return self.inner.device

    def to(self, device=None, dtype=None):
        inner = self.inner.to(device=device)
        self.inner = inner
        return super().to(device=device)

    def forward_pv(self, ph: torch.Tensor) -> torch.Tensor:
        qh1 = self.b11.to(ph.dtype) * ph[..., 0, :, :] \
            + self.b12.to(ph.dtype) * ph[..., 1, :, :]
        qh2 = self.b21.to(ph.dtype) * ph[..., 0, :, :] \
            + self.b22.to(ph.dtype) * ph[..., 1, :, :]
        return torch.stack([qh1, qh2], dim=-3)

    def _rk4_q(self, qh, dt, U1, U2, beta, rek, wind_state_t):
        return self.inner._rk4_step(qh, dt, U1, U2, beta, rek, wind_state_t)

    def step(self, state, forcing=None, wind_state_t=None, **kwargs):
        U1 = kwargs.get("U1", self.inner.U1)
        U2 = kwargs.get("U2", self.inner.U2)
        beta = kwargs.get("beta", self.inner.beta)
        rek = kwargs.get("rek", self.inner.rek)
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        psi = self.inner._grid(state_b)
        psh = torch.fft.rfft2(psi, dim=(-2, -1))
        qh = self.forward_pv(psh)
        qh_next = self._rk4_q(qh, self.inner.dt, U1, U2, beta, rek, wind_state_t)
        psh_next = self.inner._invert(qh_next)
        psi_next = torch.fft.irfft2(psh_next, s=(self.ny, self.nx), dim=(-2, -1))
        out = self.inner._flatten(psi_next)
        return out.squeeze(0) if single else out

    def rollout_trajectory(self, state, steps, wind_state=None, **kwargs):
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        psi = self.inner._grid(state_b)
        psh = torch.fft.rfft2(psi, dim=(-2, -1))
        qh = self.forward_pv(psh)
        U1 = kwargs.get("U1", self.inner.U1)
        U2 = kwargs.get("U2", self.inner.U2)
        beta = kwargs.get("beta", self.inner.beta)
        rek = kwargs.get("rek", self.inner.rek)
        wstate = wind_state if wind_state is not None \
            else torch.zeros(steps, 3, device=psh.device, dtype=self.dtype)
        traj = [torch.fft.irfft2(psh, s=(self.ny, self.nx), dim=(-2, -1))]
        for k in range(steps):
            qh = self._rk4_q(qh, self.inner.dt, U1, U2, beta, rek, wstate[k])
            psh = self.inner._invert(qh)
            traj.append(torch.fft.irfft2(psh, s=(self.ny, self.nx), dim=(-2, -1)))
        out = self.inner._flatten(torch.stack(traj, dim=0))
        return out.squeeze(1) if single else out


class QG1LPsiDynamics(_PsiMixin, DynamicsBase):
    """Reduced-gravity single-layer QG with `psi` as the state variable.

    Mirrors `QGPsiDynamics` for the 1-layer `QG1LDynamics`: forward PV is
    `q_hat = -(K2 + rd^-2) * psi_hat` (the spectral inverse of the 1-layer
    `_invert`, zeroed at K2 == 0), RK4 in q-space, then back to psi.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.state_dim = inner.state_dim
        self.param_dim = inner.param_dim
        self.param_names = inner.param_names
        self.forcing_dim = inner.forcing_dim
        self.dtype = inner.dtype

        K2 = inner.K2.to(torch.float64)
        zero = K2 == 0
        ainv = -(K2 + float(inner.rd) ** -2)
        ainv[zero] = 0.0
        self.register_buffer("ainv", ainv.to(inner.dtype))

    @property
    def device(self) -> torch.device:
        return self.inner.device

    def to(self, device=None, dtype=None):
        inner = self.inner.to(device=device)
        self.inner = inner
        return super().to(device=device)

    def forward_pv(self, ph: torch.Tensor) -> torch.Tensor:
        return self.ainv.to(ph.dtype) * ph

    def _rk4_q(self, qh, dt, U1, beta, rek, wind_state_t):
        return self.inner._rk4_step(qh, dt, U1, beta, rek, wind_state_t)

    def step(self, state, forcing=None, wind_state_t=None, **kwargs):
        U1 = kwargs.get("U1", self.inner.U1)
        beta = kwargs.get("beta", self.inner.beta)
        rek = kwargs.get("rek", self.inner.rek)
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        psi = self.inner._grid(state_b)
        psh = torch.fft.rfft2(psi, dim=(-2, -1))
        qh = self.forward_pv(psh)
        qh_next = self._rk4_q(qh, self.inner.dt, U1, beta, rek, wind_state_t)
        psh_next = self.inner._invert(qh_next)
        psi_next = torch.fft.irfft2(psh_next, s=(self.ny, self.nx), dim=(-2, -1))
        out = self.inner._flatten(psi_next)
        return out.squeeze(0) if single else out

    def rollout_trajectory(self, state, steps, wind_state=None, **kwargs):
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        psi = self.inner._grid(state_b)
        psh = torch.fft.rfft2(psi, dim=(-2, -1))
        qh = self.forward_pv(psh)
        U1 = kwargs.get("U1", self.inner.U1)
        beta = kwargs.get("beta", self.inner.beta)
        rek = kwargs.get("rek", self.inner.rek)
        wstate = wind_state if wind_state is not None \
            else torch.zeros(steps, 3, device=psh.device, dtype=self.dtype)
        traj = [torch.fft.irfft2(psh, s=(self.ny, self.nx), dim=(-2, -1))]
        for k in range(steps):
            qh = self._rk4_q(qh, self.inner.dt, U1, beta, rek, wstate[k])
            psh = self.inner._invert(qh)
            traj.append(torch.fft.irfft2(psh, s=(self.ny, self.nx), dim=(-2, -1)))
        out = self.inner._flatten(torch.stack(traj, dim=0))
        return out.squeeze(1) if single else out


def wrap_psi(inner, da_model: str):
    """Wrap a q-dynamics instance into its psi-state counterpart."""
    if da_model == "qg1l":
        return QG1LPsiDynamics(inner)
    return QGPsiDynamics(inner)