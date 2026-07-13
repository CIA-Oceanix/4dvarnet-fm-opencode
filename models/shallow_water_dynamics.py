"""Two-layer rotating shallow water equations on a periodic 2D domain.

State layout: [h1, u1, v1, h2, u2, v2] each of shape (Nx*Ny,)
Layer 1 = ocean (slow), Layer 2 = atmosphere (fast).
Periodic boundary conditions via torch.roll.
"""

import numpy as np
import torch

from models.dynamics import DynamicsBase


class ShallowWaterDynamics(DynamicsBase):
    """Two-layer rotating shallow water dynamics on a periodic 2-D grid.

    The state is ``[h1, u1, v1, h2, u2, v2]`` where each component is a
    flat tensor of length ``Nx * Ny``.  Layer 1 represents the ocean (slow)
    and layer 2 the atmosphere (fast).  Spatial derivatives are computed
    with central differences and periodic boundary conditions implemented
    through ``torch.roll``.

    Physical parameterisation includes:
    - Coriolis force (f-plane)
    - Pressure gradient from layer thickness
    - Inter-layer momentum coupling (linear)
    - Wind-stress forcing with meridional structure sin(2 pi y / L)
    - Linear Rayleigh friction to balance wind-stress energy input
    - Lateral Laplacian viscosity for numerical stability
    """

    param_names: list[str] = [
        "tau0", "f_cor", "g1", "g2", "coupling", "friction",
    ]
    param_dim: int = 6
    forcing_dim: int = 2

    def __init__(
        self,
        Nx: int = 64,
        Ny: int = 64,
        dt: float = 0.01,
        K: int = 5,
        tau0: float = 0.01,
        f_cor: float = 0.1,
        g1: float = 0.02,
        g2: float = 0.01,
        coupling: float = 0.05,
        friction: float = 0.1,
        viscosity: float = 0.001,
        land_mask_type: str = "none",
    ):
        super().__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.state_dim = 6 * Nx * Ny
        self.dt = dt
        self.K = K
        self.tau0 = tau0
        self.f_cor = f_cor
        self.g1 = g1
        self.g2 = g2
        self.coupling_coeff = coupling
        self.friction = friction
        self.viscosity = viscosity
        self.dx = 1.0
        self.dy = 1.0
        self.Lx = Nx * self.dx
        self.Ly = Ny * self.dy

        # Land mask ----------------------------------------------------------
        self.land_mask_type = land_mask_type
        self.land_mask = self._build_land_mask()

        # Static wind-stress spatial pattern: sin(2*pi*y / Ly) ---------------
        y_coords = torch.arange(Ny, dtype=torch.float32) * self.dy
        self.wind_pattern = (
            torch.sin(2.0 * torch.pi * y_coords / self.Ly)
            .unsqueeze(0)
            .expand(Nx, Ny)
            .reshape(-1)
            .contiguous()
        )

    # ------------------------------------------------------------------
    # Land mask
    # ------------------------------------------------------------------

    def _build_land_mask(self) -> torch.Tensor:
        """Return land mask *M* of shape ``(Nx*Ny,)`` with values in {0, 1}.

        ``M = 1`` denotes ocean (active) and ``M = 0`` denotes land (masked).
        Currently only ``"none"`` is fully implemented; ``"coastline"`` and
        ``"checkerboard"`` are placeholders for future development.
        """
        Nx, Ny = self.Nx, self.Ny
        if self.land_mask_type == "none":
            return torch.ones(Nx * Ny, dtype=torch.float32)
        if self.land_mask_type == "coastline":
            # Placeholder: uniform ocean for now
            return torch.ones(Nx * Ny, dtype=torch.float32)
        if self.land_mask_type == "checkerboard":
            # Placeholder: uniform ocean for now
            return torch.ones(Nx * Ny, dtype=torch.float32)
        raise ValueError(f"Unknown land_mask_type: {self.land_mask_type!r}")

    # ------------------------------------------------------------------
    # Spatial gradients  (central differences, periodic BC)
    # ------------------------------------------------------------------

    def _grad_x(self, f: torch.Tensor) -> torch.Tensor:
        """Central difference along the *x* (row) axis, periodic."""
        return (
            torch.roll(f, -self.Ny, dims=-1) - torch.roll(f, self.Ny, dims=-1)
        ) / (2.0 * self.dx)

    def _grad_y(self, f: torch.Tensor) -> torch.Tensor:
        """Central difference along the *y* (column) axis, periodic."""
        return (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) / (
            2.0 * self.dy
        )

    def _laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """5-point Laplacian on the periodic 2-D grid (flat layout).

        The flat index ``i*Ny + j`` corresponds to grid point ``(i, j)``.
        """
        return (
            torch.roll(f, -self.Ny, dims=-1)
            + torch.roll(f, self.Ny, dims=-1)
            + torch.roll(f, -1, dims=-1)
            + torch.roll(f, 1, dims=-1)
            - 4.0 * f
        ) / (self.dx * self.dy)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _clip_layer_thickness(self, state: torch.Tensor) -> torch.Tensor:
        """Clamp layer thicknesses *h1* and *h2* to ``>= 1e-6`` (in place)."""
        NxNy = self.Nx * self.Ny
        state = state.clone()
        state[..., :NxNy] = torch.clamp(state[..., :NxNy], min=1e-6)
        state[..., 3 * NxNy : 4 * NxNy] = torch.clamp(
            state[..., 3 * NxNy : 4 * NxNy], min=1e-6,
        )
        return state

    # ------------------------------------------------------------------
    # Right-hand side
    # ------------------------------------------------------------------

    def _derivative(
        self,
        state: torch.Tensor,
        forcing: torch.Tensor,
        tau0: float,
        f_cor: float,
        g1: float,
        g2: float,
        coupling_coeff: float,
        friction: float,
        viscosity: float,
    ) -> torch.Tensor:
        """Compute *dx/dt* for the two-layer rotating shallow water system.

        Equations (vector-invariant momentum, conservative continuity,
        with Rayleigh friction and lateral viscosity):

        .. math::

            \\partial_t h_l = -\\nabla \\cdot (h_l \\mathbf{u}_l)

            \\partial_t u_l = -u_l \\partial_x u_l - v_l \\partial_y u_l
                             + f\\,v_l - (g_l/h_l)\\,\\partial_x h_l
                             + \\tau_l / h_l + \\text{coupling}
                             - r\\,u_l + \\nu\\,\\nabla^2 u_l

            \\partial_t v_l = -u_l \\partial_x v_l - v_l \\partial_y v_l
                             - f\\,u_l - (g_l/h_l)\\,\\partial_y h_l
                             + \\text{coupling}
                             - r\\,v_l + \\nu\\,\\nabla^2 v_l
        """
        NxNy = self.Nx * self.Ny

        # Unpack state components -------------------------------------------
        h1 = state[..., 0:NxNy]
        u1 = state[..., NxNy : 2 * NxNy]
        v1 = state[..., 2 * NxNy : 3 * NxNy]
        h2 = state[..., 3 * NxNy : 4 * NxNy]
        u2 = state[..., 4 * NxNy : 5 * NxNy]
        v2 = state[..., 5 * NxNy : 6 * NxNy]

        # Clamp layer thicknesses to avoid division by zero / negatives
        eps_h = 1e-6
        h1c = torch.clamp(h1, min=eps_h)
        h2c = torch.clamp(h2, min=eps_h)

        # Wind stress:  tau(y,t) = tau0 * sin(2*pi*y/L) * (1 + eps(t))
        f1 = forcing[..., 0]
        f2 = forcing[..., 1]
        while f1.dim() < state.dim():
            f1 = f1.unsqueeze(-1)
        while f2.dim() < state.dim():
            f2 = f2.unsqueeze(-1)

        wind1 = tau0 * self.wind_pattern * (1.0 + f1)
        wind2 = tau0 * self.wind_pattern * (1.0 + f2)

        # Spatial gradients for momentum (vector-invariant form) ------------
        du1dx, du1dy = self._grad_x(u1), self._grad_y(u1)
        dv1dx, dv1dy = self._grad_x(v1), self._grad_y(v1)
        du2dx, du2dy = self._grad_x(u2), self._grad_y(u2)
        dv2dx, dv2dy = self._grad_x(v2), self._grad_y(v2)

        # Pressure-gradient: grad(h) for each layer
        dh1dx, dh1dy = self._grad_x(h1c), self._grad_y(h1c)
        dh2dx, dh2dy = self._grad_x(h2c), self._grad_y(h2c)

        # Laplacian for momentum diffusion ----------------------------------
        lap_u1 = self._laplacian(u1)
        lap_v1 = self._laplacian(v1)
        lap_u2 = self._laplacian(u2)
        lap_v2 = self._laplacian(v2)

        # Continuity (conservative form):  dh/dt = -div(h u) = -Dx(hu) - Dy(hv)
        # Using conservative form ensures exact mass conservation on periodic grids
        hu1 = h1c * u1
        hv1 = h1c * v1
        dh1dt = -(self._grad_x(hu1) + self._grad_y(hv1))

        hu2 = h2c * u2
        hv2 = h2c * v2
        dh2dt = -(self._grad_x(hu2) + self._grad_y(hv2))

        # Momentum -- layer 1 (ocean) ---------------------------------------
        du1dt = (
            -u1 * du1dx
            - v1 * du1dy
            + f_cor * v1
            - (g1 / h1c) * dh1dx
            + wind1 / h1c
            + coupling_coeff * (u2 - u1)
            - friction * u1
            + viscosity * lap_u1
        )
        dv1dt = (
            -u1 * dv1dx
            - v1 * dv1dy
            - f_cor * u1
            - (g1 / h1c) * dh1dy
            + coupling_coeff * (v2 - v1)
            - friction * v1
            + viscosity * lap_v1
        )

        # Momentum -- layer 2 (atmosphere) ----------------------------------
        du2dt = (
            -u2 * du2dx
            - v2 * du2dy
            + f_cor * v2
            - (g2 / h2c) * dh2dx
            + wind2 / h2c
            + coupling_coeff * (u1 - u2)
            - friction * u2
            + viscosity * lap_u2
        )
        dv2dt = (
            -u2 * dv2dx
            - v2 * dv2dy
            - f_cor * u2
            - (g2 / h2c) * dh2dy
            + coupling_coeff * (v1 - v2)
            - friction * v2
            + viscosity * lap_v2
        )

        # Apply land mask (zero out tendencies on land cells) ---------------
        M = self.land_mask
        while M.dim() < state.dim():
            M = M.unsqueeze(0)
        dh1dt = dh1dt * M
        du1dt = du1dt * M
        dv1dt = dv1dt * M
        dh2dt = dh2dt * M
        du2dt = du2dt * M
        dv2dt = dv2dt * M

        return torch.cat([dh1dt, du1dt, dv1dt, dh2dt, du2dt, dv2dt], dim=-1)

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

    def _rk4_step(
        self,
        state: torch.Tensor,
        forcing: torch.Tensor,
        tau0: float,
        f_cor: float,
        g1: float,
        g2: float,
        coupling_coeff: float,
        friction: float,
        viscosity: float,
        dt: float,
    ) -> torch.Tensor:
        """Positivity-preserving 4th-order Runge-Kutta step.

        Layer thicknesses *h1* and *h2* are clipped to ``>= 1e-6`` after
        every intermediate stage so that the pressure-gradient term *g/h*
        never encounters non-physical negative thickness.
        """
        args = (tau0, f_cor, g1, g2, coupling_coeff, friction, viscosity)
        k1 = self._derivative(state, forcing, *args)
        s2 = self._clip_layer_thickness(state + 0.5 * dt * k1)
        k2 = self._derivative(s2, forcing, *args)
        s3 = self._clip_layer_thickness(state + 0.5 * dt * k2)
        k3 = self._derivative(s3, forcing, *args)
        s4 = self._clip_layer_thickness(state + dt * k3)
        k4 = self._derivative(s4, forcing, *args)
        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self._clip_layer_thickness(new_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self, state: torch.Tensor, forcing: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Single forward step with default physical parameters from ``self``."""
        tau0 = kwargs.get("tau0", self.tau0)
        f_cor = kwargs.get("f_cor", self.f_cor)
        g1 = kwargs.get("g1", self.g1)
        g2 = kwargs.get("g2", self.g2)
        coupling_coeff = kwargs.get("coupling", self.coupling_coeff)
        friction = kwargs.get("friction", self.friction)
        viscosity = kwargs.get("viscosity", self.viscosity)
        dt = kwargs.get("dt", self.dt)
        return self._rk4_step(
            state, forcing, tau0, f_cor, g1, g2,
            coupling_coeff, friction, viscosity, dt,
        )

    def generate_full_trajectory(
        self,
        num_steps: int,
        seed: int = 42,
        device=None,
        spinup_steps: int = 500,
    ) -> tuple:
        """Generate a trajectory with temporally varying wind-stress forcing.

        Returns
        -------
        traj : Tensor ``(num_steps, state_dim)``
            The trajectory after spin-up.
        forcing : Tensor ``(num_steps, 2)``
            Temporal perturbation series ``(eps_layer1, eps_layer2)`` that was
            used during the trajectory window (post spin-up).
        """
        rng = np.random.RandomState(seed)
        total = num_steps + spinup_steps

        # Temporal perturbations  eps(t) for each layer  (weakly correlated)
        eps_raw = rng.randn(total, 2).astype(np.float32) * 0.1
        gamma_ar = 0.9
        eps = np.empty_like(eps_raw)
        eps[0] = eps_raw[0]
        for i in range(1, total):
            eps[i] = gamma_ar * eps[i - 1] + np.sqrt(1.0 - gamma_ar**2) * eps_raw[i]

        forcing_t = torch.tensor(eps, dtype=torch.float32)

        # Initial condition: small perturbation around h = 1, u = v = 0
        rng2 = np.random.RandomState(seed + 100)
        NxNy = self.Nx * self.Ny
        h1_0 = 1.0 + torch.tensor(rng2.randn(NxNy) * 0.01, dtype=torch.float32)
        u1_0 = torch.tensor(rng2.randn(NxNy) * 0.01, dtype=torch.float32)
        v1_0 = torch.tensor(rng2.randn(NxNy) * 0.01, dtype=torch.float32)
        h2_0 = 1.0 + torch.tensor(rng2.randn(NxNy) * 0.01, dtype=torch.float32)
        u2_0 = torch.tensor(rng2.randn(NxNy) * 0.01, dtype=torch.float32)
        v2_0 = torch.tensor(rng2.randn(NxNy) * 0.01, dtype=torch.float32)
        s0 = torch.cat([h1_0, u1_0, v1_0, h2_0, u2_0, v2_0])

        # Spin-up (discard transient)
        s = s0
        for i in range(spinup_steps):
            s = self._rk4_step(
                s, forcing_t[i], self.tau0, self.f_cor,
                self.g1, self.g2, self.coupling_coeff,
                self.friction, self.viscosity, self.dt,
            )

        # Collect trajectory
        traj_list = [s.clone()]
        for i in range(spinup_steps, spinup_steps + num_steps - 1):
            s = self._rk4_step(
                s, forcing_t[i], self.tau0, self.f_cor,
                self.g1, self.g2, self.coupling_coeff,
                self.friction, self.viscosity, self.dt,
            )
            traj_list.append(s.clone())
        traj = torch.stack(traj_list)

        forcing_out = forcing_t[spinup_steps : spinup_steps + num_steps]
        return traj, forcing_out

    def rollout_with_q(
        self,
        x0: torch.Tensor,
        q: torch.Tensor,
        forcing: torch.Tensor,
        steps: int,
        **kwargs,
    ) -> torch.Tensor:
        """Rollout with additive model-error injection at every step.

        Parameters
        ----------
        x0 : Tensor  (..., state_dim)
        q : Tensor  (..., steps, state_dim)
        forcing : Tensor  (..., steps, 2)
        steps : int
        """
        traj = [x0]
        for t in range(1, steps):
            next_s = self.step(traj[-1], forcing[..., t - 1], **kwargs)
            next_s = next_s + q[..., t, :]
            traj.append(next_s)
        return torch.stack(traj, dim=-2)
