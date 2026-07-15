"""Two-layer rotating shallow water equations on a periodic 2D domain.

State layout: [h1, u1, v1, h2, u2, v2] each of shape (Nx*Ny,)
Layer 1 = ocean (slow), Layer 2 = atmosphere (fast).
Periodic boundary conditions via torch.roll.
"""

import math
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
        dt: float = 0.1,
        K: int = 5,
        tau0: float = 0.0,
        f_cor: float = 1.0,
        g1: float = 1.0,
        g2: float = 4.0,
        coupling: float = 0.01,
        friction: float = 0.0,
        viscosity: float = 0.0001,
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

        # Bickley-jet parameters -------------------------------------------------
        y_coords = torch.arange(Ny, dtype=torch.float32) * self.dy
        self.y_center = y_coords - self.Ly / 2.0  # centered at domain mid

        # Static wind-stress spatial pattern: sin(2*pi*y / Ly) ---------------
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
    # Bickley-jet initial condition
    # ------------------------------------------------------------------

    def _init_bickley_jet(
        self,
        seed: int = 42,
        U: float = 1.0,
        U2: float = 0.6,
        L_jet_frac: float = 0.15,
        epsilon: float = 1e-4,
        H_ref: float = 1.0,
        perturbation_mode: str = "sinusoidal",
    ) -> torch.Tensor:
        """Geostrophically balanced Bickley jet initial condition.

        Both layers contain a ``sech²(y / L_jet)`` jet profile centred
        at the domain mid-point.  The free-surface height is obtained
        from geostrophic balance for each layer *l*:

            ∂_y h_l = -(f / g_l) u_l

        so that

            h_l(y) = H_ref - (f / g_l) * U_l * L_jet * tanh(y / L_jet)

        A small perturbation *epsilon* (sinusoidal or random) is added to
        seed the barotropic/baroclinic instability.

        Parameters
        ----------
        seed : int
            RNG seed for the perturbation.
        U : float
            Maximum jet velocity for layer 1 (ocean).
        U2 : float
            Maximum jet velocity for layer 2 (atmosphere).
        L_jet_frac : float
            Jet half-width as a fraction of domain width *Ly*.
        epsilon : float
            Amplitude of the random perturbation.
        H_ref : float
            Reference layer thickness.

        Returns
        -------
        state : Tensor ``(state_dim,)``
        """
        Nx, Ny = self.Nx, self.Ny
        NxNy = Nx * Ny
        y_center = self.y_center  # (Ny,) centred at 0
        L_jet = self.Ly * L_jet_frac

        # y_center needs to be expanded to (Nx*Ny,) for flat state layout
        y = y_center.unsqueeze(0).expand(Nx, -1).reshape(-1).clone()

        # Layer 1 (ocean) jet
        u1 = U * (1.0 / torch.cosh(y / L_jet)) ** 2
        v1 = torch.zeros(NxNy)
        h1 = H_ref - (self.f_cor / self.g1) * U * L_jet * torch.tanh(y / L_jet)

        # Layer 2 (atmosphere) jet
        u2 = U2 * (1.0 / torch.cosh(y / L_jet)) ** 2
        v2 = torch.zeros(NxNy)
        h2 = H_ref - (self.f_cor / self.g2) * U2 * L_jet * torch.tanh(y / L_jet)

        # Small perturbation to seed instability
        if perturbation_mode == "sinusoidal":
            # Product of cosines: cos(kx*x) * cos(ky*y)
            # kx = 2π/Lx (full wave in x), ky = π/Ly (half wave in y, non-zero mean)
            Lx, Ly = self.Lx, self.Ly
            x = torch.linspace(0, Lx, Nx + 1)[:-1]  # (Nx,)
            y_coords = self.y_center                    # (Ny,)
            xx, yy = torch.meshgrid(x, y_coords, indexing="ij")
            pert = (torch.cos(xx * (2.0 * math.pi / Lx))
                    * torch.cos(yy * (math.pi / Ly))).reshape(-1)
            # Normalize so rms = epsilon
            pert = pert / pert.std() * epsilon
            h1 = h1 + pert
            u1 = u1 + pert
            v1 = v1 + pert
            h2 = h2 + pert
            u2 = u2 + pert
            v2 = v2 + pert
        elif perturbation_mode == "random_balanced":
            # Add small ageostrophic random perturbation to all velocity components.
            # This seeds 2D structure directly in u and v (not through geostrophic balance,
            # which would force zero y-mean in u).  The SW dynamics adjusts the height
            # field toward geostrophic balance within a few inertial periods.
            rng = torch.Generator()
            rng.manual_seed(seed)

            # Random velocity perturbations (independent for each component)
            u1 = u1 + epsilon * torch.randn(NxNy, generator=rng)
            v1 = v1 + epsilon * torch.randn(NxNy, generator=rng)
            h1 = h1 + epsilon * torch.randn(NxNy, generator=rng) * (self.f_cor / self.g1)
            u2 = u2 + epsilon * torch.randn(NxNy, generator=rng)
            v2 = v2 + epsilon * torch.randn(NxNy, generator=rng)
            h2 = h2 + epsilon * torch.randn(NxNy, generator=rng) * (self.f_cor / self.g2)
        else:
            # Original random perturbation
            rng = torch.Generator()
            rng.manual_seed(seed)
            h1 = h1 + epsilon * torch.randn(NxNy, generator=rng)
            u1 = u1 + epsilon * torch.randn(NxNy, generator=rng)
            h2 = h2 + epsilon * torch.randn(NxNy, generator=rng)

        state = torch.stack([h1, u1, v1, h2, u2, v2])
        return state.reshape(-1)  # (state_dim,)

    # ------------------------------------------------------------------
    # Spatial gradients  (central differences, periodic BC)
    # ------------------------------------------------------------------

    def _grad_x(self, f: torch.Tensor) -> torch.Tensor:
        """Central difference along the *x* (row) axis, periodic."""
        f2d = f.reshape(self.Nx, self.Ny)
        return (
            torch.roll(f2d, -1, dims=0) - torch.roll(f2d, 1, dims=0)
        ).reshape(-1) / (2.0 * self.dx)

    def _grad_y(self, f: torch.Tensor) -> torch.Tensor:
        """Central difference along the *y* (column) axis, periodic."""
        f2d = f.reshape(self.Nx, self.Ny)
        return (
            torch.roll(f2d, -1, dims=1) - torch.roll(f2d, 1, dims=1)
        ).reshape(-1) / (2.0 * self.dy)

    def _laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """5-point Laplacian on the periodic 2-D grid (flat layout).

        The flat index ``i*Ny + j`` corresponds to grid point ``(i, j)``.
        """
        f2d = f.reshape(self.Nx, self.Ny)
        return (
            torch.roll(f2d, -1, dims=0)
            + torch.roll(f2d, 1, dims=0)
            + torch.roll(f2d, -1, dims=1)
            + torch.roll(f2d, 1, dims=1)
            - 4.0 * f2d
        ).reshape(-1) / (self.dx * self.dy)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _clip_layer_thickness(self, state: torch.Tensor) -> torch.Tensor:
        """Clamp layer thicknesses *h1* and *h2* to ``>= 1e-6``.

        Uses out-of-place operations so that the result is differentiable
        (autograd-safe).
        """
        NxNy = self.Nx * self.Ny
        h1 = torch.clamp(state[..., :NxNy], min=1e-6)
        h2 = torch.clamp(state[..., 3 * NxNy : 4 * NxNy], min=1e-6)
        return torch.cat([
            h1,                                  # h1 (clamped)
            state[..., NxNy : 2 * NxNy],         # u1
            state[..., 2 * NxNy : 3 * NxNy],     # v1
            h2,                                  # h2 (clamped)
            state[..., 4 * NxNy : 5 * NxNy],     # u2
            state[..., 5 * NxNy : 6 * NxNy],     # v2
        ], dim=-1)

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

        wind_pattern = self.wind_pattern.to(state.device)
        wind1 = tau0 * wind_pattern * (1.0 + f1)
        wind2 = tau0 * wind_pattern * (1.0 + f2)

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
        M = self.land_mask.to(state.device)
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
        bickley_jet: bool = True,
        tau0: float | None = None,
        bickley_U: float = 1.0,
        bickley_U2: float = 0.6,
        bickley_H_ref: float = 1.0,
        bickley_L_jet_frac: float = 0.15,
        bickley_perturbation_mode: str = "sinusoidal",
        bickley_epsilon: float = 0.01,
    ) -> tuple:
        """Generate a trajectory, optionally from a Bickley jet initial condition.

        When *bickley_jet* is True the initial state is a geostrophically
        balanced Bickley jet (both layers) superposed with a small random
        perturbation.  Otherwise the old perturbed-resting-state init is used.

        Parameters
        ----------
        num_steps : int
            Number of steps to collect.
        seed : int
            RNG seed.
        spinup_steps : int
            Number of spin-up steps (discarded).
        bickley_jet : bool
            If True, use Bickley jet initial condition.
        tau0 : float, optional
            Wind-stress amplitude.  Defaults to ``self.tau0``.

        Returns
        -------
        traj : Tensor ``(num_steps, state_dim)``
        forcing : Tensor ``(num_steps, 2)``
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
        tau0_eff = self.tau0 if tau0 is None else tau0

        # Initial condition
        if bickley_jet:
            s0 = self._init_bickley_jet(seed=seed + 100, U=bickley_U, U2=bickley_U2,
                                          H_ref=bickley_H_ref, L_jet_frac=bickley_L_jet_frac,
                                          perturbation_mode=bickley_perturbation_mode,
                                          epsilon=bickley_epsilon)
        else:
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
                s, forcing_t[i], tau0_eff, self.f_cor,
                self.g1, self.g2, self.coupling_coeff,
                self.friction, self.viscosity, self.dt,
            )

        # Collect trajectory
        traj_list = [s.clone()]
        for i in range(spinup_steps, spinup_steps + num_steps - 1):
            s = self._rk4_step(
                s, forcing_t[i], tau0_eff, self.f_cor,
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
