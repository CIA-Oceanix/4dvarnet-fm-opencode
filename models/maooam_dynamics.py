"""MAOOAM dynamics: 2-layer QG atmosphere + SW ocean via qgs package.

Wraps the Modular Arbitrary-Order Ocean-Atmosphere Model (MAOOAM) from the
``qgs`` package into the ``DynamicsBase`` interface used by our 4DVarNet-FM
framework.

State vector layout (spectral coefficients):
    [psi_a(0..Natm-1), theta_a(0..Natm-1), psi_o(0..Noc-1), delta_T_o(0..Noc-1)]

where:
    psi_a  = atmospheric barotropic streamfunction
    theta_a = atmospheric baroclinic streamfunction
    psi_o  = oceanic streamfunction
    delta_T_o = oceanic temperature anomaly

Physical-layer streamfunctions:
    Upper atmosphere: psi^1 = psi_a + theta_a
    Lower atmosphere: psi^3 = psi_a - theta_a
    Ocean:            psi_o

References
----------
Demaeyer, De Cruz & Vannitsem (2020). qgs: A flexible Python framework of
    reduced-order multiscale climate models. JOSS, 5(56), 2597.
"""

from __future__ import annotations

import numpy as np
import torch
from models.dynamics import DynamicsBase

# ---------------------------------------------------------------------------
# Helpers to compute mode counts from qgs parameterization
# ---------------------------------------------------------------------------

def _count_atm_modes(nx: int, ny: int) -> int:
    """Number of atmospheric spectral modes for given truncation."""
    n = 0
    for i_nx in range(1, nx + 1):
        for i_ny in range(1, ny + 1):
            n += 3 if i_nx == 1 else 2
    return n


def _count_oc_modes(nx: int, ny: int) -> int:
    """Number of oceanic spectral modes for given truncation."""
    return nx * ny


# ---------------------------------------------------------------------------
# Dynamics wrapper
# ---------------------------------------------------------------------------

class MaooamDynamics(DynamicsBase):
    """MAOOAM coupled ocean-atmosphere dynamics.

    Parameters
    ----------
    dt : float
        Time step in nondimensional units (0.1 ≈ 16 min physical).
    K : int
        Steps per DA window.
    atm_nx, atm_ny : int
        Atmospheric Fourier truncation.
    occ_nx, occ_ny : int
        Oceanic Fourier truncation.
    kd, kdp : float
        Atmospheric bottom / internal friction coefficients.
    sigma : float
        Atmospheric static stability parameter.
    r : float
        Oceanic bottom friction.
    h : float
        Ocean layer depth (m).
    d : float
        Ocean–atmosphere mechanical coupling.
    eps : float
        Grey-body emissivity.
    T0_atm, T0_oc : float
        Reference temperatures (K).
    hlambda : float
        Sensible + turbulent heat exchange (W/m²/K).
    gamma_oc : float
        Oceanic heat capacity (J/m²/K).
    C_atm, C_oc : float
        Insolation parameters.
    scale : float
        Characteristic meridional scale (m).
    f0 : float
        Coriolis parameter (s⁻¹).
    n_ratio : float
        Aspect ratio n = 2*Ly/Lx.
    T4 : bool
        Use nonlinear T⁴ radiation.
    dynamic_T : bool
        Evolve 0th-order temperature mode dynamically.
    stochastic_forcing : bool
        Add AR(1) stochastic forcing to represent model error.
    forcing_amplitude : float
        Amplitude of stochastic forcing.
    """

    param_names: list[str] = []
    param_dim: int = 0

    def __init__(
        self,
        dt: float = 0.1,
        K: int = 5,
        # Atmosphere
        atm_nx: int = 4,
        atm_ny: int = 4,
        kd: float = 0.0290,
        kdp: float = 0.0290,
        sigma: float = 0.2,
        # Ocean
        occ_nx: int = 4,
        occ_ny: int = 4,
        r: float = 1e-7,
        h: float = 136.5,
        d: float = 1.1e-7,
        # Temperature
        eps: float = 0.7,
        T0_atm: float = 289.3,
        hlambda: float = 15.06,
        gamma_oc: float = 5.6e8,
        T0_oc: float = 301.46,
        # Forcing
        C_atm: float = 103.33,
        C_oc: float = 310.0,
        # Domain
        scale: float = 5e6,
        f0: float = 1.032e-4,
        n_ratio: float = 1.5,
        # Radiation / temperature options
        T4: bool = False,
        dynamic_T: bool = False,
        # Stochastic forcing for DA model error
        stochastic_forcing: bool = False,
        forcing_amplitude: float = 0.01,
    ):
        super().__init__()
        from qgs.params.params import QgParams
        from qgs.functions.tendencies import create_tendencies

        self.dt = dt
        self.K = K

        # --- Build qgs parameter object ---
        self._qgs_params = QgParams({
            'phi0_npi': 0.25,
            'n': n_ratio,
        })

        # Scale / domain
        self._qgs_params.scale_params.set_params({
            'scale': scale,
            'f0': f0,
        })

        # Atmospheric modes
        self._qgs_params.set_atmospheric_channel_fourier_modes(atm_nx, atm_ny)

        # Atmospheric friction / stability
        self._qgs_params.atmospheric_params.set_params({
            'kd': kd, 'kdp': kdp, 'sigma': sigma,
        })

        # Atmospheric temperature
        self._qgs_params.atemperature_params.set_params({
            'eps': eps, 'T0': T0_atm, 'hlambda': hlambda,
        })
        self._qgs_params.atemperature_params.set_insolation(C_atm, 0)

        # Ocean
        self._qgs_params.set_oceanic_basin_fourier_modes(occ_nx, occ_ny)
        self._qgs_params.oceanic_params.set_params({
            'r': r, 'h': h, 'd': d,
        })

        # Ocean temperature
        self._qgs_params.gotemperature_params.set_params({
            'gamma': gamma_oc, 'T0': T0_oc,
        })
        self._qgs_params.gotemperature_params.set_insolation(C_oc, 0)

        # Radiation options
        self._qgs_params.atemperature_params.set_params({'T4': T4})
        self._qgs_params.gotemperature_params.set_params({'T4': T4})
        self._qgs_params.atemperature_params.set_params({'dynamic_T': dynamic_T})
        self._qgs_params.gotemperature_params.set_params({'dynamic_T': dynamic_T})

        # --- Create numba-JIT tendencies ---
        self._f, self._Df = create_tendencies(self._qgs_params)

        # State dimension
        self.state_dim = self._qgs_params.ndim

        # Variable ranges: [psi_a, theta_a, psi_o, delta_T_o]
        vr = self._qgs_params.variables_range
        self._vr = vr  # tuple of 4 ints: (Natm, 2*Natm, 2*Natm+Noc, 2*Natm+2*Noc)
        self.Natm = vr[0]
        self.Ntheta = vr[1] - vr[0]
        self.Npsi_o = vr[2] - vr[1]
        self.NdT_o = vr[3] - vr[2]

        # Stochastic forcing
        self.stochastic_forcing = stochastic_forcing
        self.forcing_amplitude = forcing_amplitude
        self.forcing_dim = self.state_dim if stochastic_forcing else 0

        # Precompute RK4 coefficients
        self._a = np.array([0.0, 0.5, 0.5, 1.0])
        self._b = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])

    # ------------------------------------------------------------------
    # Internal RK4 (pure numpy, calls numba-JIT tendencies)
    # ------------------------------------------------------------------

    def _rk4_numpy(self, x: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
        """One classical RK4 step on a numpy state vector."""
        dt = self.dt
        f = self._f  # signature: f(t, x)

        k1 = f(0.0, x)
        k2 = f(0.0, x + 0.5 * dt * k1)
        k3 = f(0.0, x + 0.5 * dt * k2)
        k4 = f(0.0, x + dt * k3)

        x_new = x + dt * (k1/6.0 + k2/3.0 + k3/3.0 + k4/4.0)

        # Add stochastic forcing if requested
        if q is not None:
            x_new = x_new + q

        return x_new

    # ------------------------------------------------------------------
    # DynamicsBase interface
    # ------------------------------------------------------------------

    def step(self, state: torch.Tensor, forcing: torch.Tensor,
             *args, **kwargs) -> torch.Tensor:
        """Single forward step."""
        orig_shape = state.shape
        x = state.detach().cpu().numpy().astype(np.float64).ravel()
        q = None
        if self.stochastic_forcing and forcing is not None and forcing.numel() > 0:
            q = forcing.detach().cpu().numpy().astype(np.float64).ravel()
        x_new = self._rk4_numpy(x, q)
        return torch.tensor(x_new, dtype=state.dtype, device=state.device).reshape(orig_shape)

    def generate_full_trajectory(
        self,
        num_steps: int,
        seed: int = 42,
        device=None,
        spinup_steps: int = 5000,
        bickley_jet: bool = True,  # ignored, kept for interface compat
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a long trajectory using qgs's native integrator.

        Returns
        -------
        traj : Tensor ``(num_steps, state_dim)``
        forcing : Tensor ``(num_steps, forcing_dim)``
        """
        from qgs.integrators.integrator import RungeKuttaIntegrator

        rng = np.random.RandomState(seed)

        # Random initial condition on the attractor basin
        ic = rng.randn(self.state_dim).astype(np.float64) * 0.01

        # --- Spinup (discard transient) ---
        integrator = RungeKuttaIntegrator()
        integrator.set_func(self._f)
        integrator.integrate(0.0, spinup_steps * self.dt, self.dt,
                             ic=ic, write_steps=0)
        _, y = integrator.get_trajectories()

        # --- Collect trajectory ---
        integrator.integrate(0.0, num_steps * self.dt, self.dt,
                             ic=y, write_steps=1)
        _, traj_np = integrator.get_trajectories()
        integrator.terminate()

        # traj_np shape: (state_dim, num_steps+1) → transpose
        traj_np = traj_np.T  # (num_steps+1, state_dim)
        # Drop last point if we got one extra
        if traj_np.shape[0] > num_steps:
            traj_np = traj_np[:num_steps]

        traj = torch.tensor(traj_np, dtype=torch.float32)

        # Forcing (zeros or stochastic)
        if self.stochastic_forcing:
            forcing_np = rng.randn(num_steps, self.state_dim).astype(np.float32) * self.forcing_amplitude
            forcing_t = torch.tensor(forcing_np, dtype=torch.float32)
        else:
            forcing_t = torch.zeros(num_steps, 1, dtype=torch.float32)

        return traj, forcing_t

    # ------------------------------------------------------------------
    # Physical field reconstruction
    # ------------------------------------------------------------------

    def spectral_to_physical(self, state: np.ndarray, interp_size: int | None = None):
        """Reconstruct physical fields from spectral state.

        Parameters
        ----------
        state : ndarray (state_dim,)
        interp_size : int or None
            If set, interpolate all fields to (interp_size, interp_size) using
            cubic spline (scipy.ndimage.zoom).

        Returns
        -------
        dict with keys:
            'psi_upper' : upper atmosphere streamfunction
            'psi_lower' : lower atmosphere streamfunction
            'psi_oc'    : ocean streamfunction
            'T_atm'     : atmospheric temperature anomaly
            'T_oc'      : ocean temperature anomaly
        """
        from qgs.diagnostics.streamfunctions import (
            UpperLayerAtmosphericStreamfunctionDiagnostic,
            LowerLayerAtmosphericStreamfunctionDiagnostic,
            OceanicLayerStreamfunctionDiagnostic,
        )
        from qgs.diagnostics.temperatures import (
            MiddleAtmosphericTemperatureAnomalyDiagnostic,
            OceanicLayerTemperatureAnomalyDiagnostic,
        )

        state_2d = state.reshape(-1, 1)
        t = np.array([0.0])

        psi_up = UpperLayerAtmosphericStreamfunctionDiagnostic(self._qgs_params)
        psi_lo = LowerLayerAtmosphericStreamfunctionDiagnostic(self._qgs_params)
        psi_oc = OceanicLayerStreamfunctionDiagnostic(self._qgs_params)
        T_atm  = MiddleAtmosphericTemperatureAnomalyDiagnostic(self._qgs_params)
        T_oc   = OceanicLayerTemperatureAnomalyDiagnostic(self._qgs_params)

        fields = {
            'psi_upper': psi_up(t, state_2d)[0],
            'psi_lower': psi_lo(t, state_2d)[0],
            'psi_oc':    psi_oc(t, state_2d)[0],
            'T_atm':     T_atm(t, state_2d)[0],
            'T_oc':      T_oc(t, state_2d)[0],
        }

        if interp_size is not None:
            from scipy.ndimage import zoom
            out = {}
            for k, v in fields.items():
                zoom_factor = interp_size / v.shape[0]
                out[k] = zoom(v, zoom_factor, order=3)
            return out

        return fields

    def get_field_names(self) -> list[str]:
        return ['psi_upper', 'psi_lower', 'psi_oc', 'T_atm', 'T_oc']

    def __repr__(self):
        return (f"MaooamDynamics(state_dim={self.state_dim}, "
                f"Natm={self.Natm}, Noc={self.Npsi_o}, dt={self.dt})")
