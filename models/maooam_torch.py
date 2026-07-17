"""PyTorch-native MAOOAM dynamics — no numba JIT compilation needed.

Extracts the sparse tensor from ``qgs`` and implements the bilinear RHS
contraction using PyTorch sparse operations, enabling GPU acceleration,
autograd differentiation, and scaling beyond numba's JIT limits.

The RHS is a sparse rank-3 bilinear form:

    dx_i / dt = sum_{j,k} T[i,j,k] * xx[j] * xx[k]

where ``xx = [1, x]`` (constant 1 prepended).

Extraction from qgs triggers **no JIT** — we access the tensor coords/data
directly via ``QgsTensor`` internals.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from models.dynamics import DynamicsBase


class MaooamTorchDynamics(DynamicsBase):
    """PyTorch-native MAOOAM coupled ocean-atmosphere dynamics.

    Parameters, state layout, and variable ranges are identical to
    ``MaooamDynamics``, but the RHS is implemented via PyTorch sparse
    operations instead of numba-JIT.

    Parameters
    ----------
    device : str or torch.device
        Device for tensor operations (``"cpu"`` or ``"cuda"``).
    dt : float
        Time step in nondimensional units.
    K : int
        Steps per DA window.
    atm_nx, atm_ny : int
        Atmospheric Fourier truncation.
    occ_nx, occ_ny : int
        Oceanic Fourier truncation.
    kd, kdp : float
        Atmospheric bottom / internal friction.
    sigma : float
        Atmospheric static stability.
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
    compile : bool
        Use ``torch.compile`` on the RK4 integrator (2-9x faster).
    """

    param_names: list[str] = []
    param_dim: int = 0

    def __init__(
        self,
        device: str | torch.device = "cpu",
        dt: float = 0.1,
        K: int = 5,
        compile: bool = True,
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
        self._device = torch.device(device)
        self.dt = dt
        self.K = K
        self.stochastic_forcing = stochastic_forcing
        self.forcing_amplitude = forcing_amplitude

        # --- Build qgs params (pure Python, no JIT) ---
        from qgs.params.params import QgParams
        from qgs.inner_products.analytic import (
            AtmosphericAnalyticInnerProducts,
            OceanicAnalyticInnerProducts,
        )
        from qgs.tensors.qgtensor import QgsTensor

        params = QgParams({
            'phi0_npi': 0.25,
            'n': n_ratio,
        })
        params.scale_params.set_params({
            'scale': scale,
            'f0': f0,
        })
        params.set_atmospheric_channel_fourier_modes(atm_nx, atm_ny)
        params.atmospheric_params.set_params({
            'kd': kd, 'kdp': kdp, 'sigma': sigma,
        })
        params.atemperature_params.set_params({
            'eps': eps, 'T0': T0_atm, 'hlambda': hlambda,
        })
        params.atemperature_params.set_insolation(C_atm, 0)
        params.set_oceanic_basin_fourier_modes(occ_nx, occ_ny)
        params.oceanic_params.set_params({
            'r': r, 'h': h, 'd': d,
        })
        params.gotemperature_params.set_params({
            'gamma': gamma_oc, 'T0': T0_oc,
        })
        params.gotemperature_params.set_insolation(C_oc, 0)
        params.atemperature_params.set_params({'T4': T4})
        params.gotemperature_params.set_params({'T4': T4})
        params.atemperature_params.set_params({'dynamic_T': dynamic_T})
        params.gotemperature_params.set_params({'dynamic_T': dynamic_T})

        # Build inner products + tensor (pure Python + sparse, no JIT)
        aip = AtmosphericAnalyticInnerProducts(params)
        oip = OceanicAnalyticInnerProducts(params)
        aip.connect_to_ocean(oip)
        tens = QgsTensor(params, aip, oip)

        # Store qgs params for spectral_to_physical
        self._qgs_params = params

        # --- Extract sparse tensor ---
        coo = tens.tensor.coords.T  # (nnz, 3) int64
        val = tens.tensor.data      # (nnz,) float64
        jcoo = tens.jacobian_tensor.coords.T
        jval = tens.jacobian_tensor.data

        # --- Store as PyTorch buffers (device-movable) ---
        self.state_dim = params.ndim
        self.register_buffer("_coo_idx", torch.tensor(coo, dtype=torch.long))
        self.register_buffer("_coo_val", torch.tensor(val, dtype=torch.float64))
        self.register_buffer("_jcoo_idx", torch.tensor(jcoo, dtype=torch.long))
        self.register_buffer("_jcoo_val", torch.tensor(jval, dtype=torch.float64))

        # Move buffers to target device
        self.to(self._device)

        # Optionally compile the RK4 step (2-9x faster)
        if compile:
            compiled = torch.compile(self._rk4_step, fullgraph=False)
            self._rk4_step = lambda x: compiled(x)

        # Variable ranges
        self.Natm = params.nmod[0]
        self.Npsi_o = params.nmod[1]
        self.NdT_o = params.nmod[1]

        # Forcing dimension
        self.forcing_dim = self.state_dim if stochastic_forcing else 1

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------

    def _rhs(self, x: torch.Tensor) -> torch.Tensor:
        """Bilinear sparse contraction: dx/dt = T[i,j,k] * xx[j] * xx[k].

        Parameters
        ----------
        x : Tensor (state_dim,)
            State vector (spectral coefficients).

        Returns
        -------
        Tensor (state_dim,)
            Tendency vector.
        """
        xx = torch.cat([x.new_ones(1), x])
        i = self._coo_idx[:, 0]
        j = self._coo_idx[:, 1]
        k = self._coo_idx[:, 2]
        products = xx[j] * xx[k] * self._coo_val
        result = torch.zeros(len(xx), device=x.device, dtype=x.dtype)
        result.scatter_add_(0, i, products)
        return result[1:]

    def _jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Jacobian matrix: A[i,j] = sum_k J[i,j,k] * xx[k].

        Parameters
        ----------
        x : Tensor (state_dim,)
            State vector.

        Returns
        -------
        Tensor (state_dim, state_dim)
            Jacobian matrix.
        """
        xx = torch.cat([x.new_ones(1), x])
        i = self._jcoo_idx[:, 0]
        j = self._jcoo_idx[:, 1]
        k = self._jcoo_idx[:, 2]
        values = xx[k] * self._jcoo_val
        n = len(xx)
        result = torch.zeros(n, n, device=x.device, dtype=x.dtype)
        result.scatter_add_(
            0,
            i.unsqueeze(1).expand_as(values.reshape(-1, 1)),
            values.reshape(-1, 1),
        )
        return result[1:, 1:]

    # ------------------------------------------------------------------
    # RK4 integrator
    # ------------------------------------------------------------------

    def _rk4_step(self, x: torch.Tensor) -> torch.Tensor:
        """One classical RK4 step (correct weights: 1/6, 1/3, 1/3, 1/6)."""
        dt = self.dt
        k1 = self._rhs(x)
        k2 = self._rhs(x + 0.5 * dt * k1)
        k3 = self._rhs(x + 0.5 * dt * k2)
        k4 = self._rhs(x + dt * k3)
        return x + dt * (k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0)

    # ------------------------------------------------------------------
    # DynamicsBase interface
    # ------------------------------------------------------------------

    def step(self, state: torch.Tensor, forcing: torch.Tensor,
             *args, **kwargs) -> torch.Tensor:
        """Single forward step.

        Parameters
        ----------
        state : Tensor (batch, state_dim) or (state_dim,)
        forcing : Tensor (batch, forcing_dim) or (forcing_dim,)

        Returns
        -------
        Tensor same shape as ``state``.
        """
        squeeze = state.ndim == 1
        if squeeze:
            state = state.unsqueeze(0)
            if forcing is not None and forcing.ndim == 1:
                forcing = forcing.unsqueeze(0)

        batch = state.shape[0]
        out = []
        for b in range(batch):
            x = state[b].to(dtype=torch.float64)
            x_new = self._rk4_step(x)
            if self.stochastic_forcing and forcing is not None and b < forcing.shape[0]:
                x_new = x_new + forcing[b].to(dtype=torch.float64)
            out.append(x_new.to(dtype=state.dtype))

        result = torch.stack(out)
        return result.squeeze(0) if squeeze else result

    def generate_full_trajectory(
        self,
        num_steps: int,
        seed: int = 42,
        device=None,
        spinup_steps: int = 5000,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a long trajectory using pure-PyTorch RK4.

        Parameters
        ----------
        num_steps : int
        seed : int
        device : torch.device or None
        spinup_steps : int

        Returns
        -------
        traj : Tensor ``(num_steps, state_dim)``
        forcing : Tensor ``(num_steps, forcing_dim)``
        """
        rng = torch.Generator(device=self.device).manual_seed(seed)
        dtype = torch.float64
        ic = torch.randn(self.state_dim, generator=rng, dtype=dtype,
                         device=self.device) * 0.01

        # --- Spinup ---
        x = ic.clone()
        for _ in range(spinup_steps):
            x = self._rk4_step(x)

        # --- Collect ---
        traj = [x.clone()]
        for _ in range(num_steps - 1 if num_steps > 1 else 0):
            x = self._rk4_step(x)
            traj.append(x.clone())
        traj = torch.stack(traj)

        # Forcing
        if self.stochastic_forcing:
            f = torch.randn(num_steps, self.state_dim, generator=rng,
                           dtype=dtype, device=self.device) * self.forcing_amplitude
        else:
            f = torch.zeros(num_steps, 1, dtype=dtype, device=self.device)

        return traj.float(), f.float()

    # ------------------------------------------------------------------
    # Physical field reconstruction
    # ------------------------------------------------------------------

    def spectral_to_physical(self, state: np.ndarray, interp_size: int | None = None):
        """Reconstruct physical fields from spectral state using qgs diagnostics.

        Parameters
        ----------
        state : ndarray (state_dim,)
        interp_size : int or None
            If set, interpolate to ``(interp_size, interp_size)`` using
            cubic spline (``scipy.ndimage.zoom``).

        Returns
        -------
        dict with keys ``psi_upper``, ``psi_lower``, ``psi_oc``, ``T_atm``, ``T_oc``.
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
        T_atm = MiddleAtmosphericTemperatureAnomalyDiagnostic(self._qgs_params)
        T_oc = OceanicLayerTemperatureAnomalyDiagnostic(self._qgs_params)

        fields = {
            "psi_upper": psi_up(t, state_2d)[0],
            "psi_lower": psi_lo(t, state_2d)[0],
            "psi_oc": psi_oc(t, state_2d)[0],
            "T_atm": T_atm(t, state_2d)[0],
            "T_oc": T_oc(t, state_2d)[0],
        }

        if interp_size is not None:
            from scipy.ndimage import zoom
            out = {}
            for k, v in fields.items():
                zf = interp_size / v.shape[0]
                out[k] = zoom(v, zf, order=3)
            return out
        return fields

    def get_field_names(self) -> list[str]:
        return ["psi_upper", "psi_lower", "psi_oc", "T_atm", "T_oc"]

    def __repr__(self):
        return (
            f"MaooamTorchDynamics(state_dim={self.state_dim}, "
            f"Natm={self.Natm}, Noc={self.Npsi_o}, "
            f"dt={self.dt}, device={self.device})"
        )