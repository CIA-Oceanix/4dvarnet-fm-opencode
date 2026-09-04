import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNet1D

# Recognized update_input tokens (mirrors the config-string taxonomy explored on
# CIA-Oceanix/4dvarnet-global-mapping's ronan_devs branch, contrib/4dvarnet_latent/
# models.py::GradSolver_withStep). Only the gradient-free modes are implemented
# here; the gradient/pseudo-gradient modes need obs_cost/prior_cost (a real or
# proxy variational cost) ported from ocean4dvarnet's GradSolver, deferred to a
# follow-up ("FDV2").
_IMPLEMENTED_UPDATE_INPUTS = ("obs+state", "obs-only")
_DEFERRED_UPDATE_INPUTS = ("grad-only", "grad+state", "subgrad+state")


class FourDVarNetSolver(nn.Module):
    """Unrolled 4DVarNet-style solver: the per-iteration update is the output
    of a UNet fed the current state and/or the observations, run for a fixed
    number of iterations -- no explicit variational cost or gradient term.

    This is deliberately simpler than ``ocean4dvarnet``'s ``GradSolver``
    (which computes ``torch.autograd.grad`` of a ``prior_cost + obs_cost``
    variational cost each step, and feeds only that gradient to a ConvLSTM
    update block). ``update_input`` selects what the update block sees each
    iteration, matching the config-string taxonomy explored on
    CIA-Oceanix/4dvarnet-global-mapping's ``ronan_devs`` branch
    (``GradSolver_withStep``'s ``input_grad_update``):

    - ``"obs+state"`` (default): ``concat(state, obs)`` -- exactly the variant
      requested first: a UNet block conditioned on both the observations and
      the running state estimate, no cost function at all.
    - ``"obs-only"``: just the observations, no state feedback.

    ``"grad-only"``/``"grad+state"``/``"subgrad+state"`` (real or pseudo
    gradient-conditioned modes) are recognized names but not yet implemented
    -- they need a ported ``obs_cost``/``prior_cost`` -- and raise
    ``NotImplementedError`` at construction rather than silently falling back
    to a different mode.
    """

    def __init__(self, state_dim=24, hidden_channels=None, time_emb_dim=64,
                 N_outer=10, dropout=0.1, update_input="obs+state"):
        super().__init__()
        if update_input in _DEFERRED_UPDATE_INPUTS:
            raise NotImplementedError(
                f"update_input={update_input!r} needs a ported obs_cost/prior_cost "
                "(gradient-based modes) -- not yet implemented, see FDV2."
            )
        if update_input not in _IMPLEMENTED_UPDATE_INPUTS:
            raise ValueError(
                f"Unknown update_input={update_input!r}; expected one of "
                f"{_IMPLEMENTED_UPDATE_INPUTS + _DEFERRED_UPDATE_INPUTS}"
            )
        self.update_input = update_input
        self.state_dim = state_dim
        self.N_outer = N_outer
        in_state_dim = state_dim if update_input == "obs-only" else 2 * state_dim
        self.unet = UNet1D(
            state_dim=in_state_dim,
            hidden_channels=hidden_channels,
            time_emb_dim=time_emb_dim,
            use_obs=False,
            use_energy=False,
            dropout=dropout,
            output_dim=state_dim,
        )

    def _update_input_tensor(self, x, obs_clean):
        if self.update_input == "obs-only":
            return obs_clean
        return torch.cat([x, obs_clean], dim=-1)

    def forward(self, batch, N_outer=None):
        N = self.N_outer if N_outer is None else N_outer
        obs_clean = torch.nan_to_num(batch.obs, nan=0.0)  # (B, T, D)
        B, T, D = obs_clean.shape
        x = torch.zeros(B, T, D, device=obs_clean.device)  # x_0 = 0
        denom = max(N - 1, 1)
        for k in range(N):
            tau_k = torch.full((B,), k / denom, device=x.device)
            inp = self._update_input_tensor(x, obs_clean).transpose(1, 2)
            gmod = self.unet(inp, tau=tau_k).transpose(1, 2)
            x = x - (1.0 / N) * gmod
        return x

    def compute_loss(self, batch):
        return F.mse_loss(self.forward(batch), batch.states)

    def sample(self, batch, N_outer=None):
        return self.forward(batch, N_outer=N_outer)
