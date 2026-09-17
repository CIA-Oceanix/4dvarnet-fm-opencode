"""EXPERIMENTAL / NOT PRODUCTION -- see docs/results/psi_decomposition_results.md section 3.

Status (2026-09-16):
  * `guided_sample_pigdm` WORKS but is under-tuned: best gamma=1.0 gives S0 RMSE
    0.991 vs the repo's tuned normalized-gradient baseline at 0.569 (both far
    better than unguided, 1.746). The residual gap is traced to the tau-SCHEDULE,
    not the overall scale -- see the results note.
  * `smc_conditional_sample` is DEGENERATE at realistic dimension and must not be
    used as-is: with a prior proposal the twisting weights collapse (observed:
    resampling at every step, ensemble spread 0.03, RMSE 1.72). Fixing it requires
    a guided (twisted) proposal with its density ratio, as in TDS -- not the
    prior-proposal simplification taken here.

Two principled conditional samplers for a flow-matching prior, to replace /
benchmark against the repo's fixed-length normalized-gradient guidance.

Conventions match models/interpolant.py: x_tau = alpha_tau*x0 + beta_tau*x1 with
alpha_tau = (1-tau)*sigma0, beta_tau = tau, x0 ~ N(0, sigma0^2 I).

(1) `guided_sample_pigdm` -- PiGDM / Rozet-Louppe style VARIANCE-WEIGHTED guidance.
    Gaussian approximation p(x1|x_tau) ~ N(xhat1, r_tau^2 I) with
        r_tau^2 = p*alpha^2 / (beta^2*p + alpha^2)      (p = prior variance)
    so  p(y|x_tau) ~ N(y; xhat1, (sigma_y^2 + gamma*r_tau^2) I)  and
        grad log p(y|x_tau) = -grad_x ||y-xhat1||^2 / (2(sigma_y^2 + gamma r_tau^2)).
    The posterior probability-flow velocity is
        v_post = v + (alpha^2 / (beta*(1-tau))) * grad log p(y|x_tau),
    which for this interpolant reduces the coefficient to (1-tau)*sigma0^2/tau.
    Unlike the normalized scheme this RETAINS the magnitude, so sigma_y (=R_var)
    and the denoiser uncertainty actually enter -- both are inert under g/||g||.

(2) `smc_conditional_sample` -- twisted SMC, asymptotically exact as n_particles->inf.
    Intermediate targets pi_k(x) ~ p_tau(x) * g_tau(x) with the same Gaussian
    twisting function g_tau. Using the UNGUIDED flow-SDE transition as the
    proposal makes the incremental weight collapse to g_new/g_old (no transition
    densities needed). At tau=1, xhat1 = x so g is the exact likelihood, hence
    the weighted particle set targets p(x1|y) exactly in the large-N limit.

    The prior SDE is  dx = [v + (s_tau^2/2) score] dtau + s_tau dW  with
    score = (beta*xhat1 - x)/alpha^2 (Tweedie). Choosing s_tau = c*(1-tau)*sigma0
    cancels the alpha^2 denominator exactly, giving the singularity-free drift
        v + (c^2/2)*(tau*xhat1 - x).
"""
import torch


def r_tau_sq(tau: float, prior_var: float, sigma0: float) -> float:
    a2 = ((1.0 - tau) * sigma0) ** 2
    b2 = tau ** 2
    return prior_var * a2 / (b2 * prior_var + a2 + 1e-12)


def _masked_sq_resid(x_hat1, y, obs_mask, obs_channel_mask=None):
    """(B,) sum of squared residuals over observed entries."""
    y_clean = torch.nan_to_num(y, nan=0.0)
    mask = obs_mask.to(x_hat1.dtype).unsqueeze(-1)
    if obs_channel_mask is not None:
        mask = mask * obs_channel_mask.to(x_hat1.dtype)
    return (((x_hat1 - y_clean) ** 2) * mask).flatten(1).sum(1)


def guided_sample_pigdm(model, batch, R_var, N_outer=10, prior_var=1.0, gamma=1.0,
                        n_members=1, max_step_norm=None, obs_channel_mask=None):
    """Item (1). Returns (x1, n_forward); x1 is (B,T,D) or (B,T,D,n_members)."""
    obs = batch.obs
    B, T, _ = obs.shape
    device = obs.device
    dt = 1.0 / N_outer
    s0 = model.sigma_prior

    def _run_one():
        x = torch.randn(B, T, model.state_dim, device=device) * s0
        for step in range(N_outer):
            tau_val = step / N_outer
            tau = torch.full((B,), tau_val, device=device)
            beta = max(tau_val, dt / 2.0)          # guard tau=0
            a2 = ((1.0 - tau_val) * s0) ** 2
            var_eff = R_var + gamma * r_tau_sq(tau_val, prior_var, s0)
            coef = dt * a2 / (beta * max(1.0 - tau_val, 1e-6))

            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                v = model.forward(x, batch, tau)
                x_hat1 = x + (1.0 - tau_val) * v
                sq = _masked_sq_resid(x_hat1, batch.obs, batch.obs_mask,
                                      obs_channel_mask).sum()
                grad = torch.autograd.grad(sq, x)[0]
            # grad log p(y|x_tau) = -grad(sq)/(2 var_eff)
            step_vec = -coef * grad / (2.0 * var_eff)
            if max_step_norm is not None:
                nrm = step_vec.flatten(1).norm(dim=1).clamp_min(1e-12)
                scale = (max_step_norm / nrm).clamp(max=1.0)
                step_vec = step_vec * scale.view(B, *([1] * (step_vec.dim() - 1)))
            x = (x + dt * v.detach() + step_vec).detach()
        return x

    if n_members == 1:
        return _run_one(), N_outer
    return torch.stack([_run_one() for _ in range(n_members)], dim=-1), N_outer


def _systematic_resample(logw):
    """Per-batch-element systematic resampling. logw (N,B) -> idx (N,B) long."""
    N, B = logw.shape
    w = torch.softmax(logw, dim=0)                       # (N,B)
    cdf = torch.cumsum(w, dim=0)
    cdf = cdf / cdf[-1:].clamp_min(1e-12)
    u0 = torch.rand(1, B, device=logw.device) / N
    pts = u0 + torch.arange(N, device=logw.device).view(N, 1) / N   # (N,B)
    idx = torch.searchsorted(cdf.t().contiguous(), pts.t().contiguous()).t()
    return idx.clamp_(0, N - 1)


def smc_conditional_sample(model, batch, R_var, N_outer=10, prior_var=1.0,
                           n_particles=32, churn=1.0, gamma=1.0, ess_frac=0.5,
                           obs_channel_mask=None, repeat_batch=None):
    """Item (3). Twisted SMC. Returns (particles, logw, n_resample).

    particles: (B,T,D,N); logw: (N,B) final log-weights (unnormalized).
    `repeat_batch(batch, N)` must return a batch whose fields are tiled N times
    along dim 0 (particle-major: index p*B + b).
    """
    obs = batch.obs
    B, T, D = obs.shape
    device = obs.device
    dt = 1.0 / N_outer
    s0 = model.sigma_prior
    Np = n_particles

    big = repeat_batch(batch, Np)
    y = batch.obs
    om = batch.obs_mask
    ocm = obs_channel_mask

    x = torch.randn(Np, B, T, D, device=device) * s0
    logw = torch.zeros(Np, B, device=device)
    n_resample = 0

    def log_g(xf, tau_val):
        """log twisting fn, (Np,B). xf is (Np,B,T,D)."""
        var_eff = R_var + gamma * r_tau_sq(tau_val, prior_var, s0)
        with torch.no_grad():
            tau = torch.full((Np * B,), tau_val, device=device)
            v = model.forward(xf.reshape(Np * B, T, D), big, tau)
            xh = xf.reshape(Np * B, T, D) + (1.0 - tau_val) * v
            sq = _masked_sq_resid(xh, y.repeat(Np, 1, 1), om.repeat(Np, 1),
                                  None if ocm is None else ocm.repeat(Np, 1, 1))
        return (-sq / (2.0 * var_eff)).view(Np, B), v.view(Np, B, T, D), xh.view(Np, B, T, D)

    lg_prev, v_prev, xh_prev = log_g(x, 0.0)

    for step in range(N_outer):
        tau_val = step / N_outer
        tau_next = (step + 1) / N_outer
        s_tau = churn * (1.0 - tau_val) * s0
        # singularity-free drift: v + (c^2/2)*(tau*xhat1 - x)
        drift = v_prev + (churn ** 2 / 2.0) * (tau_val * xh_prev - x)
        x = x + dt * drift + s_tau * (dt ** 0.5) * torch.randn_like(x)

        lg_new, v_new, xh_new = log_g(x, tau_next)
        logw = logw + (lg_new - lg_prev)
        lg_prev, v_prev, xh_prev = lg_new, v_new, xh_new

        ess = 1.0 / torch.softmax(logw, dim=0).pow(2).sum(0)      # (B,)
        if float(ess.min()) < ess_frac * Np:
            idx = _systematic_resample(logw)                       # (Np,B)
            gi = idx.view(Np, B, 1, 1).expand(Np, B, T, D)
            x = torch.gather(x, 0, gi)
            lg_prev = torch.gather(lg_prev, 0, idx)
            v_prev = torch.gather(v_prev, 0, gi)
            xh_prev = torch.gather(xh_prev, 0, gi)
            logw = torch.zeros_like(logw)
            n_resample += 1

    return x.permute(1, 2, 3, 0).contiguous(), logw, n_resample
