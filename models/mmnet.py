import torch
import torch.nn as nn
import torch.nn.functional as F

CURVATURES = ("fixed", "bound", "lstm_const", "lstm_bound")
LOSSES = ("mse", "cosine")


class ConvPrior(nn.Module):
    def __init__(self, state_dim: int = 3, hidden_channels: int = 32, dilations=(1, 2, 4)):
        super().__init__()
        layers = []
        cin = state_dim
        for d in dilations:
            layers += [nn.Conv1d(cin, hidden_channels, 3, padding=d, dilation=d), nn.SiLU()]
            cin = hidden_channels
        for d in reversed(dilations[1:]):
            layers += [nn.Conv1d(cin, hidden_channels, 3, padding=d, dilation=d), nn.SiLU()]
        layers.append(nn.Conv1d(cin, state_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class LSTMCurvature(nn.Module):
    def __init__(self, state_dim: int = 3, hidden: int = 32, p_init: float = 1.0):
        super().__init__()
        self.f1 = nn.LSTMCell(state_dim, hidden)
        self.head1 = nn.Linear(hidden, state_dim)
        self.f2 = nn.LSTMCell(2 * state_dim, hidden)
        self.head2 = nn.Linear(hidden, state_dim)
        for head in (self.head1, self.head2):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, p_init ** 0.5)

    def forward(self, x: torch.Tensor, g: torch.Tensor, state):
        B, T, D = x.shape
        s1, s2 = state if state is not None else (None, None)
        s1 = self.f1(g.reshape(B * T, D), s1)
        feat = self.head1(s1[0])
        s2 = self.f2(torch.cat([x.reshape(B * T, D), feat], dim=-1), s2)
        return (feat * self.head2(s2[0])).reshape(B, T, D), (s1, s2)


def _norm(v: torch.Tensor) -> torch.Tensor:
    return v.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)


def hessian_spectral_estimate(g: torch.Tensor, x: torch.Tensor, iters: int = 5):
    v = torch.randn_like(x)
    v = v / _norm(v)
    for _ in range(iters):
        (u,) = torch.autograd.grad((g * v).sum(), x, retain_graph=True)
        v = u / _norm(u)
    (u,) = torch.autograd.grad((g * v).sum(), x, retain_graph=True)
    return (v * u).flatten(1).sum(dim=1).abs()


class MMNet(nn.Module):
    def __init__(self, state_dim: int = 3, hidden_channels: int = 32, n_iter: int = 10,
                 step: float = 0.5, prior_weight: float = 1.0, init_std: float = 1e-3,
                 dilations=(1, 2, 4), curvature: str = "fixed", loss_type: str = "mse",
                 lstm_hidden: int = 32, power_iters: int = 5, kappa: float = 1.1,
                 p_min: float = 1e-2, p_max: float = 1.0, p_cap: float = 10.0, p_init: float = 1.0,
                 bound_estimator: str = "rayleigh", warm_start: bool = False, bound_grad: bool = False,
                 grad_weight: float = 0.0):
        super().__init__()
        assert curvature in CURVATURES and loss_type in LOSSES and bound_estimator in ("rayleigh", "hvnorm")
        self.state_dim = state_dim
        self.n_iter = n_iter
        self.step = step
        self.prior_weight = prior_weight
        self.init_std = init_std
        self.curvature = curvature
        self.loss_type = loss_type
        self.power_iters = power_iters
        self.kappa = kappa
        self.p_min = p_min
        self.p_max = p_max
        self.p_cap = p_cap
        self.bound_estimator = bound_estimator
        self.warm_start = warm_start
        self.bound_grad = bound_grad
        self.grad_weight = grad_weight
        self.prior = ConvPrior(state_dim, hidden_channels, tuple(dilations))
        self.curv_net = LSTMCurvature(state_dim, lstm_hidden, p_init) if curvature.startswith("lstm") else None
        self.record_diag = False
        self.diag: dict = {}

    def _cost_per_sample(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        phi = self.prior(x)
        if self.loss_type == "cosine":
            data = 1.0 - F.cosine_similarity((mask * x).flatten(1), y.flatten(1), dim=1, eps=1e-8)
            prior = 1.0 - F.cosine_similarity(x.flatten(1), phi.flatten(1), dim=1, eps=1e-8)
            return data + self.prior_weight * prior
        data = 0.5 * (mask * (y - x) ** 2).flatten(1).sum(1)
        prior = 0.5 * self.prior_weight * ((x - phi) ** 2).flatten(1).sum(1)
        return data + prior

    def _cost(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._cost_per_sample(x, y, mask).sum()

    def _upper_bound(self, g: torch.Tensor, x: torch.Tensor, v=None, differentiable: bool = False):
        if self.bound_estimator == "rayleigh":
            lam = hessian_spectral_estimate(g, x, self.power_iters)
            bar = 1.0 / (self.kappa * lam.clamp_min(1e-6))
            return bar.clamp(self.p_min, self.p_cap).view(-1, 1, 1).detach(), None
        v = torch.randn_like(x) if v is None or not self.warm_start else v
        v = v / _norm(v)
        for _ in range(self.power_iters):
            (u,) = torch.autograd.grad((g * v).sum(), x, retain_graph=True)
            v = u / _norm(u)
        (u,) = torch.autograd.grad((g * v).sum(), x, retain_graph=True, create_graph=differentiable)
        lam = u.flatten(1).norm(dim=1)
        bar = (1.0 / (self.kappa * lam.clamp_min(1e-6))).clamp(self.p_min, self.p_cap).view(-1, 1, 1)
        return (bar if differentiable else bar.detach()), v.detach()

    def _majorant(self, x, g, state, bar):
        if self.curvature == "fixed":
            return torch.ones_like(g), state
        if self.curvature == "bound":
            return bar.expand_as(g), state
        p_tilde, state = self.curv_net(x, g, state)
        if self.curvature == "lstm_const":
            return p_tilde.clamp(self.p_min, self.p_max), state
        return torch.minimum(p_tilde.clamp_min(self.p_min), bar.expand_as(g)), state

    def forward(self, batch) -> torch.Tensor:
        obs = batch.obs
        mask = (~torch.isnan(obs)).to(obs.dtype)
        y = torch.nan_to_num(obs, nan=0.0)
        x = self.init_std * torch.randn_like(y)
        create_graph = self.training and torch.is_grad_enabled()
        needs_hvp = self.curvature in ("bound", "lstm_bound")
        state = None
        v_state = None
        record = [] if self.record_diag else None
        with torch.enable_grad():
            for _ in range(self.n_iter):
                if not create_graph:
                    x = x.detach()
                x = x.requires_grad_(True) if not x.requires_grad else x
                cost = self._cost_per_sample(x, y, mask)
                (g,) = torch.autograd.grad(cost.sum(), x, create_graph=create_graph or needs_hvp)
                bar = None
                if needs_hvp:
                    bar, v_state = self._upper_bound(g, x, v_state, create_graph and self.bound_grad)
                if not create_graph:
                    g = g.detach()
                p, state = self._majorant(x, g, state, bar)
                if record is not None:
                    pd = p.detach()
                    record.append({"cost": cost.detach(), "p_mean": pd.mean().item(),
                                   "at_min": (pd <= self.p_min * (1 + 1e-4)).float().mean().item(),
                                   "at_bar": None if bar is None else (pd >= bar * (1 - 1e-4)).float().mean().item(),
                                   "bar": None if bar is None else bar.detach().flatten()})
                x = x - self.step * p * g
            if record is not None:
                record.append({"cost": self._cost_per_sample(x, y, mask).detach()})
                self.diag = {"trace": record}
        return x if create_graph else x.detach()

    def compute_loss(self, batch) -> torch.Tensor:
        x_hat = self(batch)
        rep = self.prior(batch.states)
        if self.loss_type == "cosine":
            rec = 1.0 - F.cosine_similarity(x_hat.flatten(1), batch.states.flatten(1), dim=1, eps=1e-8)
            reg = 1.0 - F.cosine_similarity(rep.flatten(1), batch.states.flatten(1), dim=1, eps=1e-8)
            return rec.mean() + reg.mean()
        rec = F.mse_loss(x_hat, batch.states)
        if self.grad_weight > 0 and x_hat.shape[1] > 1:
            rec = rec + self.grad_weight * F.mse_loss(x_hat[:, 1:] - x_hat[:, :-1], batch.states[:, 1:] - batch.states[:, :-1])
        return rec + F.mse_loss(rep, batch.states)
