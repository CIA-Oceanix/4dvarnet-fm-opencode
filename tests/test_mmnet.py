import torch

from models.mmnet import MMNet


class _Batch:
    def __init__(self, B=2, T=40, D=3, interval=10):
        self.states = torch.randn(B, T, D)
        self.obs = torch.full((B, T, D), float("nan"))
        self.obs[:, ::interval] = self.states[:, ::interval] + 0.1 * torch.randn(B, T // interval, D)
        self.batch_size = B


def test_forward_shape_and_finite():
    model = MMNet(hidden_channels=8, n_iter=3)
    out = model(_Batch())
    assert out.shape == (2, 40, 3)
    assert torch.isfinite(out).all()


def test_eval_under_no_grad():
    model = MMNet(hidden_channels=8, n_iter=3).eval()
    with torch.no_grad():
        out = model(_Batch())
    assert out.shape == (2, 40, 3) and not out.requires_grad


def test_loss_reaches_prior():
    model = MMNet(hidden_channels=8, n_iter=3)
    model.compute_loss(_Batch()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.prior.parameters())


def test_unit_step_recovers_observations():
    model = MMNet(hidden_channels=8, n_iter=1, step=1.0, prior_weight=0.0, init_std=0.0)
    batch = _Batch()
    out = model(batch)
    observed = ~torch.isnan(batch.obs)
    assert torch.allclose(out[observed], batch.obs[observed], atol=1e-6)
    assert torch.all(out[~observed] == 0)


def test_overfit_decreases_loss():
    torch.manual_seed(0)
    model = MMNet(hidden_channels=16, n_iter=3)
    batch = _Batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = model.compute_loss(batch).item()
    for _ in range(40):
        opt.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        opt.step()
    assert loss.item() < first
