import pytest
import torch

monai = pytest.importorskip("monai")

from models.monai_unet_adapter import MonaiUNet1D  # noqa: E402


@pytest.mark.parametrize("state_dim,obs_dim,length", [(1, 1, 64), (3, 3, 32), (40, 24, 200)])
def test_forward_shape(state_dim, obs_dim, length):
    m = MonaiUNet1D(state_dim=state_dim, obs_dim=obs_dim, hidden_channels=[16, 32], num_res_blocks=1, norm_num_groups=8)
    B = 2
    x = torch.randn(B, state_dim, length)
    obs = torch.randn(B, obs_dim, length)
    tau = torch.rand(B)
    y = m(x, obs=obs, tau=tau)
    assert y.shape == (B, state_dim, length)


def test_no_tau_defaults_to_zero():
    m = MonaiUNet1D(state_dim=2, obs_dim=2, hidden_channels=[16, 32], num_res_blocks=1, norm_num_groups=8)
    x = torch.randn(3, 2, 48)
    obs = torch.randn(3, 2, 48)
    y = m(x, obs=obs)
    assert y.shape == (3, 2, 48)


def test_gradient_flow():
    # MONAI zero-initializes the final output conv (standard diffusion-model
    # practice: the network starts by predicting exactly zero). That makes
    # dL/dx exactly zero on this very first forward pass too (the Jacobian
    # through a zero-weight linear map is zero) even though dL/dW for that
    # same zero-initialized layer is NOT zero -- so we check parameter
    # gradients (the thing that actually drives training), not x.grad.
    m = MonaiUNet1D(state_dim=2, obs_dim=2, hidden_channels=[16, 32], num_res_blocks=1, norm_num_groups=8)
    x = torch.randn(4, 2, 48, requires_grad=True)
    obs = torch.randn(4, 2, 48)
    tau = torch.rand(4)
    y = m(x, obs=obs, tau=tau)
    assert torch.all(y == 0), "expected MONAI's zero-initialized output conv at init"
    y.sum().backward()
    assert x.grad is not None
    n_nonzero = 0
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient reached parameter {name}"
        if not torch.allclose(p.grad, torch.zeros_like(p.grad)):
            n_nonzero += 1
    assert n_nonzero > 0, "no parameter received a nonzero gradient"


def test_use_obs_false_raises_no_error_and_uses_state_only():
    m = MonaiUNet1D(state_dim=2, use_obs=False, hidden_channels=[16, 32], num_res_blocks=1, norm_num_groups=8)
    x = torch.randn(2, 2, 48)
    tau = torch.rand(2)
    y = m(x, obs=None, tau=tau)
    assert y.shape == (2, 2, 48)


def test_use_obs_true_requires_obs():
    m = MonaiUNet1D(state_dim=2, obs_dim=2, hidden_channels=[16, 32], num_res_blocks=1, norm_num_groups=8)
    x = torch.randn(2, 2, 48)
    with pytest.raises(ValueError):
        m(x, obs=None)


def _n_resblock_dropouts(model):
    from monai.networks.nets import diffusion_model_unet as _dmu
    return sum(
        1 for mod in model.backbone.modules()
        if isinstance(mod, _dmu.DiffusionUNetResnetBlock) and hasattr(mod, "dropout")
    )


def test_dropout_is_actually_applied():
    # MONAI's DiffusionUNetResnetBlock has no dropout mechanism at all (no
    # constructor arg, no layer) -- MonaiUNet1D attaches nn.Dropout to each
    # resblock post-construction. Every resblock's conv2 is zero_module-
    # initialized (standard diffusion-model practice), so at init the
    # dropout's *input* is exactly zero everywhere and this test would pass
    # vacuously; train briefly first (against a nonzero target, so the
    # gradient at y=0 isn't itself zero) so the dropout input is nonzero.
    m = MonaiUNet1D(state_dim=2, obs_dim=2, hidden_channels=[16, 32], num_res_blocks=1,
                     norm_num_groups=8, dropout=0.9)
    assert _n_resblock_dropouts(m) > 0, "no dropout attached to any resblock"

    x = torch.randn(4, 2, 48)
    obs = torch.randn(4, 2, 48)
    tau = torch.rand(4)
    target = torch.randn(4, 2, 48)
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    for _ in range(10):
        opt.zero_grad()
        y = m(x, obs=obs, tau=tau)
        (y - target).pow(2).mean().backward()
        opt.step()

    m.train()
    y1 = m(x, obs=obs, tau=tau)
    y2 = m(x, obs=obs, tau=tau)
    assert not torch.allclose(y1, y2), "dropout=0.9 should make train-mode outputs stochastic"

    m.eval()
    y3 = m(x, obs=obs, tau=tau)
    y4 = m(x, obs=obs, tau=tau)
    assert torch.allclose(y3, y4), "eval-mode outputs must be deterministic (dropout off)"


def test_dropout_zero_attaches_no_resblock_dropout():
    # MONAI's own attention block (SABlock, in the middle bottleneck) has its
    # own pre-existing internal nn.Dropout layers unrelated to this -- so we
    # check specifically for dropout attached to DiffusionUNetResnetBlock
    # instances, not a global nn.Dropout count.
    m = MonaiUNet1D(state_dim=2, obs_dim=2, hidden_channels=[16, 32], num_res_blocks=1,
                     norm_num_groups=8, dropout=0.0)
    assert _n_resblock_dropouts(m) == 0, "dropout=0.0 should attach no resblock dropout"


def test_monai_direct_unet_forwards_dropout():
    from models.monai_unet_adapter import MonaiDirectUNet
    md = MonaiDirectUNet(state_dim=4, hidden_channels=[16, 32], num_res_blocks=1,
                          norm_num_groups=8, dropout=0.5)
    assert _n_resblock_dropouts(md.monai_unet) > 0, \
        "MonaiDirectUNet must forward its dropout arg to MonaiUNet1D"
