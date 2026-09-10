import pytest
import torch

monai = pytest.importorskip("monai")

from models.monai_unet_qg2d import MonaiDirectUNetQG  # noqa: E402


class _FakeBatch:
    pass


def _batch(B, T, ny, nx, cond_extra_dim=0, param_dim=0):
    D = 2 * ny * nx
    b = _FakeBatch()
    b.obs = torch.randn(B, T, D)
    b.forcing = torch.zeros(B, T)
    b.params = torch.zeros(B, param_dim)
    return b


def test_forward_shape():
    ny = nx = 16
    m = MonaiDirectUNetQG(ny=ny, nx=nx, nlayers=2, hidden_channels=[8, 16])
    b = _batch(2, 3, ny, nx)
    out = m(b)
    assert out.shape == (2, 3, 2 * ny * nx)


def test_nan_obs_handled():
    ny = nx = 16
    m = MonaiDirectUNetQG(ny=ny, nx=nx, nlayers=2, hidden_channels=[8, 16])
    b = _batch(1, 2, ny, nx)
    b.obs[0, 0, 0] = float("nan")
    out = m(b)
    assert torch.isfinite(out).all()


def test_circular_shift_equivariance():
    """The whole point of this model over `models.direct_unet.DirectUNet`
    (which never convolves spatially at all) is respecting the QG domain's
    doubly-periodic (ny, nx) grid: shifting the input by a multiple of the
    backbone's total downsampling factor (2 pool stages -> 4) must shift the
    output identically, in both x and y."""
    torch.manual_seed(0)
    ny = nx = 16
    m = MonaiDirectUNetQG(ny=ny, nx=nx, nlayers=2, hidden_channels=[8, 16])
    # Break MONAI's zero-initialized output conv (standard diffusion-model
    # init) so the output is a real, non-degenerate test signal.
    final = m.unet.backbone.out[2].conv
    torch.nn.init.normal_(final.weight, std=0.05)
    torch.nn.init.normal_(final.bias, std=0.05)
    m.eval()

    B, T = 1, 2
    obs = torch.randn(B, T, 2 * ny * nx)
    b = _FakeBatch()
    b.obs = obs
    b.forcing = torch.zeros(B, T)
    b.params = torch.zeros(B, 0)

    shift = 4
    with torch.no_grad():
        out1 = m(b).reshape(B * T, 2, ny, nx)

        obs_grid = obs.reshape(B * T, 2, ny, nx)
        b2 = _FakeBatch()
        b2.obs = torch.roll(obs_grid, shifts=shift, dims=-1).reshape(B, T, -1)
        b2.forcing = b.forcing
        b2.params = b.params
        out2 = m(b2).reshape(B * T, 2, ny, nx)
        diff_x = (torch.roll(out1, shifts=shift, dims=-1) - out2).abs().max()
        assert diff_x < 1e-4

        b3 = _FakeBatch()
        b3.obs = torch.roll(obs_grid, shifts=shift, dims=-2).reshape(B, T, -1)
        b3.forcing = b.forcing
        b3.params = b.params
        out3 = m(b3).reshape(B * T, 2, ny, nx)
        diff_y = (torch.roll(out1, shifts=shift, dims=-2) - out3).abs().max()
        assert diff_y < 1e-4
