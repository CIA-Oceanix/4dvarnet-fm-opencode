import pytest
import torch

monai = pytest.importorskip("monai")

from models.monai_unet_qg2d import (  # noqa: E402
    MonaiDirectUNetQG,
    MonaiDirectUNetQGChannelTime,
    MonaiUNet2DQGSolver,
)


class _FakeBatch:
    pass


def _batch(B, T, ny, nx, cond_extra_dim=0, param_dim=0, ic_dim=0):
    D = 2 * ny * nx
    b = _FakeBatch()
    b.obs = torch.randn(B, T, D)
    # forcing is always a (B, T, ny, nx) spatial field (the wind-curl map,
    # see data.qg_neural's Q3/Q4 conditioning docstring) -- zeros when
    # cond_extra_dim=0 since forward() never reads it in that case.
    b.forcing = torch.zeros(B, T, ny, nx)
    b.params = torch.zeros(B, param_dim)
    # ic (Q5) is one static (B, ic_dim*ny*nx) field per window -- zeros when
    # ic_dim=0 since forward() never reads it in that case.
    b.ic = torch.zeros(B, ic_dim * ny * nx)
    return b


def test_forward_with_forcing_and_params_conditioning():
    """The forward pass actually uses forcing/params when cond_extra_dim/
    param_dim > 0, not just accepting the shape: zeroing them out must
    change the output vs. real conditioning (sanity-checks Q3/Q4's
    conditioning is wired, not dead)."""
    ny = nx = 8
    days = 3
    model = MonaiDirectUNetQG(ny=ny, nx=nx, nlayers=2, param_dim=4, cond_extra_dim=1,
                              hidden_channels=[8, 16])
    # MONAI's DiffusionModelUNet zero-initializes its output conv (standard
    # diffusion-model init, see test_circular_shift_equivariance above) --
    # break it so the output isn't trivially all-zero regardless of input.
    final = model.unet.backbone.out[2].conv
    torch.nn.init.normal_(final.weight, std=0.05)
    torch.nn.init.normal_(final.bias, std=0.05)
    model.eval()

    b = _FakeBatch()
    b.obs = torch.randn(1, days, 2 * ny * nx)
    b.forcing = torch.randn(1, days, ny, nx)
    b.params = torch.randn(1, 4)
    out = model(b)
    assert out.shape == (1, days, 2 * ny * nx)
    assert torch.isfinite(out).all()

    b_zero = _FakeBatch()
    b_zero.obs = b.obs
    b_zero.forcing = torch.zeros_like(b.forcing)
    b_zero.params = torch.zeros_like(b.params)
    out_zero = model(b_zero)
    assert not torch.allclose(out, out_zero)


def test_forward_with_ic_conditioning():
    """Q5: the forward pass actually uses `batch.ic` when ic_dim > 0 (zeroing
    it must change the output), and the SAME static IC field is broadcast
    identically across all T days (unlike forcing, which varies per day)."""
    ny = nx = 8
    days = 3
    model = MonaiDirectUNetQG(ny=ny, nx=nx, nlayers=2, ic_dim=2, hidden_channels=[8, 16])
    final = model.unet.backbone.out[2].conv
    torch.nn.init.normal_(final.weight, std=0.05)
    torch.nn.init.normal_(final.bias, std=0.05)
    model.eval()

    b = _FakeBatch()
    b.obs = torch.randn(1, days, 2 * ny * nx)
    b.forcing = torch.zeros(1, days, ny, nx)
    b.params = torch.zeros(1, 0)
    b.ic = torch.randn(1, 2 * ny * nx)
    out = model(b)
    assert out.shape == (1, days, 2 * ny * nx)
    assert torch.isfinite(out).all()

    b_zero = _FakeBatch()
    b_zero.obs = b.obs
    b_zero.forcing = b.forcing
    b_zero.params = b.params
    b_zero.ic = torch.zeros_like(b.ic)
    out_zero = model(b_zero)
    assert not torch.allclose(out, out_zero)


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


def test_monai2d_qg_solver_merges_days_into_channels_not_batch():
    """MonaiUNet2DQGSolver (unet_backbone="monai2d" for FourDVarNetSolver):
    merges T (days) into the channel axis, keeping every day's true (ny,
    nx) grid -- unlike MonaiDirectUNetQG (T folded into batch, fully
    independent per-day processing). forward(x, tau) takes/returns
    (B, C, T) (MonaiUNet1D's own drop-in convention), C a whole multiple
    of ny*nx."""
    B, T, nlayers, ny, nx = 2, 5, 2, 8, 8
    state_dim = nlayers * ny * nx
    model = MonaiUNet2DQGSolver(state_dim=2 * state_dim, T=T, ny=ny, nx=nx,
                                hidden_channels=[8, 16], num_res_blocks=1,
                                norm_num_groups=4, output_dim=state_dim,
                                dropout=0.1)
    x = torch.randn(B, 2 * state_dim, T)
    tau = torch.zeros(B)
    out = model(x, tau=tau)
    assert out.shape == (B, state_dim, T)
    assert torch.isfinite(out).all()
    out.pow(2).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_monai2d_qg_solver_wrong_t_raises():
    model = MonaiUNet2DQGSolver(state_dim=8, T=5, ny=2, nx=2, hidden_channels=[8, 16],
                                num_res_blocks=1, norm_num_groups=4)
    x = torch.randn(1, 8, 3)  # T=3, but model expects T=5
    with pytest.raises(ValueError, match="expected T"):
        model(x)


def test_monai2d_qg_solver_state_dim_not_multiple_of_grid_raises():
    with pytest.raises(ValueError, match="whole multiple"):
        MonaiUNet2DQGSolver(state_dim=10, T=5, ny=4, nx=4, hidden_channels=[8, 16])


def test_direct_unet_channeltime_forward_shape_and_grad():
    """MonaiDirectUNetQGChannelTime (Q7): DirectUNet-style single-pass
    regression, but merges T into channels (via MonaiUNet2DQGSolver)
    instead of MonaiDirectUNetQG's batch-folding. Obs-only forward(batch)
    contract, same shape convention as MonaiDirectUNetQG."""
    ny = nx = 8
    T = 5
    nlayers = 2
    D = nlayers * ny * nx
    model = MonaiDirectUNetQGChannelTime(ny=ny, nx=nx, T=T, nlayers=nlayers,
                                         hidden_channels=[8, 16], num_res_blocks=1,
                                         norm_num_groups=4)
    b = _FakeBatch()
    b.obs = torch.randn(2, T, D)
    out = model(b)
    assert out.shape == (2, T, D)
    assert torch.isfinite(out).all()
    out.pow(2).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_direct_unet_channeltime_nan_obs_handled():
    ny = nx = 8
    T = 5
    model = MonaiDirectUNetQGChannelTime(ny=ny, nx=nx, T=T, nlayers=2,
                                         hidden_channels=[8, 16], num_res_blocks=1,
                                         norm_num_groups=4)
    b = _FakeBatch()
    b.obs = torch.randn(1, T, 2 * ny * nx)
    b.obs[0, 0, 0] = float("nan")
    out = model(b)
    assert torch.isfinite(out).all()


def test_direct_unet_channeltime_wrong_t_raises():
    model = MonaiDirectUNetQGChannelTime(ny=8, nx=8, T=5, nlayers=2,
                                         hidden_channels=[8, 16], num_res_blocks=1,
                                         norm_num_groups=4)
    b = _FakeBatch()
    b.obs = torch.randn(1, 3, 2 * 8 * 8)  # T=3, but model expects T=5
    with pytest.raises(ValueError, match="expected T"):
        model(b)


def test_direct_unet_channeltime_forcing_and_params_conditioning_changes_output():
    """Q8/Q9 (T-channels retrains of Q3/Q4): forcing/params conditioning is
    actually wired into the per-day channel block before T merges into
    channels, not just accepted-and-ignored -- zeroing them must change the
    output (mirrors MonaiDirectUNetQG's own test of the same property)."""
    ny = nx = 8
    T = 3
    model = MonaiDirectUNetQGChannelTime(ny=ny, nx=nx, T=T, nlayers=2, param_dim=4,
                                         cond_extra_dim=1, hidden_channels=[8, 16],
                                         num_res_blocks=1, norm_num_groups=2)
    final = model.unet.backbone2d.backbone.out[2].conv
    torch.nn.init.normal_(final.weight, std=0.05)
    torch.nn.init.normal_(final.bias, std=0.05)
    model.eval()

    b = _FakeBatch()
    b.obs = torch.randn(1, T, 2 * ny * nx)
    b.forcing = torch.randn(1, T, ny, nx)
    b.params = torch.randn(1, 4)
    out = model(b)
    assert out.shape == (1, T, 2 * ny * nx)
    assert torch.isfinite(out).all()

    b_zero = _FakeBatch()
    b_zero.obs = b.obs
    b_zero.forcing = torch.zeros_like(b.forcing)
    b_zero.params = torch.zeros_like(b.params)
    out_zero = model(b_zero)
    assert not torch.allclose(out, out_zero)


def test_direct_unet_channeltime_ic_conditioning_changes_output():
    """Q10 (T-channels retrain of Q5): the static per-window IC field is
    broadcast identically across all T days and actually used."""
    ny = nx = 8
    T = 3
    model = MonaiDirectUNetQGChannelTime(ny=ny, nx=nx, T=T, nlayers=2, ic_dim=2,
                                         hidden_channels=[8, 16], num_res_blocks=1,
                                         norm_num_groups=2)
    final = model.unet.backbone2d.backbone.out[2].conv
    torch.nn.init.normal_(final.weight, std=0.05)
    torch.nn.init.normal_(final.bias, std=0.05)
    model.eval()

    b = _FakeBatch()
    b.obs = torch.randn(1, T, 2 * ny * nx)
    b.ic = torch.randn(1, 2 * ny * nx)
    out = model(b)
    assert out.shape == (1, T, 2 * ny * nx)
    assert torch.isfinite(out).all()

    b_zero = _FakeBatch()
    b_zero.obs = b.obs
    b_zero.ic = torch.zeros_like(b.ic)
    out_zero = model(b_zero)
    assert not torch.allclose(out, out_zero)


def test_direct_unet_channeltime_all_conditioning_combined_shape_and_grad():
    """Q10's exact combination (forcing + params + ic all at once) --
    checks the concatenation order/shapes compose without error and the
    MONAI norm_num_groups divisibility constraint is satisfiable."""
    ny = nx = 8
    T = 5
    nlayers = 2
    D = nlayers * ny * nx
    model = MonaiDirectUNetQGChannelTime(ny=ny, nx=nx, T=T, nlayers=nlayers,
                                         param_dim=3, cond_extra_dim=1, ic_dim=2,
                                         hidden_channels=[8, 16], num_res_blocks=1,
                                         norm_num_groups=2)
    b = _FakeBatch()
    b.obs = torch.randn(2, T, D)
    b.forcing = torch.randn(2, T, ny, nx)
    b.params = torch.randn(2, 3)
    b.ic = torch.randn(2, 2 * ny * nx)
    out = model(b)
    assert out.shape == (2, T, D)
    assert torch.isfinite(out).all()
    out.pow(2).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
