import os
from types import SimpleNamespace

import pytest
import torch

monai = pytest.importorskip("monai")

from models.monai_unet_adapter import MonaiDirectUNet  # noqa: E402
from models.monai_unet_l96_2d import (  # noqa: E402
    MonaiDirectUNet2D,
    SectorCircularConv2d,
    from_sector_grid,
    to_sector_grid,
)

NO, J_OBS = 8, 2
D = NO + NO * J_OBS


def _small_model(**kw) -> MonaiDirectUNet2D:
    torch.manual_seed(0)
    m = MonaiDirectUNet2D(state_dim=D, NO=NO, hidden_channels=[16, 32], num_res_blocks=1,
                          norm_num_groups=8, dropout=0.0, use_flash_attention=False, **kw)
    for p in m.parameters():
        p.data.normal_(0.0, 0.05)
    return m.eval()


def _roll_sectors(v: torch.Tensor, shift: int) -> torch.Tensor:
    B, T, _ = v.shape
    X = v[..., :NO].roll(shift, -1)
    Y = v[..., NO:].reshape(B, T, NO, J_OBS).roll(shift, -2).reshape(B, T, NO * J_OBS)
    return torch.cat([X, Y], dim=-1)


def test_sector_grid_layout_round_trip_and_order():
    v = torch.randn(3, 20, D)
    g = to_sector_grid(v, NO)
    assert g.shape == (3, 1 + J_OBS, 20, NO)
    assert torch.equal(g[:, 0, :, 5], v[..., 5])
    assert torch.equal(g[:, 1 + 1, :, 5], v[..., NO + 5 * J_OBS + 1])
    assert torch.equal(from_sector_grid(g), v)


def test_forward_shape_and_isinstance_dispatch():
    m = _small_model()
    assert isinstance(m, MonaiDirectUNet)
    y = m(SimpleNamespace(obs=torch.randn(2, 64, D)))
    assert y.shape == (2, 64, D)


def test_nan_obs_treated_as_zero():
    m = _small_model()
    obs = torch.randn(2, 64, D)
    obs_nan = obs.clone()
    obs_nan[:, ::3, 4] = float("nan")
    obs_zero = obs.clone()
    obs_zero[:, ::3, 4] = 0.0
    with torch.no_grad():
        assert torch.allclose(m(SimpleNamespace(obs=obs_nan)), m(SimpleNamespace(obs=obs_zero)))


@pytest.mark.parametrize("mid_attention", ["time", "full", "none"])
@pytest.mark.parametrize("shift", [1, 3])
def test_exact_equivariance_to_cyclic_sector_shifts(mid_attention, shift):
    m = _small_model(mid_attention=mid_attention)
    obs = torch.randn(2, 64, D)
    with torch.no_grad():
        y = m(SimpleNamespace(obs=obs))
        y_shifted = m(SimpleNamespace(obs=_roll_sectors(obs, shift)))
    assert y.abs().max() > 1e-3
    assert torch.allclose(y_shifted, _roll_sectors(y, shift), atol=1e-5)


def test_convs_pad_circularly_in_sector_and_with_zeros_in_time():
    m = _small_model()
    convs = [c for c in m.modules() if isinstance(c, SectorCircularConv2d)]
    assert convs and all(type(c) is SectorCircularConv2d for c in m.modules() if isinstance(c, torch.nn.Conv2d))
    conv = SectorCircularConv2d(1, 1, 3, padding=1, bias=False)
    torch.nn.init.ones_(conv.weight)
    x = torch.zeros(1, 1, 6, NO)
    x[0, 0, -1, -1] = 1.0
    y = conv(x)
    assert y[0, 0, -1, 0] == 1.0
    assert y[0, 0, 0, :].abs().sum() == 0.0


def test_sector_axis_is_never_downsampled():
    m = _small_model()
    seen = []
    hooks = [blk.register_forward_hook(lambda _m, _i, out: seen.append(out[0].shape))
             for blk in m.unet2d.backbone.down_blocks]
    with torch.no_grad():
        m(SimpleNamespace(obs=torch.randn(1, 64, D)))
    for h in hooks:
        h.remove()
    assert [s[-1] for s in seen] == [NO] * len(seen)
    assert [s[-2] for s in seen] == [32, 32]


def test_rejects_scalar_conditioning():
    with pytest.raises(ValueError):
        MonaiDirectUNet2D(state_dim=D, NO=NO, hidden_channels=[16, 32], norm_num_groups=8,
                          cond_extra_dim=1)


def test_rejects_unknown_mid_attention():
    with pytest.raises(ValueError):
        _small_model(mid_attention="axial")


def test_gradient_reaches_parameters():
    torch.manual_seed(0)
    m = MonaiDirectUNet2D(state_dim=D, NO=NO, hidden_channels=[16, 32], num_res_blocks=1,
                          norm_num_groups=8, use_flash_attention=False)
    y = m(SimpleNamespace(obs=torch.randn(2, 64, D)))
    (y - 1.0).pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


@pytest.mark.parametrize("name,n_params", [
    ("L96B_directunet2d_monaiS", 2_032_851),
    ("L96B_directunet2d_monaiM", 5_634_483),
    ("L96B_directunet2d_monaiL", 22_500_963),
    ("L96B_directunet2d_monaiMwide", 14_406_531),
])
def test_experiment_configs_build_through_model_factory(name, n_params):
    from hydra import compose, initialize_config_dir
    from train import model_factory

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=f"experiment/{name}")
    assert cfg.model.model_type == "monai_direct_unet_2d"
    model = model_factory(cfg, torch.device("cpu"))
    assert isinstance(model, MonaiDirectUNet2D)
    assert sum(p.numel() for p in model.parameters()) == n_params
