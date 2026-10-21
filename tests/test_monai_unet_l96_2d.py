import os
from types import SimpleNamespace

import pytest
import torch

monai = pytest.importorskip("monai")

from models.monai_unet_adapter import MonaiDirectUNet  # noqa: E402
from models.monai_unet_l96_2d import (  # noqa: E402
    MonaiDirectUNet2D,
    MonaiUNet2DSectorSolver,
    MonaiUnconditionalPriorCFM2D,
    MonaiVanillaCFM2D,
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


def _small_solver_backbone(in_blocks: int, out_blocks: int = 1) -> MonaiUNet2DSectorSolver:
    torch.manual_seed(0)
    m = MonaiUNet2DSectorSolver(state_dim=in_blocks * D, block_dim=D, NO=NO,
                                output_dim=out_blocks * D, hidden_channels=[16, 32],
                                num_res_blocks=1, norm_num_groups=8, dropout=0.0,
                                use_flash_attention=False)
    for p in m.parameters():
        p.data.normal_(0.0, 0.05)
    return m.eval()


def _roll_blocks(x: torch.Tensor, n_blocks: int, shift: int) -> torch.Tensor:
    B, _, T = x.shape
    v = x.reshape(B, n_blocks, D, T).permute(0, 1, 3, 2).reshape(B * n_blocks, T, D)
    v = _roll_sectors(v, shift)
    return v.reshape(B, n_blocks, T, D).permute(0, 1, 3, 2).reshape(B, n_blocks * D, T)


def test_solver_backbone_block_layout_round_trip():
    m = _small_solver_backbone(4)
    x = torch.randn(2, 4 * D, 32)
    g = m._to_grid(x, 4)
    assert g.shape == (2, 4 * (1 + J_OBS), 32, NO)
    assert torch.equal(g[:, 3 * (1 + J_OBS), :, 5], x[:, 3 * D + 5])
    assert torch.equal(m._from_grid(g, 4), x)


@pytest.mark.parametrize("in_blocks,out_blocks", [(2, 1), (4, 1), (1, 1), (4, 2)])
def test_solver_backbone_shape_and_sector_equivariance(in_blocks, out_blocks):
    m = _small_solver_backbone(in_blocks, out_blocks)
    x = torch.randn(2, in_blocks * D, 64)
    tau = torch.full((2,), 0.3)
    with torch.no_grad():
        y = m(x, tau=tau)
        y_shifted = m(_roll_blocks(x, in_blocks, 3), tau=tau)
    assert y.shape == (2, out_blocks * D, 64)
    assert y.abs().max() > 1e-3
    assert torch.allclose(y_shifted, _roll_blocks(y, out_blocks, 3), atol=1e-5)


def test_solver_backbone_rejects_non_block_multiple():
    with pytest.raises(ValueError):
        MonaiUNet2DSectorSolver(state_dim=2 * D + 1, block_dim=D, NO=NO,
                                hidden_channels=[16, 32], norm_num_groups=8)


@pytest.mark.parametrize("update_input", ["obs+state", "subgrad+mask+state"])
def test_fourdvarnet_solver_with_sector_backbone_is_equivariant(update_input):
    from models.fourdvarnet import FourDVarNetSolver

    torch.manual_seed(0)
    model = FourDVarNetSolver(state_dim=D, hidden_channels=[16, 32], N_outer=2, dropout=0.0,
                              update_input=update_input, unet_backbone="monai_sector2d",
                              monai_norm_num_groups=8, monai_num_res_blocks=1,
                              aux_var_cost_weight=0.01 if "subgrad" in update_input else 0.0,
                              l96_NO=NO).eval()
    assert isinstance(model.unet, MonaiUNet2DSectorSolver)
    assert (model.prior_unet is not None) == ("subgrad" in update_input)
    for p in model.parameters():
        p.data.normal_(0.0, 0.05)
    obs = torch.randn(2, 64, D)
    obs[torch.rand(2, 64, D) < 0.7] = float("nan")
    mask = torch.ones(2, 64)
    with torch.no_grad():
        y = model(SimpleNamespace(obs=obs, obs_mask=mask))
        y_shifted = model(SimpleNamespace(obs=_roll_sectors(obs, 2), obs_mask=mask))
    assert y.shape == (2, 64, D)
    assert torch.allclose(y_shifted, _roll_sectors(y, 2), atol=1e-4)


def test_vanilla_cfm_2d_velocity_is_equivariant_and_samples():
    from models.vanilla_cfm import VanillaCFM

    torch.manual_seed(0)
    m = MonaiVanillaCFM2D(state_dim=D, NO=NO, hidden_channels=[16, 32], num_res_blocks=1,
                          norm_num_groups=8, dropout=0.0, use_flash_attention=False, N_outer=2)
    assert isinstance(m, VanillaCFM)
    for p in m.parameters():
        p.data.normal_(0.0, 0.05)
    m.eval()
    obs = torch.randn(2, 64, D)
    obs[torch.rand(2, 64, D) < 0.7] = float("nan")
    x = torch.randn(2, 64, D)
    tau = torch.full((2,), 0.4)
    with torch.no_grad():
        v = m(x, SimpleNamespace(obs=obs, forcing=None, params=None), tau)
        v_shifted = m(_roll_sectors(x, 3), SimpleNamespace(obs=_roll_sectors(obs, 3), forcing=None,
                                                           params=None), tau)
        sample = m.sample(SimpleNamespace(obs=obs, forcing=None, params=None))
    assert v.shape == sample.shape == (2, 64, D)
    assert torch.allclose(v_shifted, _roll_sectors(v, 3), atol=1e-5)
    batch = SimpleNamespace(obs=obs, forcing=None, params=None, states=torch.randn(2, 64, D))
    assert torch.isfinite(m.compute_cfm_loss(batch))


def test_sda1_2d_prior_is_equivariant_and_drives_the_guided_sampler():
    from evaluation.sda_sampler import sda_guided_sample
    from models.sda import UnconditionalPriorCFM

    torch.manual_seed(0)
    m = MonaiUnconditionalPriorCFM2D(state_dim=D, NO=NO, hidden_channels=[16, 32], num_res_blocks=1,
                                     norm_num_groups=8, dropout=0.0, use_flash_attention=False,
                                     N_outer=2)
    assert isinstance(m, UnconditionalPriorCFM)
    for p in m.parameters():
        p.data.normal_(0.0, 0.05)
    m.eval()
    x = torch.randn(2, 64, D)
    tau = torch.full((2,), 0.4)
    with torch.no_grad():
        v = m(x, None, tau)
        v_shifted = m(_roll_sectors(x, 3), None, tau)
    assert v.shape == (2, 64, D)
    assert torch.allclose(v_shifted, _roll_sectors(v, 3), atol=1e-5)
    obs_mask = torch.zeros(2, 64, dtype=torch.bool)
    obs_mask[:, ::8] = True
    obs = torch.where(obs_mask[..., None], torch.randn(2, 64, D), torch.tensor(float("nan")))
    batch = SimpleNamespace(obs=obs, obs_mask=obs_mask, forcing=None, params=None,
                            states=torch.randn(2, 64, D))
    assert torch.isfinite(m.compute_cfm_loss(batch))
    pred, _ = sda_guided_sample(m, batch, R_var=0.5, N_outer=2, guidance_weight=2.0)
    assert pred.shape == (2, 64, D) and torch.isfinite(pred).all()


@pytest.mark.parametrize("name,cls,n_params", [
    ("L96B_vanillacfm_monai2dS_ep1200", MonaiVanillaCFM2D, 2_033_499),
    ("L96B_vanillacfm_monai2dM_ep1200", MonaiVanillaCFM2D, 5_635_563),
    ("L96B_vanillacfm_monai2dL_ep1200", MonaiVanillaCFM2D, 22_503_123),
    ("L96B_sda1_monai2dS_ep1200", MonaiUnconditionalPriorCFM2D, 2_032_851),
    ("L96B_sda1_monai2dM_ep1200", MonaiUnconditionalPriorCFM2D, 5_634_483),
    ("L96B_sda1_monai2dL_ep1200", MonaiUnconditionalPriorCFM2D, 22_500_963),
])
def test_cfm_2d_configs_build_through_model_factory(name, cls, n_params):
    from hydra import compose, initialize_config_dir
    from train import model_factory

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=f"experiment/{name}")
    model = model_factory(cfg, torch.device("cpu"))
    assert isinstance(model, cls)
    assert sum(p.numel() for p in model.parameters()) == n_params


@pytest.mark.parametrize("name,n_unet,n_prior", [
    ("L96B_fdv1_monai2dM", 5_635_563, 0),
    ("L96B_fdvsubgradmask_monai2dM_aux01det_frozen_philr01", 5_637_723, 5_634_483),
])
def test_fdv_2d_configs_build_through_model_factory(name, n_unet, n_prior):
    from hydra import compose, initialize_config_dir
    from train import model_factory

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=f"experiment/{name}")
    model = model_factory(cfg, torch.device("cpu"))
    assert isinstance(model.unet, MonaiUNet2DSectorSolver)
    assert sum(p.numel() for p in model.unet.parameters()) == n_unet
    n = sum(p.numel() for p in model.prior_unet.parameters()) if model.prior_unet is not None else 0
    assert n == n_prior
