import numpy as np
import pytest
import torch

from data.dataloader import FlowMatchingBatch
from data.lorenz96 import (
    Lorenz96Config,
    RandomParamLorenz96Dataset,
    _make_lorenz96_dynamics,
)
from evaluation.run_l96 import make_obs_j_indices
from models.direct_unet import JointDirectUNet, ParamHeadUNet
from models.vanilla_cfm import JointCFM, JointCFMCoupled, ParamFlowUNet
from train import _make_eval_batch, evaluate_model, make_l96_dataloaders, model_factory

PARAM_NAMES = ("F", "c1", "hx", "eps", "w1", "w2", "w3", "w4")
SD, PD = 24, 8


@pytest.fixture
def l96_joint_cfg():
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    return Lorenz96Config(
        T_max=0.1, dt=0.001, obs_interval=20,
        num_windows=2, spinup_steps=500, seed=42, param_bias=0.0,
        obs_var_indices=obs_var_indices,
    )


@pytest.fixture
def l96_joint_dataset(l96_joint_cfg):
    dyn = _make_lorenz96_dynamics(l96_joint_cfg)
    return RandomParamLorenz96Dataset(l96_joint_cfg, param_noise=0.2, dynamics=dyn)


def _joint_batch(w):
    states = w["true_state"][:, make_obs_j_indices(8, 4, 2)].unsqueeze(0)
    obs = w["obs"].unsqueeze(0)
    mask = w["obs_mask"].unsqueeze(0)
    forcing = w["forcing_corrupted"].unsqueeze(0)
    params = torch.tensor([[float(w[nm]) for nm in PARAM_NAMES]])
    true_params = torch.tensor([[float(w[f"true_{nm}"]) for nm in PARAM_NAMES]])
    return FlowMatchingBatch(states, obs, mask, forcing,
                             params=params, true_params=true_params)


def test_joint_cfm_l96_shapes(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                     train_tau_0_only=True)
    batch = _joint_batch(w)
    loss = model.compute_cfm_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    x, p = model.sample(batch, return_params=True)
    assert x.shape == (1, w["true_state"].shape[0], SD)
    assert p.shape == (1, PD)
    assert model.cond_extra_dim == 1
    assert model.unet.output_dim == SD
    assert model.param_flow.param_dim == PD


def test_joint_cfm_param_normalization(l96_joint_dataset):
    w = l96_joint_dataset[0]
    ref = torch.tensor([8.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 0.1])
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                     param_ref=ref.tolist())
    batch = _joint_batch(w)
    true_vec = batch.true_params
    norm = model._norm(true_vec)
    assert torch.allclose(
        norm,
        (true_vec - ref.unsqueeze(0)) / (0.2 * ref.unsqueeze(0)),
        atol=1e-5,
    )
    assert torch.allclose(model._denorm(norm), true_vec, atol=1e-5)
    assert torch.allclose(model.param_ref, ref)
    assert torch.allclose(model.param_scale, 0.2 * ref)
    loss = model.compute_cfm_loss(batch)
    assert torch.isfinite(loss) and loss.ndim == 0
    _, p = model.sample(batch, N_outer=3, return_params=True)
    assert p.shape == (1, PD)
    ref_diff = torch.abs(p - ref.unsqueeze(0))
    assert torch.all(ref_diff / ref.unsqueeze(0) < 3.0), (
        f"denormalized sample {p} too far from reference {ref}")


def test_joint_cfm_default_param_ref_is_ones(l96_joint_dataset):
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    assert torch.allclose(model.param_ref, torch.ones(PD))
    assert torch.allclose(model.param_scale, 0.2 * torch.ones(PD))


def test_joint_cfm_oracle_gone(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    model.eval()
    batch = _joint_batch(w)
    x_t = torch.randn_like(batch.states)
    tau = torch.tensor([0.3])
    param_tau = torch.randn(1, PD)
    v_state_a, v_param_a, _ = model.forward(x_t, batch, tau, param_tau)
    batch_b = FlowMatchingBatch(
        batch.states, batch.obs, batch.obs_mask, batch.forcing,
        params=torch.randn_like(batch.params), true_params=batch.true_params)
    v_state_b, v_param_b, _ = model.forward(x_t, batch_b, tau, param_tau)
    assert torch.allclose(v_state_a, v_state_b, atol=1e-6)
    assert torch.allclose(v_param_a, v_param_b, atol=1e-6)


def test_joint_cfm_param_flow_oracle_gone_at_inference(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16], N_outer=10)
    model.eval()
    batch = _joint_batch(w)
    torch.manual_seed(0)
    with torch.no_grad():
        _, p_a = model.sample(batch, N_outer=10, return_params=True)
    batch_b = FlowMatchingBatch(
        batch.states, batch.obs, batch.obs_mask, batch.forcing,
        params=batch.params, true_params=torch.randn_like(batch.true_params))
    torch.manual_seed(0)
    with torch.no_grad():
        _, p_b = model.sample(batch_b, N_outer=10, return_params=True)
    assert torch.allclose(p_a, p_b, atol=1e-6)


def test_joint_cfm_param_flow_target(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    batch = _joint_batch(w)
    tau = torch.tensor([0.5])
    param_tau = torch.randn(1, PD)
    _, v_param, _ = model.forward(torch.randn_like(batch.states), batch, tau, param_tau)
    assert v_param.shape == (1, PD)
    assert torch.isfinite(v_param).all()


def test_joint_cfm_stop_grad_xhat(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    batch = _joint_batch(w)
    tau = torch.tensor([0.3])
    param_tau = torch.randn(1, PD, requires_grad=False)
    x_t = torch.randn_like(batch.states)
    _, v_param, _ = model.forward(x_t, batch, tau, param_tau)
    loss = v_param.sum()
    loss.backward()
    state_grads = [p.grad for n, p in model.unet.named_parameters()]
    assert all(g is None for g in state_grads), "state UNet must not receive param-flow grads (stop_grad on x_hat_1)"
    param_flow_grads = [p.grad for n, p in model.param_flow.named_parameters()]
    assert any(g is not None for g in param_flow_grads)


def test_joint_cfm_param_flow_conditions_on_passed_param(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    model.eval()
    batch = _joint_batch(w)
    x_t = torch.randn_like(batch.states)
    tau = torch.tensor([1.0])
    param_tau = torch.randn(1, PD)
    with torch.no_grad():
        _, v_param, x_hat = model.forward(x_t, batch, tau, param_tau)
    assert torch.allclose(x_hat, x_t, atol=1e-6)
    assert v_param.shape == (1, PD)
    assert torch.isfinite(v_param).all()


def test_joint_direct_unet_l96_shapes(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    batch = _joint_batch(w)
    loss = model.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    x, p = model.sample(batch, return_params=True)
    assert x.shape == (1, w["true_state"].shape[0], SD)
    assert p.shape == (1, PD)
    assert model.cond_extra_dim == 1
    assert model.unet.output_dim == SD
    assert model.param_head.param_dim == PD


def test_joint_direct_unet_oracle_gone(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    model.eval()
    batch = _joint_batch(w)
    v_state_a, _ = model.forward(batch)
    batch_b = FlowMatchingBatch(
        batch.states, batch.obs, batch.obs_mask, batch.forcing,
        params=torch.randn_like(batch.params), true_params=batch.true_params)
    v_state_b, _ = model.forward(batch_b)
    assert torch.allclose(v_state_a, v_state_b, atol=1e-6)


def test_joint_direct_unet_param_head_stop_grad(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    batch = _joint_batch(w)
    _, params = model.forward(batch)
    params.sum().backward()
    state_grads = [p.grad for n, p in model.unet.named_parameters()]
    assert all(g is None for g in state_grads), "state UNet must not receive param-head grads (detach on x_hat)"
    param_grads = [p.grad for n, p in model.param_head.named_parameters()]
    assert any(g is not None for g in param_grads)


def test_joint_models_use_true_params(l96_joint_dataset):
    w = l96_joint_dataset[0]
    batch = _joint_batch(w)
    true_vec = torch.tensor([[float(w[f"true_{nm}"]) for nm in PARAM_NAMES]])

    jcfm = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                    train_tau_0_only=True)
    _, p_cfm = jcfm.sample(batch, return_params=True)
    assert p_cfm.shape == true_vec.shape
    assert torch.isfinite(p_cfm).all()

    jdu = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    _, p_du = jdu.sample(batch, return_params=True)
    assert p_du.shape == true_vec.shape
    assert torch.isfinite(p_du).all()


def test_joint_dataloader_with_params(l96_joint_cfg):
    dyn = _make_lorenz96_dynamics(l96_joint_cfg)
    cfg_val = Lorenz96Config(**{**l96_joint_cfg.__dict__, "seed": 99})
    datasets = {
        "train": RandomParamLorenz96Dataset(l96_joint_cfg, param_noise=0.2, dynamics=dyn),
        "val": RandomParamLorenz96Dataset(cfg_val, param_noise=0.2, dynamics=dyn),
    }
    loaders = make_l96_dataloaders(
        datasets, batch_size=1, obs_interval=20, R_var=0.5,
        param_names=PARAM_NAMES, with_params=True,
        obs_var_indices=l96_joint_cfg.obs_var_indices,
    )
    b = next(iter(loaders["train"]))
    assert b.params is not None and b.true_params is not None
    assert b.params.shape[1] == PD
    assert b.true_params.shape[1] == PD


def test_make_eval_batch_l96_joint(l96_joint_dataset):
    w = l96_joint_dataset[0]
    device = torch.device("cpu")
    b = _make_eval_batch(w, device, param_names=PARAM_NAMES, param_dim=PD)
    assert b.params.shape == (1, PD)
    assert b.true_params.shape == (1, PD)
    for i, nm in enumerate(PARAM_NAMES):
        assert abs(float(b.true_params[0, i]) - w[f"true_{nm}"]) < 1e-6


def test_evaluate_model_joint_returns_param_rmse(l96_joint_dataset):
    device = torch.device("cpu")
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    model.eval()
    m, _, prmse = evaluate_model(model, l96_joint_dataset, device,
                                 model_type="joint_direct_unet", return_params=True,
                                 param_names=PARAM_NAMES, param_dim=PD,
                                 obs_var_indices=make_obs_j_indices(8, 4, 2))
    assert m.shape == (SD,)
    assert prmse.shape == (PD,)
    assert np.all(np.isfinite(prmse))


def test_model_factory_joint_l96(l96_joint_cfg):
    from hydra import compose, initialize

    def build(name):
        with initialize(version_base="1.3", config_path="../config"):
            cfg = compose(config_name="experiment/" + name)
        return model_factory(cfg, torch.device("cpu"))

    for name, cls in [("L7_joint_cfm_s0s1", JointCFM),
                      ("L8_joint_direct_unet_s0s1", JointDirectUNet),
                      ("L9_joint_cfm_s0s1_multitau", JointCFM)]:
        model = build(name)
        assert isinstance(model, cls)
        assert model.param_dim == PD


def test_litmodel_joint_training_step(l96_joint_dataset):
    from training.lightning_module import LitModel

    w = l96_joint_dataset[0]
    for cls, mtype, extra in [
        (JointCFM, "joint_cfm", {"train_tau_0_only": True}),
        (JointDirectUNet, "joint_direct_unet", {}),
    ]:
        model = cls(state_dim=SD, param_dim=PD, hidden_channels=[8, 16], **extra)
        lit = LitModel(model, model_type=mtype, stage=1)
        batch = _joint_batch(w)
        loss = lit._forward_and_loss(batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        grad_ok = any(p.grad is not None for p in model.parameters())
        assert grad_ok, f"{mtype} produced no gradients"


def test_param_flow_cnn_attn_pool_shape(l96_joint_dataset):
    from models.vanilla_cfm import ParamFlowCNN

    w = l96_joint_dataset[0]
    batch = _joint_batch(w)
    B, T = 1, batch.obs.shape[1]
    obs = batch.obs
    forcing = batch.forcing
    x_hat = torch.randn(B, T, SD)
    param_tau = torch.randn(B, T, PD)
    tau = torch.zeros(B)
    for pool in ("mean", "attn"):
        model = ParamFlowCNN(param_dim=PD, state_dim=SD, hidden_channels=[8, 16],
                             time_emb_dim=16, pool=pool)
        out = model(obs, forcing, x_hat, param_tau, tau)
        assert out.shape == (B, PD), f"{pool} pool output shape {out.shape}"
        assert torch.isfinite(out).all()
    attn = ParamFlowCNN(param_dim=PD, state_dim=SD, hidden_channels=[8, 16],
                        time_emb_dim=16, pool="attn")
    assert hasattr(attn, "attn_pool")


def test_param_head_cnn_attn_pool_shape(l96_joint_dataset):
    from models.direct_unet import ParamHeadCNN

    w = l96_joint_dataset[0]
    batch = _joint_batch(w)
    B, T = 1, batch.obs.shape[1]
    obs = batch.obs
    forcing = batch.forcing
    x_hat = torch.randn(B, T, SD)
    for pool in ("mean", "attn"):
        model = ParamHeadCNN(param_dim=PD, state_dim=SD, hidden_channels=[8, 16],
                             pool=pool)
        out = model(obs, forcing, x_hat)
        assert out.shape == (B, PD)
        assert torch.isfinite(out).all()
    attn = ParamHeadCNN(param_dim=PD, state_dim=SD, hidden_channels=[8, 16], pool="attn")
    assert hasattr(attn, "attn_pool")


def test_joint_direct_unet_normalization_roundtrip(l96_joint_dataset):
    w = l96_joint_dataset[0]
    ref = torch.tensor([8.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 0.1])
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_ref=ref.tolist())
    batch = _joint_batch(w)
    true_vec = batch.true_params
    norm = model._norm(true_vec)
    assert torch.allclose(model._denorm(norm), true_vec, atol=1e-5)
    assert torch.allclose(model.param_ref, ref)
    assert torch.allclose(model.param_scale, 0.2 * ref)
    loss = model.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    _, p = model.sample(batch, return_params=True)
    assert p.shape == (1, PD)
    assert torch.isfinite(p).all()


def test_joint_direct_unet_default_param_ref_is_ones(l96_joint_dataset):
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    assert torch.allclose(model.param_ref, torch.ones(PD))
    assert torch.allclose(model.param_scale, 0.2 * torch.ones(PD))


def test_joint_models_stage2_param_only_loss_and_freeze(l96_joint_dataset):
    from training.lightning_module import LitModel

    w = l96_joint_dataset[0]
    batch = _joint_batch(w)
    for cls, mtype, extra in [
        (JointCFM, "joint_cfm", {"train_tau_0_only": True}),
        (JointDirectUNet, "joint_direct_unet", {}),
    ]:
        model = cls(state_dim=SD, param_dim=PD, hidden_channels=[8, 16], **extra)
        lit = LitModel(model, model_type=mtype, stage=2)
        lit.on_train_start()
        unet_trainable = any(p.requires_grad for p in model.unet.parameters())
        assert not unet_trainable, f"{mtype}: stage-2 state UNet should be frozen"
        loss = lit._forward_and_loss(batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        unet_grads = [p.grad for p in model.unet.parameters() if p.grad is not None]
        assert len(unet_grads) == 0, f"{mtype}: frozen UNet got gradients"


def test_joint_cfm_stage2_param_loss_conditions_on_real_state(l96_joint_dataset):
    """Stage-2 compute_param_loss must condition the param flow on the real
    sampled state path (``x_tau = mix(x0, states, tau)``), not a degenerate zero
    state. Under the old bug the param-flow gradient was invariant to
    ``batch.states``; with the fix it must differ when the true state changes."""
    w = l96_joint_dataset[0]
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    batch_a = _joint_batch(w)
    batch_b = FlowMatchingBatch(
        batch_a.states + 1.0, batch_a.obs, batch_a.obs_mask, batch_a.forcing,
        params=batch_a.params, true_params=batch_a.true_params)

    def param_grad(batch):
        model.zero_grad()
        torch.manual_seed(1234)
        loss = model.compute_param_loss(batch)
        loss.backward()
        return [p.grad.clone() for p in model.param_flow.parameters()
                if p.grad is not None]

    ga = param_grad(batch_a)
    gb = param_grad(batch_b)
    assert len(ga) > 0
    diff = sum(torch.abs(g1 - g2).sum() for g1, g2 in zip(ga, gb))
    assert diff > 0, "stage-2 param flow must be conditioned on the real state"


def test_joint_cfm_stage2_param_loss_real_state_finite(l96_joint_dataset):
    from training.lightning_module import LitModel

    w = l96_joint_dataset[0]
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16])
    lit = LitModel(model, model_type="joint_cfm", stage=2)
    lit.on_train_start()
    loss = lit._forward_and_loss(_joint_batch(w))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    unet_grads = [p.grad for p in model.unet.parameters() if p.grad is not None]
    assert len(unet_grads) == 0, "frozen UNet got gradients in stage-2 param loss"


def test_joint_cfm_sample_params_from_state(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFM(state_dim=SD, param_dim=PD, hidden_channels=[8, 16], N_outer=3)
    model.eval()
    batch = _joint_batch(w)
    x_hat = batch.states.clone()
    with torch.no_grad():
        p = model.sample_params_from_state(batch, x_hat, N_outer=3)
    assert p.shape == (1, PD)
    assert torch.isfinite(p).all()


def test_joint_cfm_coupled_shapes(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFMCoupled(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_flow_channels=[4, 8], N_outer=3)
    batch = _joint_batch(w)
    loss = model.compute_cfm_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    x, p = model.sample(batch, N_outer=3, return_params=True)
    assert x.shape == (1, w["true_state"].shape[0], SD)
    assert p.shape == (1, PD)
    assert model.unet.output_dim == SD
    assert model.unet.cond_encoder.proj.in_features == 2 * SD + 1 + PD
    assert isinstance(model.param_flow, ParamFlowUNet)
    assert model.param_flow.param_dim == PD


def test_joint_cfm_coupled_oracle_gone(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFMCoupled(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_flow_channels=[4, 8])
    model.eval()
    batch = _joint_batch(w)
    x_t = torch.randn_like(batch.states)
    tau = torch.tensor([0.3])
    theta_tau = torch.randn(1, PD)
    v_state_a, v_param_a = model.forward(x_t, batch, tau, theta_tau)
    # biased `params` (batch.params) are never conditioning; true params enter
    # only as the CFM target theta_1. Shuffle both -> outputs unchanged.
    batch_b = FlowMatchingBatch(
        batch.states, batch.obs, batch.obs_mask, batch.forcing,
        params=torch.randn_like(batch.params), true_params=torch.randn_like(batch.true_params))
    v_state_b, v_param_b = model.forward(x_t, batch_b, tau, theta_tau)
    assert torch.allclose(v_state_a, v_state_b, atol=1e-6)
    assert torch.allclose(v_param_a, v_param_b, atol=1e-6)


def test_joint_cfm_coupled_sample_oracle_free_inference(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFMCoupled(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_flow_channels=[4, 8], N_outer=5)
    model.eval()
    batch = _joint_batch(w)
    torch.manual_seed(0)
    with torch.no_grad():
        _, p_a = model.sample(batch, N_outer=5, return_params=True)
    batch_b = FlowMatchingBatch(
        batch.states, batch.obs, batch.obs_mask, batch.forcing,
        params=batch.params, true_params=torch.randn_like(batch.true_params))
    torch.manual_seed(0)
    with torch.no_grad():
        _, p_b = model.sample(batch_b, N_outer=5, return_params=True)
    assert torch.allclose(p_a, p_b, atol=1e-6)


def test_joint_cfm_coupled_grads_both_flows(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFMCoupled(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_flow_channels=[4, 8], param_loss_weight=0.1)
    batch = _joint_batch(w)
    loss = model.compute_cfm_loss(batch)
    loss.backward()
    unet_g = sum(p.grad is not None and p.grad.abs().sum() > 0
                 for p in model.unet.parameters())
    pf_g = sum(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.param_flow.parameters())
    assert unet_g > 0
    assert pf_g > 0


def test_joint_cfm_coupled_multitau_no_shortcut(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFMCoupled(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_flow_channels=[4, 8], N_outer=10)
    model.eval()
    batch = _joint_batch(w)
    torch.manual_seed(0)
    with torch.no_grad():
        x1, p1 = model.sample(batch, N_outer=1, return_params=True)
    torch.manual_seed(0)
    with torch.no_grad():
        x10, p10 = model.sample(batch, N_outer=10, return_params=True)
    # multi-tau coupled ODE: integration depth changes the result (no tau=0 shortcut)
    assert not torch.allclose(x1, x10, atol=1e-6)
    assert not torch.allclose(p1, p10, atol=1e-6)


def test_joint_cfm_coupled_sample_finite(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFMCoupled(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_flow_channels=[4, 8], N_outer=3)
    model.eval()
    batch = _joint_batch(w)
    with torch.no_grad():
        x, p = model.sample(batch, N_outer=3, return_params=True)
    assert torch.isfinite(x).all()
    assert torch.isfinite(p).all()


def test_joint_direct_unet_unet_head_shapes(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_head_channels=[4, 8], param_head_backbone="unet")
    batch = _joint_batch(w)
    loss = model.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    x, p = model.sample(batch, return_params=True)
    assert x.shape == (1, w["true_state"].shape[0], SD)
    assert p.shape == (1, PD)
    assert isinstance(model.param_head, ParamHeadUNet)


def test_joint_direct_unet_unet_head_oracle_gone(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_head_channels=[4, 8], param_head_backbone="unet")
    model.eval()
    batch = _joint_batch(w)
    params_a = model.estimate_params(batch)
    batch_b = FlowMatchingBatch(
        batch.states, batch.obs, batch.obs_mask, batch.forcing,
        params=torch.randn_like(batch.params), true_params=torch.randn_like(batch.true_params))
    params_b = model.estimate_params(batch_b)
    assert torch.allclose(params_a, params_b, atol=1e-6)


def test_joint_direct_unet_unet_head_stop_grad(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_head_channels=[4, 8], param_head_backbone="unet")
    batch = _joint_batch(w)
    loss = model.compute_param_loss(batch)
    loss.backward()
    state_grads = [p.grad for p in model.unet.parameters() if p.grad is not None]
    assert len(state_grads) == 0, "state UNet must not receive param-head grads (stop_grad on x_hat)"
    head_grads = [p.grad for p in model.param_head.parameters() if p.grad is not None]
    assert len(head_grads) > 0


def test_param_flow_unet_attn_pool_shape(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointCFMCoupled(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_flow_channels=[4, 8], param_flow_pool="attn")
    assert hasattr(model.param_flow, "attn_pool")
    batch = _joint_batch(w)
    assert model.param_flow(batch.obs, batch.forcing, batch.states,
                            torch.randn(1, PD), torch.tensor([0.5])).shape == (1, PD)


def test_param_head_unet_attn_pool_shape(l96_joint_dataset):
    w = l96_joint_dataset[0]
    model = JointDirectUNet(state_dim=SD, param_dim=PD, hidden_channels=[8, 16],
                            param_head_channels=[4, 8], param_head_backbone="unet",
                            param_head_pool="attn")
    assert hasattr(model.param_head, "attn_pool")
    batch = _joint_batch(w)
    assert model.param_head(batch.obs, batch.forcing, batch.states).shape == (1, PD)
