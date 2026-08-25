"""Neural model inference utilities."""
import os
import torch
import torch.nn as nn
from typing import Any, Optional
import numpy as np
from data.dataloader import FlowMatchingDataset, collate_fn as collate_fm, FlowMatchingBatch
from data.lorenz96 import Lorenz63Config
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM, JointCFM, TweedieCFM, PredictStateCFM
from evaluation.metrics import rmse as compute_rmse


def collate_eval(batch: list) -> FlowMatchingBatch:
    """Collate batch for prediction."""
    obs = torch.stack([item["obs"] for item in batch])
    states = torch.stack([item["true_state"] for item in batch])
    forcing = torch.stack([item["forcing_corrupted"] for item in batch])
    mask = torch.stack([item["obs_mask"] for item in batch])
    params = None
    true_params = None
    if "param" in item[0]:
        params = torch.stack([item["param"] for item in batch]).unsqueeze(-1)
    if "true_param" in item[0]:
        true_params = torch.stack([item["true_param"] for item in batch]).unsqueeze(-1)
    return FlowMatchingBatch(states, obs, mask, forcing, params=params, true_params=true_params)


def collate_joint_eval(batch: list) -> FlowMatchingBatch:
    """Collate batch for joint state-parameter prediction."""
    multi_param_batch = collate_eval(batch)
    param = torch.stack([[item.get("param_gas", 8.0)] for item in batch], dim=0).unsqueeze(-1)
    true_param = torch.stack([[item.get("true_param_gas", 8.0)] for item in batch], dim=0).unsqueeze(-1)
    return FlowMatchingBatch(
        multi_param_batch.states, multi_param_batch.obs, multi_param_batch.obs_mask,
        multi_param_batch.forcing, params=param, true_params=true_param
    )


# --- Loading and configuration ---

def load_checkpoint(checkpoint_path: str, config_path: Optional[str] = None) -> tuple:
    """Load checkpoint and config."""
    from hydra import compose, initialize

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if config_path:
        if isinstance(config_path, str) and ".yaml" in config_path and os.path.exists(config_path):
            config = OmegaConf.load(config_path)
        else:
            with initialize(config_path=os.path.dirname(config_path) if config_path else "config"):
                cfg = compose(config_name=os.path.basename(config_path).replace(".yaml", ""))
                config = cfg
        # Remove model.*.weight prefix if present
        state = checkpoint["state_dict"]
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[6:]] = v
            else:
                new_state[k] = v
        checkpoint["state_dict"] = new_state
    elif hasattr(checkpoint, "state_dict"):
        state = checkpoint.state_dict()
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[6:]] = v
            else:
                new_state[k] = v
        checkpoint["state_dict"] = new_state

    return checkpoint, config


def resolve_model_class(cfg: Any) -> tuple:
    """Resolve model class from config."""
    model_type = cfg.model.get("type", "DirectUnet")

    # Normalize model type
    model_type = model_type.replace("_", "").replace("-", "").upper()

    if model_type == "DIRECTUNET":
        return DirectUNet, cfg
    elif model_type == "VANILLACFM":
        return VanillaCFM, cfg
    elif model_type == "PREDICTSTATECFM":
        return PredictStateCFM, cfg
    elif model_type == "TWEEDIECFM":
        return TweedieCFM, cfg
    elif model_type == "JOINTDIRECTUNET":
        return JointDirectUnet, cfg
    elif model_type == "JOINTCFM":
        return JointCFM, cfg
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model(model_class, cfg: Any) -> nn.Module:
    """Create model instance from config."""
    hidden = cfg.model.get("hidden_channels", cfg.model.get("hidden", [64, 128, 256]))

    if model_class == DirectUNet:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            cond_extra_dim=cfg.model.get("cond_extra_dim", 0),
        )
    elif model_class == VanillaCFM:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            time_emb_dim=cfg.model.get("time_emb_dim", 64),
            N_outer=cfg.model.get("N_outer", 10),
            sigma_prior=cfg.model.get("sigma_prior", 0.5),
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            train_tau_0_only=cfg.model.get("train_tau_0_only", False),
            cond_extra_dim=cfg.model.get("cond_extra_dim", 0),
        )
    elif model_class == PredictStateCFM:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            time_emb_dim=cfg.model.get("time_emb_dim", 64),
            N_outer=cfg.model.get("N_outer", 10),
            sigma_prior=cfg.model.get("sigma_prior", 0.5),
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            cond_extra_dim=cfg.model.get("cond_extra_dim", 0),
        )
    elif model_class == TweedieCFM:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            time_emb_dim=cfg.model.get("time_emb_dim", 64),
            K_inner=cfg.model.get("K_inner", 5),
            N_outer=cfg.model.get("N_outer", 10),
            sigma_prior=cfg.model.get("sigma_prior", 0.5),
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            cond_extra_dim=cfg.model.get("cond_extra_dim", 0),
        )
    elif model_class == JointCFM:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            time_emb_dim=cfg.model.get("time_emb_dim", 64),
            N_outer=cfg.model.get("N_outer", 10),
            sigma_prior=cfg.model.get("sigma_prior", 0.5),
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            param_loss_weight=cfg.model.get("param_loss_weight", 0.1),
            train_tau_0_only=cfg.model.get("train_tau_0_only", False),
        )
    elif model_class == JointDirectUnet:
        model = model_class(
            state_dim=cfg.model.state_dim,
            hidden_channels=hidden,
            dropout=cfg.model.get("dropout", 0.1),
            param_dim=cfg.model.get("param_dim", 1),
            param_loss_weight=cfg.model.get("param_loss_weight", 0.1),
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    return model


def load_model(checkpoint_path: str, config_path: Optional[str] = None, **kwargs) -> tuple:
    """Load model from checkpoint.

    ``overrides`` is merged into the constructed `cfg.model` before instantiation.
    Needed for `train_tau_0_only` on tau=0-trained CFM checkpoints.
    """
    overrides = kwargs.pop("overrides", None)
    state_dict, cfg = load_checkpoint(checkpoint_path, config_path)
    if overrides:
        for key, value in overrides.items():
            cfg.model[key] = value
    model_class, cfg_model = resolve_model_class(cfg)
    model = create_model(model_class, cfg_model)

    device = kwargs.get("device", "cpu")
    model.to(device)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    filtered_state_dict = {}
    model_sd = model.state_dict()
    for k, v in new_state_dict.items():
        if k in model_sd:
            filtered_state_dict[k] = v

    model.load_state_dict(filtered_state_dict)
    model.eval()
    return model, cfg_model


# --- Dataset and inference ---

def prepare_dataset(dataset, system="lorenz63", obs_var_indices=None,
                   obs_interval=100, param_names=("sigma", "rho", "beta", "c1"),
                   num_test_windows=200) -> dict:
    """Prepare test dataset for inference."""
    if system == "lorenz96":
        from data.lorenz96 import Lorenz96Config, make_l96_s0_s1_trainval

        cfg = Lorenz96Config(
            case=1,
            obs_interval=obs_interval,
            num_windows=1000 + num_test_windows + 100,
            spinup_steps=10000,
            seed=42,
            NO=8, J=4,
            h=1.0,
            hx=1.0,
            eps=0.1,
            force_state_bias=0.0,
            force_coupling="linear",
        )
        datasets = make_l96_s0_s1_trainval(
            cfg,
            num_train_windows=1000,
            num_val_windows=100,
            num_test_windows=num_test_windows,
            param_noise=0.2,
            bias_range=(0.0, 0.2),
        )
    else:
        from data.lorenz63 import Lorenz63Config, make_mixed_datasets

        cfg = Lorenz63Config(
            obs_interval=obs_interval,
            num_windows=1300,
            spinup_steps=10000,
            seed=42,
        )
        datasets = make_mixed_datasets(
            cfg,
            num_train_windows=1000,
            num_val_windows=100,
            num_test_windows=num_test_windows,
            include_randparam_test=True,
            param_noise=0.2,
        )

    return {"train": datasets["test_" + case] for case in ["s0", "s1"]}


def _run_case_inference(
    model, dataset, config: Any, model_type: str,
    temperature=1.0,                 
    n_outer=10, N_outer_total=None,
    obs_var_indices=None
):
    """Run inference on a single case."""
    device = next(model.parameters()).device
    batch_list = []

    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"].unsqueeze(0)
        states = w["true_state"].unsqueeze(0)
        forcing = w["forcing_corrupted"].unsqueeze(0)
        mask = w["obs_mask"].unsqueeze(0)
        batch = FlowMatchingBatch(states, obs, mask, forcing)
        batch_list.append(batch)

    import torch.utils.data as td
    dataset_loader = td.DataLoader(
        td.Dataset(batch_list),
        batch_size=64,
        shuffle=False,
        collate_fn=lambda x: x[0]
    )

    with torch.no_grad():
        outputs = []
        for batch in dataset_loader:
            if model_type == "direct_unet":
                pred = model(batch).cpu().numpy()
            elif model_type == "tweedie":
                pred = model(batch.obs).cpu().numpy()
            elif model_type == "vanilla_cfm":
                pred = model.sample(batch, N_outer=n_outer).cpu().numpy()
            elif model_type == "tweedie_cfm":
                pred = model.sample(batch, N_outer=n_outer).cpu().numpy()
            elif model_type == "predict_state_cfm":
                pred = model.sample(batch, N_outer=n_outer).cpu().numpy()
            elif model_type == "joint_cfm":
                results = model.sample(batch, return_params=True)
                pred, params = results
                pred = pred.cpu().numpy()
            elif model_type == "joint_direct_unet":
                results = model.sample(batch, return_params=True)
                pred, params = results
                pred = pred.cpu().numpy()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            outputs.append(pred)

    outputs = np.concatenate(outputs, axis=0)

    if obs_var_indices is not None:
        truth = dataset[0]["true_state"].numpy().reshape(1, -1, 3)
        truth = truth[..., obs_var_indices]
    else:
        truth = dataset[0]["true_state"].numpy().reshape(1, -1, 3)

    outputs = outputs[0].astype(np.float32)

    rmse = compute_rmse(outputs.flatten(), truth.flatten())
    return outputs, truth, rmse, batch_list


def run_inference(
    checkpoint_path: str,
    output_dir: str = "experiments",
    **kwargs
):
    """Main entry point for neural model inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory for outputs
        temperature: Temperature for stochastic operations
        n_outer: Number of ODE steps for sampling
        seed: Random seed for reproducibility
        cases: List of cases to evaluate (e.g., "s0", "s1")
        model_type_override: Optional override for model_type
    """
    from data.dataloader import FlowMatchingBatch
    from classification.pipeline import setup_seed
    
    model, cfg = load_model(checkpoint_path, **kwargs)
    model_type = cfg.model.get("model_type", "l96")
    if model_type_override is not None:
        model_type = model_type_override
    
    output_dir = os.path.join(output_dir, cfg.get("experiment_id", "unknown"))
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for case in cases:
        dataset = prepare_dataset(None, system="lorenz96", Number=case)
        outputs, truth, rmse, batch_list = _run_case_inference(
            model, dataset, cfg, model_type,
            obs_interval=cfg.data.obs_interval if "obs_interval" in cfg.data else 100,
            obs_var_indices=None,
        )
        results[case] = {
            "outputs": outputs,
            "truth": truth,
            "rmse": float(rmse),
        }
        
        np.savez_compressed(
            os.path.join(output_dir, f"estimates_{case}.npz"),
            estimates=outputs,
            truth=truth.astype(np.float32)
        )
    
    json_path = os.path.join(output_dir, "neural_eval.json")
    import json
    with open(json_path, "w") as f:
        json.dump({
            "experiment_id": cfg.get("experiment_id", "unknown"),
            "config": cfg.model if hasattr(cfg, "model") else {},
            "results": results,
        }, f, indent=2)
    
    return results
