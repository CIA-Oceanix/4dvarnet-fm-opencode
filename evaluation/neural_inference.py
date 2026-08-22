"""Standalone neural model inference and evaluation for L96."""
import logging
from typing import Any, Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.lorenz96 import Lorenz96Config
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM


class BatchDict:
    """Simple wrapper for batch dict access."""
    def __init__(self, d):
        self.__dict__.update(d)


def collate_eval(batch):
    """Collate function for RandomParamLorenz96Dataset (Dict format)."""
    states = torch.stack([b["true_state"] for b in batch])
    obs = torch.stack([b["obs"] for b in batch])
    masks = torch.stack([b["obs_mask"] for b in batch])
    forcing = torch.stack([b["forcing_corrupted"] for b in batch])
    return {"true_state": states, "obs": obs, "obs_mask": masks, "forcing": forcing, "params": None}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, config_path: Optional[str] = None) -> tuple:
    """Load model checkpoint and config."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    
    # Handle Lightning .ckpt files
    if "hyper_parameters" in ckpt:
        # Lightning checkpoint: extract model_type from hyper_parameters
        model_type = ckpt["hyper_parameters"].get("model_type", "direct_unet")
        
        # Infer architecture parameters from state_dict
        inferred_params = {}

        # state_dim = output_dim = first channel of the final enc_out conv.
        # enc_out.2 is Conv1d(in_c, output_dim, 3) -> weight shape (output_dim, in_c, 3),
        # so shape[0] is the state/output dimension.
        if "model.unet.enc_out.2.weight" in state_dict:
            inferred_params["state_dim"] = state_dict["model.unet.enc_out.2.weight"].shape[0]
        state_dim = inferred_params.get("state_dim", 24)

        # proj_in = state_dim + obs_dim + cond_extra_dim, with obs_dim = state_dim
        if "model.unet.cond_encoder.proj.weight" in state_dict:
            cond_proj_weight = state_dict["model.unet.cond_encoder.proj.weight"]
            inferred_params["cond_extra_dim"] = cond_proj_weight.shape[1] - 2 * state_dim

        # Infer hidden_channels from downs/ups layers
        # downs.0.conv1: [hidden[0], hidden[0], 3] (first layer, same in/out)
        # downs.1.conv1: [hidden[1], hidden[0], 3] (second layer)
        # downs.2.conv1: [hidden[2], hidden[1], 3] (third layer)
        if "model.unet.downs.1.block.conv1.weight" in state_dict:
            conv1 = state_dict["model.unet.downs.1.block.conv1.weight"]
            # Shape is [hidden[1], hidden[0], 3]
            inferred_params["hidden_channels"] = [conv1.shape[1], conv1.shape[0], 256]

        # Use inferred params or defaults
        cfg_dict = {
            "model": {
                "type": model_type,
                "state_dim": state_dim,
                "hidden_channels": inferred_params.get("hidden_channels", [64, 128, 256]),
                "time_emb_dim": 64,
                "param_dim": 0,  # Lightning checkpoints were trained with param_dim=0
                "cond_extra_dim": inferred_params.get("cond_extra_dim", 0),
                "device": "cpu",
            },
            "deterministic": False,
        }
        cfg = OmegaConf.create(cfg_dict)
    else:
        # Custom checkpoint format
        cfg_dict = ckpt.get("config", {})
        if config_path:
            cfg = OmegaConf.load(config_path)
            OmegaConf.update(cfg, "model", cfg_dict.get("model", {}), merge=True)
        else:
            cfg_dict["model"]["device"] = cfg_dict.get("device", "cpu")
            cfg = OmegaConf.create(cfg_dict)
    
    return state_dict, cfg


def resolve_model_class(cfg: Any) -> tuple:
    """Resolve model class from config."""
    model_type = cfg.model.get("type", "DirectUNet")
    
    # Normalize model type (handle both "direct_unet" and "DirectUNet")
    model_type = model_type.replace("_", "").replace("-", "").upper()
    
    if model_type == "DIRECTUNET":
        return DirectUNet, cfg
    elif model_type == "VANILLACFM":
        return VanillaCFM, cfg
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model(model_class, cfg: Any) -> torch.nn.Module:
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
    else:
        raise ValueError(f"Unknown model type: {model_class}")
    
    return model


def load_model(checkpoint_path: str, config_path: Optional[str] = None, **kwargs) -> tuple:
    """Load model from checkpoint."""
    state_dict, cfg = load_checkpoint(checkpoint_path, config_path)
    model_class, cfg_model = resolve_model_class(cfg)
    model = create_model(model_class, cfg_model)
    
    # Move model to correct device
    device = kwargs.get("device", "cpu")
    model.to(device)
    
    # Strip "model." prefix if present (Lightning wrapper)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v  # Remove "model." prefix
        else:
            new_state_dict[k] = v
    
    # Filter out mismatched keys (e.g., output_dim mismatch)
    model_sd = model.state_dict()
    filtered_state_dict = {}
    for k, v in new_state_dict.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            filtered_state_dict[k] = v
        else:
            logger.warning(f"Skipping key {k}: checkpoint shape {v.shape} vs model shape {model_sd[k].shape if k in model_sd else 'missing'}")
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model, cfg


def prepare_dataset(
    cfg: Any,
    test_dataset_path: Optional[str] = None,
    num_test_windows: int = 200,
    obs_interval: int = 100,
    obs_var_indices: tuple | None = None,
    **kwargs,
) -> tuple:
    """Prepare S0/S1 test dataloaders for evaluation.

    Returns ``(dataset, dataloaders, obs_var_indices)`` where ``dataset`` is the
    cached S0/S1 dict and ``dataloaders`` maps ``"s0"``/``"s1"`` to ``DataLoader``
    over the matching test split. The cached dataset (as used by the DA baselines)
    is loaded when ``test_dataset_path`` is given; otherwise the S0/S1 train/val/
    test split is generated. ``obs_var_indices`` (the observed subspace of the
    full 40D state) is resolved if not supplied and returned to the caller so the
    model's 24D predictions can be compared against the correct truth columns.
    """
    # Try to load cached dataset
    if test_dataset_path:
        logger.info(f"Loading cached test dataset from {test_dataset_path}")
        dataset = torch.load(test_dataset_path, weights_only=False)
    else:
        # Generate test dataset
        logger.info(f"Generating test dataset: {num_test_windows} windows, obs_interval={obs_interval}")
        from data.lorenz96 import make_l96_s0_s1_trainval
        cfg_test = Lorenz96Config(
            system="lorenz96",
            NO=cfg.data.system_config.NO,
            J=cfg.data.system_config.J,
            obs_j=cfg.data.obs_j,
            obs_interval=obs_interval,
            num_test_windows=num_test_windows,
            seed=42,
        )
        train, val, test = make_l96_s0_s1_trainval(cfg_test)
        dataset = test

    # Create dataloaders for both the S0 and S1 test splits
    dataloaders = {}
    for key, case in (("test_s0", "s0"), ("test_s1", "s1")):
        split = dataset[key]
        dataloaders[case] = DataLoader(
            split,
            batch_size=kwargs.get("batch_size", 200),
            shuffle=False,
            collate_fn=collate_eval,
            num_workers=kwargs.get("num_workers", 0),
        )

    # Resolve the observed-subspace indices of the full state. The cached DA
    # dataset stores the full 40D true_state with obs already subsampled to the
    # observed dims. When the model predicts a 24D observed state we must compare
    # against true_state[..., obs_var_indices] (a non-contiguous subset), NOT the
    # first `state_dim` columns.
    if obs_var_indices is None:
        obs_j = int(kwargs.get("obs_j", cfg.get("data", {}).get("obs_j", 2))) if hasattr(cfg, "get") else 2
        try:
            NO = int(cfg.data.system_config.NO)
        except Exception:
            NO = 8
        try:
            J = int(cfg.data.system_config.J)
        except Exception:
            J = 4
        from evaluation.run_l96 import make_obs_j_indices
        obs_var_indices = make_obs_j_indices(NO, J, obs_j)
        if obs_var_indices is None:
            obs_var_indices = tuple(range(NO + NO * J))
    obs_var_indices = tuple(obs_var_indices)

    return dataset, dataloaders, obs_var_indices


def _run_case_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    obs_var_indices: tuple | None = None,
) -> dict:
    """Run a model on a single case dataloader and return state estimates.

    Returns a dict with numpy arrays ``{"trajectories": (W, T, D), "truth": (W, T, D)}``
    where ``D`` is the observed subspace (the neural model's output dim) and
    ``truth`` is subsampled to that same subspace for direct comparison. No
    metrics are computed here — that is the job of the generic evaluator.
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            # Convert tensors to device, skip None values
            batch = {k: v.to(device) if v is not None else v for k, v in batch.items()}
            batch_obj = BatchDict(batch)

            if isinstance(model, DirectUNet):
                # DirectUNet.forward takes a batch dict
                pred = model(batch_obj)
            elif isinstance(model, VanillaCFM):
                # VanillaCFM.sample takes a batch dict
                pred = model.sample(batch_obj, N_outer=1)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            all_preds.append(pred.detach().cpu())
            all_true.append(batch["true_state"].detach().cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_true = torch.cat(all_true, dim=0)

    # The neural model predicts the observed subspace while the cached truth is
    # the full state. Subsample truth to the observed dims for a fair comparison.
    # IMPORTANT: obs_var_indices is a NON-CONTIGUOUS subset (e.g. X1-X8 then
    # Y1,Y2 of each node), so we must use those exact columns, NOT the first
    # `state_dim` columns (which would mix in unobserved fast vars).
    d_pred = all_preds.shape[-1]
    if all_true.shape[-1] > d_pred:
        if obs_var_indices is not None and len(obs_var_indices) == d_pred:
            all_true = all_true[..., list(obs_var_indices)]
        else:
            all_true = all_true[..., :d_pred]

    return {
        "trajectories": all_preds.detach().cpu().numpy(),
        "truth": all_true.detach().cpu().numpy(),
    }


def run_inference(
    model: torch.nn.Module,
    dataloaders: dict,
    device: torch.device,
    obs_var_indices: tuple | None = None,
) -> dict:
    """Run inference on both S0 and S1, returning per-case estimates.

    Returns ``{"s0": {...}, "s1": {...}}`` where each entry holds the numpy
    ``trajectories``/``truth`` arrays (no metrics). To produce scores, pass
    these to the generic evaluator (``evaluate_estimates``).
    """
    return {
        case: _run_case_inference(model, dl, device, obs_var_indices)
        for case, dl in dataloaders.items()
    }
