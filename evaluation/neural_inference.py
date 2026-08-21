"""Standalone neural model inference and evaluation for L96."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import hydra
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM
from models.dynamics import get_dynamics
from data.dataloader import FlowMatchingDataset, ConcatFMDataset, make_dataloaders, collate_fm
from evaluation.metrics import energy_score
import torch
import torch.nn.functional as F
import numpy as np


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


def _per_group_rmse(preds: torch.Tensor, truth: torch.Tensor, obs_var_indices: list) -> dict:
    """Compute per-group RMSE (slow/obs_fast)."""
    rmse_full = torch.sqrt(torch.mean((preds - truth) ** 2, dim=0))
    
    return {
        "slow": rmse_full[:8].tolist(),
        "obs_fast": rmse_full[8:].tolist(),
        "all_obs": rmse_full.tolist(),
    }


def _per_group_ev(preds: torch.Tensor, truth: torch.Tensor, obs_var_indices: list) -> dict:
    """Compute per-group explained variance."""
    mean_truth = torch.mean(truth, dim=0)
    var_truth = torch.var(truth, dim=0)
    
    mse = torch.mean((preds - truth) ** 2, dim=0)
    
    # Avoid division by zero
    var_truth_safe = torch.where(var_truth > 1e-10, var_truth, torch.ones_like(var_truth))
    
    ev_full = 1 - torch.mean(mse, dim=0) / var_truth_safe
    
    return {
        "slow": ev_full[:8].tolist(),
        "obs_fast": ev_full[8:].tolist(),
        "all_obs": ev_full.tolist(),
    }
from data.lorenz96 import Lorenz96Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    checkpoint_path: str
    config_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 200
    num_workers: int = 0
    deterministic: bool = False  # DirectUNet only
    seed: Optional[int] = None


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

        # state_dim = output_dim = first channel of the final enc_out conv
        if "model.unet.enc_out.2.weight" in state_dict:
            inferred_params["state_dim"] = state_dict["model.unet.enc_out.2.weight"].shape[1]
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
    **kwargs,
) -> ConcatFMDataset:
    """Prepare test dataset for evaluation."""
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
    
    # Create dataloader for evaluation (use S0 test dataset)
    test_s0 = dataset["test_s0"]
    dataloader = DataLoader(
        test_s0,
        batch_size=kwargs.get("batch_size", 200),
        shuffle=False,
        collate_fn=collate_eval,
        num_workers=kwargs.get("num_workers", 0),
    )
    return dataset, dataloader


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model and compute metrics."""
    model.eval()
    all_preds = []
    all_true = []
    all_obs = []
    all_forcing = []
    
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
            all_obs.append(batch["obs"].detach().cpu())
            all_forcing.append(batch["forcing_corrupted"].detach().cpu())
    
    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_true = torch.cat(all_true, dim=0)
    all_obs = torch.cat(all_obs, dim=0)
    all_forcing = torch.cat(all_forcing, dim=0)
    
    # Compute metrics (MSE)
    mse = torch.mean((all_preds - all_true) ** 2)
    rmse = torch.sqrt(mse)
    
    # Per-group metrics
    obs_var_indices = None
    if hasattr(model, 'state_dim') and model.state_dim > 0:
        obs_j = getattr(model, 'obs_j', 2)
        NO = getattr(model, 'NO', 8)
        J = getattr(model, 'J', 4)
        obs_var_indices = list(range(obs_j * NO))
    
    if obs_var_indices is not None:
        rmse_full = _per_group_rmse(all_preds, all_true, obs_var_indices)
        rmse_obs = rmse_full["obs_fast"]
        rmse_slow = rmse_full["slow"]
    else:
        rmse_full = torch.tensor([rmse.item()])
        rmse_obs = torch.tensor([rmse.item()])
        rmse_slow = torch.tensor([rmse.item()])
    
    # Explained variance
    if obs_var_indices is not None:
        ev = _per_group_ev(all_preds, all_true, obs_var_indices)
        ev_mean = np.mean(ev["all_obs"])
    else:
        mean_truth = torch.mean(all_true, dim=0)
        var_truth = torch.var(all_true, dim=0)
        mse = torch.mean((all_preds - all_true) ** 2, dim=0)
        var_truth_safe = torch.where(var_truth > 1e-10, var_truth, torch.ones_like(var_truth))
        ev = 1 - torch.mean(mse, dim=0) / var_truth_safe
        ev_mean = ev.mean().item()
    
    # Energy Score (N=1 -> MAE)
    es = energy_score(all_preds, all_true, ensemble_size=1)
    
    results = {
        "rmse": rmse.item(),
        "rmse_full": rmse_full.tolist(),
        "rmse_obs_fast": rmse_obs.tolist(),
        "rmse_slow": rmse_slow.tolist(),
        "ev": ev.item() if isinstance(ev, torch.Tensor) else ev,
        "es": es.item() if isinstance(es, torch.Tensor) else es,
        "obs_var_indices": obs_var_indices,
        "num_samples": all_true.shape[0],
    }
    
    return results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained neural model on L96 test dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", help="Path to model config YAML (optional)")
    parser.add_argument("--dataset", help="Path to cached test dataset (optional)")
    parser.add_argument("--num-windows", type=int, default=200, help="Number of test windows")
    parser.add_argument("--obs-interval", type=int, default=100, help="Observation interval")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic mode (DirectUNet only)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", default="neural_eval_results.json", help="Output JSON file")
    parser.add_argument("--exp-dir", default="experiments", help="Experiment directory")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device=device)
    logger.info(f"Model: {type(model).__name__}, state_dim={model.state_dim}")
    
    # Prepare dataset
    dataset_path = args.dataset
    if not dataset_path:
        # Infer from checkpoint filename or config
        ckpt_dir = Path(args.checkpoint).parent
        dataset_path = list(ckpt_dir.glob("l96_datasets_obsj2_int100_nwin200.pt"))[0] if ckpt_dir.glob("l96_datasets_obsj2_int100_nwin200.pt") else None
    
    dataset, dataloader = prepare_dataset(cfg, dataset_path, args.num_windows, args.obs_interval)
    logger.info(f"Dataset: {len(dataset)} windows, {args.batch_size} samples/batch")
    
    # Evaluate
    logger.info("Evaluating model...")
    results = evaluate_model(model, dataloader, device)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "checkpoint": args.checkpoint,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": {
            "path": str(dataset_path) if dataset_path else None,
            "num_windows": args.num_windows,
            "obs_interval": args.obs_interval,
        },
        "metrics": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"RMSE: {results['rmse']:.4f}")
    logger.info(f"RMSE (slow): {results['rmse_slow']:.4f}")
    logger.info(f"RMSE (obs_fast): {results['rmse_obs_fast']:.4f}")
    logger.info(f"EV: {results['ev']:.4f}")
    logger.info(f"ES: {results['es']:.4f}")
    
    return results


if __name__ == "__main__":
    main()
