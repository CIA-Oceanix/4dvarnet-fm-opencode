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
from data.dataloader import FlowMatchingDataset, ConcatFMDataset, make_dataloaders
from evaluation.metrics import energy_score
import torch
import torch.nn.functional as F
import numpy as np


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
    model = ckpt["state_dict"]
    cfg_dict = ckpt.get("config", {})
    
    if config_path:
        cfg = OmegaConf.load(config_path)
        # Merge checkpoint config into file config
        OmegaConf.update(cfg, "model", cfg_dict.get("model", {}), merge=True)
    else:
        cfg_dict["model"]["device"] = cfg_dict.get("device", "cpu")
        cfg = OmegaConf.create(cfg_dict)
    
    return model, cfg


def resolve_model_class(cfg: Any) -> tuple:
    """Resolve model class and create instance from config."""
    model_type = cfg.model.get("type", "DirectUNet")
    
    if model_type == "DirectUNet":
        return DirectUNet, cfg
    elif model_type == "VanillaCFM":
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
        )
    else:
        raise ValueError(f"Unknown model type: {model_class}")
    
    return model


def load_model(checkpoint_path: str, config_path: Optional[str] = None, **kwargs) -> tuple:
    """Load model from checkpoint."""
    model, cfg = load_checkpoint(checkpoint_path, config_path)
    model_class, cfg_model = resolve_model_class(cfg)
    model = create_model(model_class, cfg_model)
    model.load_state_dict(model)
    model.eval()
    return model, cfg


def prepare_dataset(
    cfg: Any,
    test_dataset_path: Optional[str] = None,
    num_test_windows: int = 200,
    obs_interval: int = 100,
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
    
    # Create dataloader
    dataloader = make_dataloaders(
        dataset=dataset,
        batch_size=kwargs.get("batch_size", 200),
        num_workers=kwargs.get("num_workers", 0),
        deterministic=kwargs.get("deterministic", False),
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
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if isinstance(model, DirectUNet):
                # DirectUNet.forward takes a batch dict
                pred = model(batch)
            elif isinstance(model, VanillaCFM):
                # VanillaCFM.sample takes a batch dict
                pred = model.sample(batch, N_outer=1)
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
