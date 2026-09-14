"""Per-channel z-score normalization for L96 states/observations.

Stats are computed once (see ``precompute_l96_norm_stats.py``) over the
24D observed subspace (8 slow + 16 fast channels) and reused across every
model/phase of the normalization ablation, so comparisons aren't
contaminated by different-seed stat estimates.
"""
import numpy as np
import torch


def compute_channel_stats(states) -> dict:
    """Per-channel mean/std of ``states`` (..., D), pooled over all leading dims."""
    flat = states.reshape(-1, states.shape[-1])
    if isinstance(states, np.ndarray):
        mean, std = flat.mean(axis=0), flat.std(axis=0)
    else:
        mean, std = flat.mean(dim=0), flat.std(dim=0)
    return {"mean": mean, "std": std}


def _match_stats(x, stats: dict):
    mean, std = stats["mean"], stats["std"]
    if isinstance(x, torch.Tensor):
        mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(std, dtype=x.dtype, device=x.device)
    else:
        if isinstance(mean, torch.Tensor):
            mean = mean.detach().cpu().numpy()
            std = std.detach().cpu().numpy()
        mean, std = np.asarray(mean, dtype=x.dtype), np.asarray(std, dtype=x.dtype)
    return mean, std


def normalize(x, stats: dict):
    """z-score normalize ``x`` (..., D) with per-channel ``stats['mean']``/``['std']``."""
    mean, std = _match_stats(x, stats)
    return (x - mean) / std


def denormalize(x, stats: dict):
    """Invert :func:`normalize`."""
    mean, std = _match_stats(x, stats)
    return x * std + mean


def load_norm_stats(path: str) -> dict:
    d = torch.load(path, weights_only=False)
    return {"mean": d["mean"].float(), "std": d["std"].float()}


def save_norm_stats(path: str, stats: dict) -> None:
    torch.save({"mean": stats["mean"].float(), "std": stats["std"].float()}, path)
