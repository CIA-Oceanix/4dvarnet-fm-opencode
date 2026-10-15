import argparse
import logging

import hydra
import torch

from data.lorenz63 import Lorenz63Config, make_s0_s1_trainval
from data.normalization import compute_channel_stats, save_norm_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Precompute L63 per-channel z-score stats on the training windows")
    parser.add_argument("--config", default="experiment/L63B_directunet_monaiM")
    parser.add_argument("--output", default="experiments/l63_norm_stats.pt")
    args = parser.parse_args()

    with hydra.initialize(config_path="config", version_base="1.3"):
        cfg = hydra.compose(args.config)
    dc = cfg.data
    base_cfg = Lorenz63Config(
        dt=dc.dt, T_max=dc.T_max, obs_interval=dc.obs_interval, R_var=dc.R_var, B_var=dc.B_var,
        num_windows=dc.num_windows, window_spacing=dc.window_spacing, spinup_steps=dc.spinup_steps,
        seed=dc.get("seed", 42), sigma_true=dc.sigma_true, rho_true=dc.rho_true, beta_true=dc.beta_true,
        gamma=dc.gamma, W_L_bar=dc.W_L_bar, c1=dc.c1, c2=dc.c2, sigma_0=dc.sigma_0, sigma_L=dc.sigma_L,
        tau_eta=dc.tau_eta, sigma_eta=dc.sigma_eta, param_bias=dc.get("param_bias", 0.0),
        forcing_state_bias=dc.get("forcing_state_bias", 0.0),
        forcing_coupling=dc.get("forcing_coupling", "linear"),
    )
    datasets = make_s0_s1_trainval(
        base_cfg,
        num_train_windows=dc.get("num_train_windows", 1000),
        num_val_windows=dc.get("num_val_windows", 100),
        num_test_windows=dc.get("num_test_windows", 200),
        param_noise=dc.get("test_param_noise", 0.2),
        bias_range=(0.0, dc.get("bias_max", 0.2)),
    )
    train = datasets["train"]
    states = torch.stack([train[i]["true_state"] for i in range(len(train))])
    stats = compute_channel_stats(states)
    logger.info(f"{tuple(states.shape)} windows; mean {stats['mean'].tolist()}; std {stats['std'].tolist()}")
    save_norm_stats(args.output, stats)
    logger.info(f"Saved stats to {args.output}")


if __name__ == "__main__":
    main()
