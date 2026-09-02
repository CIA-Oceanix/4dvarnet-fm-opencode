import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.run_qg_baselines import run


def probe():
    device = torch.device("cpu")
    cfg = QGConfig(
        nx=16, window_days=15.0, spinup_years=0.2,
        num_windows=2, cols_per_day=4, obs_noise_std_frac=0.05,
        obs_geometry="random_columns", seed=7,
    )
    ds = make_qg_s0_s1_datasets(cfg, cache_dir="/tmp/qg_probe_cache")
    scenarios = ("test_s0", "test_s1_qg1l")
    print("DATASET READY", flush=True)
    for obs_var in ("psi", "q"):
        for rscale in (1.0, 1e2, 1e4, 1e6):
            p = run("etkf", cfg, device=device, N_ensemble=40,
                    inflation=1.0, loc_radius=6,
                    scenarios=scenarios, init="lagged",
                    geometry="random_columns", obs_var=obs_var,
                    init_lag_days=1.0, band_half=0.25, ds=ds,
                    obs_var_r_scale=rscale)
            s0 = p["scenarios"].get("test_s0", {})
            q1 = p["scenarios"].get("test_s1_qg1l", {})
            print(
                f"ov={obs_var} rs={rscale:g} | "
                f"S0 EV={s0.get('expvar_full'):.3f} impr={s0.get('forecast_improvement'):.3f} | "
                f"Q1 EV={q1.get('expvar_full'):.3f} EVfree={q1.get('expvar_free'):.3f} "
                f"impr={q1.get('forecast_improvement'):.3f}",
                flush=True)
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    probe()
