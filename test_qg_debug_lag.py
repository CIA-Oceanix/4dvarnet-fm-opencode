import sys
sys.path.insert(0, '..')

import torch
from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.run_qg_baselines import _lagged_init_ensemble, run

print("Testing lag fix...")

cfg = QGConfig(nx=64, window_days=30, obs_geometry="random_columns",
               cols_per_day=4, seed=7, init_lead_days=10.0)
print(f"cfg loaded: init_lead_days={cfg.init_lead_days}, num_steps={cfg.num_steps}")

ds = make_qg_s0_s1_datasets(cfg)
w = ds["test_s0"][0]
print(f"Window loaded: true_state.shape={w['true_state'].shape}")

init_ensemble, mean_lag_val = _lagged_init_ensemble(cfg, w, N=80, init_lag_days=1.0, device=torch.device("cpu"))
print(f"lag=1.0: mean_lag={mean_lag_val:.2f}, std={init_ensemble.std().item():.4f}")

init_ensemble, mean_lag_val = _lagged_init_ensemble(cfg, w, N=80, init_lag_days=2.0, device=torch.device("cpu"))
print(f"lag=2.0: mean_lag={mean_lag_val:.2f}, std={init_ensemble.std().item():.4f}")

init_ensemble, mean_lag_val = _lagged_init_ensemble(cfg, w, N=80, init_lag_days=5.0, device=torch.device("cpu"))
print(f"lag=5.0: mean_lag={mean_lag_val:.2f}, std={init_ensemble.std().item():.4f}")

print("\n✓ Lag fix verified!")

# Run ETKF
p = run("etkf", cfg, device=torch.device("cpu"), N_ensemble=60,
        inflation=1.0, init="lagged", geometry="random_columns",
        scenarios=("test_s0",), out_path=None, ds=ds, init_lag_days=1.0)
print(f"ETKF test_s0 EV={p['scenarios']['test_s0']['expvar_full']:.4f}")
