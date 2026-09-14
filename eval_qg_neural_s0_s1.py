#!/usr/bin/env python3
"""Cross-scenario (S0/S1) evaluation of the QG neural schemes: the canonical
T-channels DirectUNet family -- Q1 (obs-only), Q2 (oracle forcing+param
cond.), Q3 (noisy-trained forcing+param cond.), Q4 (noisy cond. + IC).

Generalizes the earlier scratch `eval_qg_q1_s0_s1.py` (which only checked
Q1's scenario-agnostic sanity property) to the full comparison this was built
for -- see PLAN.md's 2026-09-10 QG Q3/Q4 evaluation-plan section (original
batch-folding-backbone family) and 2026-09-14 "T-channels bench refresh"
section (current family, promoted from Q7-noise05/Q8/Q9/Q10).

Q2/Q3/Q4 are evaluated with `cond_mode="scenario"` (data/qg_neural.py's
`_scenario_forcing_and_params`), NOT the `"true"`/`"noisy"` modes they were
*trained* with -- those always ignore the scenario label (appropriate for
training diversity), which would make an "S1 eval" scientifically meaningless
here. `cond_mode="scenario"` is deterministic and reads whatever the window's
own S0/S1 scenario wrapper designates as believed (`wind_state_corrupted`/
`da_params`, equal to the true values exactly on an S0-scenario window) --
the same biased forcing/params a DA method's dynamical model sees under S1,
giving a genuine apples-to-apples test of whether Q2/Q3/Q4 degrade under
model error the way DA baselines do.

`--lag-days`/`--noise-frac` default to 5.0/0.05 (the DA baselines' own
reference case, matching train_qg_neural.py's own current fallback default).
All four current SCHEMES entries (Q1-Q4) were trained directly at this exact
config, so the defaults evaluate every one of them in-distribution -- no
override needed. Mirrors `eval_qg_q1_lag5_noise05.py`'s cache-reuse trick:
the cached truth (`_truth_cache_path`) is keyed by the *whole* QGConfig
including obs_noise_std_frac/init_lag_days, so naively building
`QGConfig(..., obs_noise_std_frac=0.05, init_lag_days=5.0)` and calling
`make_qg_s0_s1_datasets` directly would MISS the cache and trigger a full
from-scratch truth rollout (~hours for 100 windows). Instead, the cached
truth is loaded once at its original (0.01, 1.0) key (see `CACHE_KW` --
unrelated to any scheme's own training config, just the pre-generated truth
cache's fixed lookup key), then obs/init-state are cheaply redrawn in-process
at the requested lag/noise via `QGS01Dataset._generate_obs_ic` (truth
generation is independent of the obs/IC protocol -- see `data/qg.py`'s
docstring).

Usage:
    python eval_qg_neural_s0_s1.py --cache-dir <path/to/qg_windows_1000_100_100/cache>
    python eval_qg_neural_s0_s1.py --schemes Q1 Q2 Q3 Q4 --s1-param-bias 0.1 --s1-amp-bias 0.1
"""
import argparse
import json
import os

import numpy as np
import torch

from data.normalization import load_norm_stats
from data.qg import QGConfig, QGS01Dataset, _truth_cache_path
from data.qg_neural import psi_daily, psi_to_q, q_daily
from train_qg_neural import build_model, estimate_windows, layer_summary, pooled_metrics

# Matches train_qg_neural.py's test_cfg defaults (obs_geometry/cols_per_day/
# obs_noise_std_frac/init_lag_days) -- the exact S0 reference-case test set
# every model here was evaluated against during its own training. This is
# also the cached truth's key: do not change without regenerating the cache.
DEFAULT_CACHE_DIR = ("/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/"
                     "reports/qg/outputs/qg_windows_1000_100_100/cache")
CACHE_KW = dict(nx=64, seed=20_042, num_windows=100, obs_geometry="random_columns",
               cols_per_day=4, obs_noise_std_frac=0.01, init_lag_days=1.0)

# Canonical T-channels benchmark family (2026-09-14, promoted from
# Q7-noise05/Q8/Q9/Q10 -- see PLAN.md's "T-channels bench refresh" section
# for the full backbone-evolution history). All four share
# model_type="direct_unet_tchannels" (models.monai_unet_qg2d.
# MonaiDirectUNetQGChannelTime) and were trained directly at this script's
# own current default (lag=5.0d/noise=0.05, the DA reference case), so
# they're all genuinely apples-to-apples with the DA baselines -- no
# --lag-days/--noise-frac override needed. The older batch-folding-backbone
# family (Q1/Q3/Q4/Q3-noise0.05/Q5, config/experiment/Q{1,3,4,5}_direct_unet_*.yaml)
# is retired from this eval script -- those config files/checkpoints still
# exist on disk as historical experiments, just no longer wired into SCHEMES.
SCHEMES = {
    "Q1": {
        "ckpt": "experiments/Q1_direct_unet_tchannels_s0/stage1_best.pt",
        "param_dim": 0, "cond_extra_dim": 0, "cond_mode": "none",
        "model_type": "direct_unet_tchannels",
    },
    "Q2": {
        "ckpt": "experiments/Q2_direct_unet_tchannels_s0_oracle_cond/stage1_best.pt",
        "param_dim": 3, "cond_extra_dim": 1, "cond_mode": "scenario",
        "model_type": "direct_unet_tchannels",
    },
    "Q3": {
        "ckpt": "experiments/Q3_direct_unet_tchannels_s1_noisy_cond/stage1_best.pt",
        "param_dim": 3, "cond_extra_dim": 1, "cond_mode": "scenario",
        "model_type": "direct_unet_tchannels",
    },
    "Q4": {
        "ckpt": "experiments/Q4_direct_unet_tchannels_s1_noisy_ic_cond/stage1_best.pt",
        "param_dim": 3, "cond_extra_dim": 1, "cond_mode": "scenario",
        "model_type": "direct_unet_tchannels", "include_ic": True, "ic_dim": 2,
    },
}


def _load_model(spec: dict, cfg: QGConfig, device: torch.device) -> torch.nn.Module:
    model_type = spec.get("model_type", "direct_unet")
    model = build_model(model_type, cfg, param_dim=spec["param_dim"],
                        cond_extra_dim=spec["cond_extra_dim"],
                        ic_dim=spec.get("ic_dim", 0))
    loaded = torch.load(spec["ckpt"], map_location="cpu")
    # Accepts both a bare state_dict (train_qg_neural.py's final stage1_best.pt)
    # and a full Lightning checkpoint (keys prefixed "model." for the
    # LightningModule's `self.model` submodule) -- e.g. a mid-training
    # checkpoints/stage1_best.ckpt for a run not yet finished.
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    state_dict = {(k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _eval_one(model, windows, cfg, device, norm, param_norm, forcing_norm, cond_mode,
             include_ic=False, model_type="direct_unet"):
    est_psi, est_rd = estimate_windows(
        model, windows, cfg, model_type, device, norm=norm,
        cond_mode=cond_mode, param_norm_stats=param_norm, noisy_max=1.5,
        forcing_norm_stats=forcing_norm, include_ic=include_ic)
    truth_psi = np.stack([psi_daily(w, cfg).numpy() for w in windows])
    truth_q = np.stack([q_daily(w, cfg).numpy() for w in windows])
    est_q = np.stack([
        psi_to_q(torch.tensor(est_psi[i], dtype=torch.float32), float(est_rd[i]),
                 cfg, device=device).cpu().numpy()
        for i in range(len(windows))
    ])
    rmse_psi, ev_psi = pooled_metrics(est_psi, truth_psi)
    rmse_q, ev_q = pooled_metrics(est_q, truth_q)
    return {"psi": layer_summary(rmse_psi, ev_psi, cfg), "q": layer_summary(rmse_q, ev_q, cfg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES.keys()),
                    choices=list(SCHEMES.keys()))
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--test-seed", type=int, default=20_042)
    ap.add_argument("--num-test", type=int, default=100)
    ap.add_argument("--lag-days", type=float, default=5.0,
                    help="Obs/init-state lag redrawn at eval time (default 5.0, "
                         "the DA baselines' reference case -- also the training "
                         "default for any new config that doesn't override it, "
                         "see train_qg_neural.py). All of SCHEMES' current "
                         "Q1-Q4 entries were trained at this exact value; only "
                         "pass 1.0 to instead evaluate the retired batch-"
                         "folding-backbone family (config/experiment/"
                         "Q{1,3,4,5}_direct_unet_*.yaml, not wired into "
                         "SCHEMES) in-distribution.")
    ap.add_argument("--noise-frac", type=float, default=0.05,
                    help="Obs noise std fraction redrawn at eval time (default "
                         "0.05, DA reference case / new training default). Pass "
                         "0.01 for schemes still trained at the old default.")
    ap.add_argument("--s1-param-bias", type=float, default=None,
                    help="S1 rd/rek bias fraction for the 'scenario' cond_mode's "
                         "da_params (default: QGConfig's own default, 0.15). The "
                         "actual DA S1 reference campaign (qg_da_s1_scratch.py) "
                         "uses 0.1, deliberately milder than the 0.15 default -- "
                         "pass --s1-param-bias 0.1 to match it for a true "
                         "apples-to-apples S1 comparison against the DA numbers.")
    ap.add_argument("--s1-amp-bias", type=float, default=None,
                    help="S1 wind-amplitude bias fraction (default: QGConfig's "
                         "own default, 0.15). Pass 0.1 to match qg_da_s1_scratch.py.")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--norm-stats-path", default="experiments/qg_psi_norm_stats.pt")
    ap.add_argument("--param-norm-stats-path", default="experiments/qg_param_norm_stats.pt")
    ap.add_argument("--forcing-norm-stats-path", default="experiments/qg_forcing_norm_stats.pt")
    ap.add_argument("--output", default=None,
                    help="Default: reports/qg/outputs/qg_neural_s0_s1_cross_scenario/"
                         "results_lag<L>_noise<N>[_biasB].json, derived from "
                         "--lag-days/--noise-frac[/--s1-param-bias].")
    args = ap.parse_args()
    if args.output is None:
        bias_suffix = f"_bias{args.s1_param_bias:g}" if args.s1_param_bias is not None else ""
        args.output = ("reports/qg/outputs/qg_neural_s0_s1_cross_scenario/"
                       f"results_lag{args.lag_days:g}_noise{args.noise_frac:g}{bias_suffix}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = load_norm_stats(args.norm_stats_path)
    param_norm = load_norm_stats(args.param_norm_stats_path)
    forcing_norm = load_norm_stats(args.forcing_norm_stats_path)

    # Cached truth is keyed at the original (0.01, 1.0) obs/IC protocol --
    # load it there, then cheaply redraw obs/init-state at the requested
    # lag/noise (truth generation doesn't depend on obs/IC settings, see
    # module docstring). Avoids missing the cache and triggering a full
    # from-scratch rollout for a different lag/noise.
    cache_cfg = QGConfig(**CACHE_KW)
    truth_path = _truth_cache_path(cache_cfg, args.num_test, args.cache_dir)
    print(f"Loading cached truth from {truth_path} ...", flush=True)
    base = torch.load(truth_path, map_location="cpu")[:args.num_test]
    print(f"Loaded {len(base)} base truth windows.", flush=True)

    eval_overrides = {"nx": args.nx, "seed": args.test_seed, "num_windows": args.num_test,
                      "obs_noise_std_frac": args.noise_frac, "init_lag_days": args.lag_days}
    if args.s1_param_bias is not None:
        eval_overrides["s1_param_bias"] = args.s1_param_bias
    if args.s1_amp_bias is not None:
        eval_overrides["s1_amp_bias"] = args.s1_amp_bias
    eval_cfg = QGConfig(**{**CACHE_KW, **eval_overrides})
    print(f"Redrawing obs/init-state at lag={args.lag_days}d, "
          f"noise_frac={args.noise_frac} for S0/S1 "
          f"(s1_param_bias={eval_cfg.s1_param_bias}, s1_amp_bias={eval_cfg.s1_amp_bias})...",
          flush=True)
    ds = {}
    for scenario in ("test_s0", "test_s1"):
        raw = QGS01Dataset(eval_cfg, scenario, base_windows=base).windows
        ic = QGS01Dataset._generate_obs_ic(eval_cfg, raw, list(range(len(raw))))
        ds[scenario] = [dict(w, **entry) for w, entry in zip(raw, ic)]

    results = {}
    for name in args.schemes:
        spec = SCHEMES[name]
        if not os.path.exists(spec["ckpt"]):
            print(f"  {name}: checkpoint not found at {spec['ckpt']}, skipping", flush=True)
            continue
        model = _load_model(spec, eval_cfg, device)
        print(f"Loaded {name} from {spec['ckpt']} (cond_mode={spec['cond_mode']})", flush=True)
        results[name] = {}
        for label, scenario in [("S0", "test_s0"), ("S1", "test_s1")]:
            windows = ds[scenario]
            summ = _eval_one(model, windows, eval_cfg, device, norm, param_norm, forcing_norm,
                             spec["cond_mode"], include_ic=spec.get("include_ic", False),
                             model_type=spec.get("model_type", "direct_unet"))
            results[name][label] = summ
            print(f"  {name} {label}: PSI EV={summ['psi']['pooled_ev']:.4f}  "
                  f"PV-q EV={summ['q']['pooled_ev']:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": {"nx": args.nx, "test_seed": args.test_seed,
                              "num_test": args.num_test, "lag_days": args.lag_days,
                              "noise_frac": args.noise_frac,
                              "s1_param_bias": eval_cfg.s1_param_bias,
                              "s1_amp_bias": eval_cfg.s1_amp_bias},
                  "results": results}, f, indent=2)
    print(f"\nWrote {args.output}")

    print("\n=== Summary (PSI EV / PV-q EV) ===")
    print(f"{'scheme':<6} {'S0 psi':>8} {'S0 q':>8} {'S1 psi':>8} {'S1 q':>8}")
    for name, r in results.items():
        print(f"{name:<6} {r['S0']['psi']['pooled_ev']:>8.4f} {r['S0']['q']['pooled_ev']:>8.4f} "
              f"{r['S1']['psi']['pooled_ev']:>8.4f} {r['S1']['q']['pooled_ev']:>8.4f}")


if __name__ == "__main__":
    main()
