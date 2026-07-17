#!/usr/bin/env python3
"""Generate MAOOAM datasets for 3 published configurations.

Configurations
--------------
ddv2016   : De Cruz et al. (2016)  — standard MAOOAM, (2,2)/(2,4)
vspd2019  : Vannitsem et al. (2019) — deep ocean, weak coupling
hamilton2023 : Hamilton et al. (2023) — dynamic temperature modes

Each configuration produces a .pt file with multiple trajectories,
reusing a single dynamics instance to avoid repeated qgs tensor extraction.

Usage:
    python batch/gen_maooam_comparison.py --config ddv2016
    python batch/gen_maooam_comparison.py --config vspd2019 --n-trajs 5
    python batch/gen_maooam_comparison.py --config hamilton2023
"""

import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from data.maooam import MaooamConfig, MaooamDataset


# ── Published configuration presets ────────────────────────────────────

PRESETS = {
    "ddv2016": {
        "atm_nx": 2, "atm_ny": 2,
        "occ_nx": 2, "occ_ny": 4,
        "dynamic_T": False,
        # All other params use MaooamConfig defaults (De Cruz et al. 2016)
    },
    "vspd2019": {
        "atm_nx": 2, "atm_ny": 2,
        "occ_nx": 2, "occ_ny": 4,
        "dynamic_T": False,
        # Deep ocean (h=1000 m), weak coupling (d=1.6e-8), reduced bottom friction
        "h": 1000.0,
        "r": 1e-8,
        "d": 1.6e-8,
        "sigma": 0.149,
        "gamma_oc": 4e9,
    },
    "hamilton2023": {
        "atm_nx": 2, "atm_ny": 2,
        "occ_nx": 2, "occ_ny": 4,
        "dynamic_T": True,
        # Same defaults as DDV2016 but with dynamic_T=True
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MAOOAM comparison datasets")
    parser.add_argument("--config", type=str, required=True,
                        choices=list(PRESETS.keys()),
                        help="Configuration preset")
    parser.add_argument("--n-trajs", type=int, default=5,
                        help="Number of independent trajectories (default: 5)")
    parser.add_argument("--num-windows", type=int, default=200,
                        help="Windows per trajectory (default: 200)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt path (default: experiments/maooam_comparison_{config}.pt)")
    parser.add_argument("--seed-start", type=int, default=42,
                        help="Starting seed (default: 42)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (avoids Triton cache issues on SLURM)")
    return parser.parse_args()


def make_cfg(preset_name, preset_params, device, num_windows, seed, compile=True):
    base = dict(
        device=device, compile=compile,
        dt=0.1, K=5,
        num_windows=num_windows,
        spinup_steps=5000,
        seed=seed,
    )
    base.update(preset_params)
    return MaooamConfig(**base)


def main():
    args = parse_args()
    cfg_name = args.config
    preset = PRESETS[cfg_name]
    output = args.output or f"experiments/maooam_comparison_{cfg_name}.pt"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.n_trajs))
    compile_flag = not args.no_compile
    base_cfg = make_cfg(cfg_name, preset, args.device, args.num_windows, seeds[0], compile=compile_flag)

    print(f"MAOOAM comparison dataset: {cfg_name}")
    print(f"  atm_nx/ny     = {preset.get('atm_nx',4)}/{preset.get('atm_ny',4)}")
    print(f"  occ_nx/ny     = {preset.get('occ_nx',4)}/{preset.get('occ_ny',4)}")
    print(f"  state_dim     = {base_cfg.state_dim}")
    print(f"  dynamic_T     = {preset.get('dynamic_T',False)}")
    print(f"  n_trajs       = {args.n_trajs}")
    print(f"  seeds         = {seeds[0]}..{seeds[-1]}")
    print(f"  num_windows   = {args.num_windows}")
    print(f"  device        = {args.device}")
    print(f"  output        = {output}")
    print(f"  compile       = {compile_flag}")
    print()

    print("Building dynamics (qgs tensor extraction)...")
    t0 = time.time()
    from models.maooam_torch import MaooamTorchDynamics
    dyn = MaooamTorchDynamics(
        device=args.device, compile=compile_flag,
        atm_nx=preset.get("atm_nx", 4), atm_ny=preset.get("atm_ny", 4),
        occ_nx=preset.get("occ_nx", 4), occ_ny=preset.get("occ_ny", 4),
        **{k: v for k, v in preset.items() if k not in ("atm_nx", "atm_ny", "occ_nx", "occ_ny")},
    )
    print(f"  dynamics ready in {time.time() - t0:.1f}s")
    print()

    all_windows = []
    t_total = time.time()

    for i, seed in enumerate(seeds):
        cfg_i = make_cfg(cfg_name, preset, args.device, args.num_windows, seed, compile=compile_flag)
        print(f"[{i+1}/{args.n_trajs}] Generating trajectory (seed={seed})...")
        t0 = time.time()
        ds = MaooamDataset(cfg_i, scenario="S0", dynamics=dyn)
        all_windows.extend(ds.windows)
        print(f"  -> {len(ds)} windows in {time.time() - t0:.1f}s")

    total_time = time.time() - t_total
    print(f"\nTotal: {len(all_windows)} windows in {total_time:.1f}s")

    torch.save({
        "config": cfg_name,
        "windows": all_windows,
        "preset_params": preset,
        "state_dim": base_cfg.state_dim,
        "n_trajs": args.n_trajs,
        "seeds": seeds,
    }, output)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()