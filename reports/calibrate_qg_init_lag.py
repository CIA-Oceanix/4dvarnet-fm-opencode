import argparse, json, math, pathlib, numpy as np, torch
from data.qg import QGConfig
from models.qg_dynamics import QGDynamics
from evaluation.metrics import explained_variance
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--spinup-years", type=float, default=2.0)
    ap.add_argument("--window-days", type=float, default=60.0)
    ap.add_argument("--dt-grid", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
                    help="dt values in days (EV background and 30d rollout)")
    ap.add_argument("--out", type=str, default="reports/outputs/figs/qg_init_lag_calibration.png")
    args = ap.parse_args()

    print("QG init-lag EV calibration")
    print(f"nx={args.nx}, spinup={args.spinup_years}y, window={args.window_days}d")
    print(f"dt grid: {args.dt_grid} d")

    steps_per_day = round(86400.0 / 7200.0)
    spinup_steps = int(args.spinup_years * 365.0 * steps_per_day)
    window_steps = int(args.window_days * steps_per_day)

    config = QGConfig(nx=args.nx, dt=7200.0, L=1e6, beta=1.5e-11, rd=15000.0, delta=0.25, U1=0.05, U2=0.0, rek=5.787e-7, filterfac=23.6, wind_amp=1e-11, wind_tau_days=15.0, wind_sigma=250000.0, wind_cx=0.5, wind_cy=0.03, wind_drift_tau_days=10.0, wind_drift_sigma=50000.0, wind_seed=7, window_days=args.window_days, obs_interval=6, R_var=1e-12, num_windows=200, window_spacing_days=90.0, spinup_years=args.spinup_years, seed=42, obs_geometry="grid", obs_field="psi", track_repeat_days=5.0, track_advance_pts=4, track_phase_seed=0, cols_per_day=3, obs_noise_std_frac=0.05, store_targets=True, param_range=0.15, s1_param_bias=0.15, s1_amp_bias=0.15, s1_loc_sigma_frac=0.25, s1_tau_days=10.0, s1_sigma_eta_frac=0.3, init_lag_days=2.0, init_seed=7001)

    dynamics = QGDynamics(nx=args.nx, L=1e6, dt=7200.0, beta=1.5e-11, rd=15000.0, delta=0.25, U1=0.05, U2=0.0, rek=5.787e-7, filterfac=23.6, wind_amp=1e-11, wind_tau_days=15.0, wind_sigma=250000.0, wind_cx=0.5, wind_cy=0.03, wind_drift_tau_days=10.0, wind_drift_sigma=50000.0, wind_seed=7)

    full_len = (config.num_windows - 1) * config.window_spacing + config.num_steps
    traj, wind_state_full = dynamics.generate_full_trajectory(num_steps=full_len, seed=config.seed, spinup_steps=spinup_steps)

    ev_bg_grid = []
    ev_rolled_30d_grid = []

    truth_psi = dynamics.streamfunctions(traj)[:, 0, :, :].cpu().numpy()
    var_t0 = np.var(truth_psi[0,:,:], axis=0).astype(np.float64)

    for dt in args.dt_grid:
        init_steps = int(dt * steps_per_day)
        if init_steps >= window_steps:
            print(f"d={dt:5.1f}d -> window too short, skipping")
            continue

        init_state = traj[init_steps]
        init_target = traj[init_steps, :]
        if init_state.shape != init_target.shape:
            init_state = init_state.unsqueeze(0)
        ev_adj = explained_variance(init_state.cpu().numpy(), init_target.cpu().numpy(), clim_var=var_t0)
        ev_bg_grid.append(float(ev_adj.mean()))

        truth_30d_slice = traj[init_steps:init_steps+30]
        truth_60d_slice = traj[init_steps+60:init_steps+90]
        ev_rolled = explained_variance(truth_30d_slice.cpu().numpy(), truth_60d_slice.cpu().numpy(), clim_var=var_t0)
        ev_rolled_30d_grid.append(float(ev_rolled.mean()))

    out_json = "reports/outputs/figs/qg_init_lag_calibration.json"
    json_data = {"nx": args.nx, "dt_days_grid": args.dt_grid, "ev_bg": ev_bg_grid, "ev_rolled_30d": ev_rolled_30d_grid}
    pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON to {out_json}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(args.dt_grid)))
    for i, dt in enumerate(args.dt_grid):
        if i < len(ev_bg_grid):
            ax[0].plot(dt, ev_bg_grid[i], "o", color=colors[i], label=f"d={dt:d}d" if dt == args.dt_grid[0] else "")
    ax[0].axhline(1.0, color="k", linestyle="--", label="EV=1")
    ax[0].axhline(0.0, color="gray", linestyle="--", label="EV=0")
    ax[0].set_xlabel("dt (days)")
    ax[0].set_ylabel("EV (background at t0)")
    ax[0].set_title("Background EV vs Initialization Lag")
    ax[0].legend()
    ax[0].grid(True)

    for i, dt in enumerate(args.dt_grid):
        if i < len(ev_rolled_30d_grid):
            ax[1].plot(dt, ev_rolled_30d_grid[i], "o", color=colors[i], label=f"d={dt:d}d" if dt == args.dt_grid[0] else "")
    ax[1].axhline(1.0, color="k", linestyle="--", label="EV=1")
    ax[1].axhline(0.0, color="gray", linestyle="--", label="EV=0")
    ax[1].set_xlabel("dt (days)")
    ax[1].set_ylabel("EV (30-day rollout)")
    ax[1].set_title("Rollout EV vs Initialization Lag")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved PNG to {args.out}")
    plt.close(fig)

    print("\\nCalibration results:")
    print(f"{'dt (days)':<12} {'EV background':<15} {'EV 30d rollout':<15}")
    for dt, ev_bg, ev_rolled in zip(args.dt_grid, ev_bg_grid, ev_rolled_30d_grid):
        print(f"{dt:<12.1f} {ev_bg:<15.4f} {ev_rolled:<15.4f}")

if __name__ == "__main__":
    main()
