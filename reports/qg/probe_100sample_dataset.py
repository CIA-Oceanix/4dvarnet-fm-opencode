"""Phase B smoke test: does dataset generation work at the new 100-sample,
4-year-source scale (num_windows=100, window_spacing_days=14.5)?

Checks: no crash, len(ds["test_s0"]) == 100, per-window content looks sane
(independent obs/init/corruption draws), and reports wall-clock + peak
memory for the truth-trajectory generation.
"""
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.qg import QGConfig, make_qg_s0_s1_datasets

NUM_WINDOWS = 100
WINDOW_SPACING_DAYS = 14.5


def main():
    cfg = QGConfig(nx=64, window_days=30.0, spinup_years=2.0,
                    num_windows=NUM_WINDOWS, window_spacing_days=WINDOW_SPACING_DAYS,
                    obs_geometry="random_columns", cols_per_day=4,
                    obs_noise_std_frac=0.01, seed=7)
    full_len = (cfg.num_windows - 1) * cfg.window_spacing + cfg.num_steps
    print(f"num_windows={cfg.num_windows} window_spacing_days={cfg.window_spacing_days} "
          f"full_len_steps={full_len} full_len_days={full_len / (86400.0 / cfg.dt):.1f}")

    t0 = time.time()
    ds = make_qg_s0_s1_datasets(cfg)
    dt = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"generation wall={dt:.1f}s peak_rss={peak_mb:.0f}MB")

    for scen in ("test_s0", "test_s1"):
        d = ds[scen]
        print(f"{scen}: len={len(d)}")
        assert len(d) == NUM_WINDOWS, f"expected {NUM_WINDOWS}, got {len(d)}"

    d0 = ds["test_s0"]
    col_draws = []
    for i in range(min(10, len(d0))):
        w = d0[i]
        assert w["true_state"].shape[0] == cfg.num_steps
        assert w["obs_mask"].sum() > 0
        col_draws.append(tuple(w["obs_columns"][0].tolist()))
    print("first-day obs columns (first 10 windows):", col_draws)
    assert len(set(col_draws)) > 1, "obs-column draws look identical across windows!"

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
