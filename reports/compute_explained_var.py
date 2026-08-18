#!/usr/bin/env python3
"""Compute explained variance from DA sweep JSON results + climatological variance."""
import json, os, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
CLIM_PATH = os.path.join(BASE, "reports", "outputs", "l96_clim_var.json")

with open(CLIM_PATH) as f:
    clim = json.load(f)

clim_var = np.array(clim["var_per_var"])  # shape (40,)
clim_std = np.sqrt(clim_var)

def explained_variance(rmse_per_var, clim_var):
    mse = np.array(rmse_per_var) ** 2
    expl_var = 1.0 - mse / clim_var
    return expl_var

def report(label):
    path = os.path.join(EXP_DIR, f"l96_sweep_{label}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    cfg = data["config"]
    pb = cfg["param_bias"]
    fsb = cfg["forcing_state_bias"]
    ss = cfg.get("s1_single_scale", False)
    ni = cfg.get("s1_no_inflation", False)
    lines = []
    lines.append(f"{label} (pb={pb}, fsb={fsb}" + (f", single-scale={ss}" if ss else "") + (f", no-infl={ni}" if ni else "") + ")")
    for case in ["s0", "s1"]:
        lines.append(f"  {case.upper()}:")
        for method in ["Strong-4DVar", "EnKF", "ETKF"]:
            rmse = np.array(data["results"][case][method]["per_var_mean"])
            mse = rmse ** 2
            ev = explained_variance(rmse, clim_var)
            overall_ev = 1.0 - np.mean(mse) / float(clim["overall_avg_var"])
            slow_ev = 1.0 - np.mean(mse[:8]) / float(clim["slow_avg_var"])
            fast_ev = 1.0 - np.mean(mse[8:]) / float(clim["fast_avg_var"])
            lines.append(f"    {method:<18} RMSE={np.mean(rmse):.4f}  EV_overall={overall_ev:.4f}  EV_slow={slow_ev:.4f}  EV_fast={fast_ev:.4f}")
    return "\n".join(lines)

if __name__ == "__main__":
    for label in ["a1","a2","a3","a4","a5","a6","b1","b2","b3","b4"]:
        out = report(label)
        if out:
            print(out)
            print()

    # Summary table
    print("=" * 90)
    print(f"{'Label':<8} {'Case':<5} {'Method':<18} {'RMSE':<8} {'EV_overall':<12} {'EV_slow':<12} {'EV_fast':<12}")
    print("-" * 90)
    for label in ["a1","a2","a3","a4","a5","a6","b1","b2","b3","b4"]:
        path = os.path.join(EXP_DIR, f"l96_sweep_{label}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for case in ["s0", "s1"]:
            for method in ["Strong-4DVar", "EnKF", "ETKF"]:
                rmse = np.array(data["results"][case][method]["per_var_mean"])
                mse = rmse ** 2
                overall_ev = 1.0 - np.mean(mse) / float(clim["overall_avg_var"])
                slow_ev = 1.0 - np.mean(mse[:8]) / float(clim["slow_avg_var"])
                fast_ev = 1.0 - np.mean(mse[8:]) / float(clim["fast_avg_var"])
                r = np.mean(rmse)
                print(f"{label:<8} {case:<5} {method:<18} {r:<8.4f} {overall_ev:<12.4f} {slow_ev:<12.4f} {fast_ev:<12.4f}")
    print("=" * 90)