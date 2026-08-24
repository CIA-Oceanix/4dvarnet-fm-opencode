#!/usr/bin/env python3
"""Re-run L96 DA baselines with the fixed Energy Score accumulator.

Each entry regenerates one historical baseline cache from its documented CLI
specification (reconstructed from the generating sbatch scripts), writing to a
parallel ``*_esfix*`` cache instead of touching the original, then validates:

- RMSE/EV groups must match the original within GPU-nondeterminism tolerance
  (a mismatch means the reconstructed config is wrong -> do NOT swap).
- Strong-4DVar (deterministic, N=1) ES must match the original closely
  (the accumulator bug vanishes at N=1).
- EnKF/ETKF ES must CHANGE (proof the fix is exercised).

Swap originals <-> esfix files manually after reviewing the validation output.
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")

S0C_RANDOMIZE = ('{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, '
                 '"c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, '
                 '"h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, '
                 '"hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, '
                 '"eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, '
                 '"fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}')
FW6_RANDOMIZE = ('{"fast_weights": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, '
                 '"F": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, '
                 '"c1": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, '
                 '"h": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, '
                 '"hx": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}, '
                 '"eps": {"randomized": true, "noise": 0.2, "biased": false, "bias": 0.0}}')

DWS500 = ["--da-window-steps", "500", "--num-test-windows", "200", "--t-max", "3.0",
          "--enkf-inflation", "2.0", "--etkf-inflation", "2.0", "--obs-j", "2"]

# orig cache filename -> (esfix cache filename, extra argv, dataset cache tag)
SPECS = {
    "l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.json": (
        "l96_baselines_dws500_s0c_esfix_inf2.0_etkf_inf2.0_obsj2_int100_fw.json",
        DWS500 + ["--obs-interval", "100", "--suffix", "_s0c_esfix",
                  "--randomize", S0C_RANDOMIZE, "--skip-weak"],
        "",  # reuse the canonical s0c int100 nwin200 dataset cache as-is
    ),
    "l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int200_fw.json": (
        "l96_baselines_dws500_s0c_esfix_inf2.0_etkf_inf2.0_obsj2_int200_fw.json",
        DWS500 + ["--obs-interval", "200", "--suffix", "_s0c_esfix",
                  "--randomize", S0C_RANDOMIZE, "--skip-weak"],
        "_s0c_int200",
    ),
    "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100_fw.json": (
        "l96_baselines_dws500_esfix_inf2.0_etkf_inf2.0_obsj2_int100_fw.json",
        DWS500 + ["--obs-interval", "100", "--suffix", "_esfix",
                  "--randomize", FW6_RANDOMIZE, "--skip-weak"],
        "_fw6_int100",
    ),
    "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int200_fw.json": (
        "l96_baselines_dws500_esfix_inf2.0_etkf_inf2.0_obsj2_int200_fw.json",
        DWS500 + ["--obs-interval", "200", "--suffix", "_esfix",
                  "--randomize", FW6_RANDOMIZE, "--skip-weak"],
        "_fw6_int200",
    ),
    "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100.json": (
        "l96_baselines_dws500_esfix_inf2.0_etkf_inf2.0_obsj2_int100.json",
        DWS500 + ["--obs-interval", "100", "--suffix", "_esfix", "--skip-weak"],
        "_legacy_int100",
    ),
    "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int200.json": (
        "l96_baselines_dws500_esfix_inf2.0_etkf_inf2.0_obsj2_int200.json",
        DWS500 + ["--obs-interval", "200", "--suffix", "_esfix", "--skip-weak"],
        "_legacy_int200",
    ),
    "l96_baselines_dws50_inf2.0_etkf_inf2.0_obsj2_int200.json": (
        "l96_baselines_dws50_esfix_inf2.0_etkf_inf2.0_obsj2_int200.json",
        ["--da-window-steps", "50", "--num-test-windows", "20", "--t-max", "3.0",
         "--enkf-inflation", "2.0", "--etkf-inflation", "2.0", "--obs-j", "2",
         "--obs-interval", "200", "--suffix", "_esfix", "--skip-weak", "--skip-strong"],
        "_dws50_legacy_int200",
    ),
    "l96_baselines_dws50_inf2.0_etkf_inf2.0_obsj2_int200_fw.json": (
        "l96_baselines_dws50_esfix_inf2.0_etkf_inf2.0_obsj2_int200_fw.json",
        ["--da-window-steps", "50", "--num-test-windows", "20", "--t-max", "3.0",
         "--enkf-inflation", "2.0", "--etkf-inflation", "2.0", "--obs-j", "2",
         "--obs-interval", "200", "--suffix", "_esfix", "--randomize", FW6_RANDOMIZE,
         "--skip-weak", "--skip-strong"],
        "_dws50_fw6_int200",
    ),
}

RMSE_TOL = 2e-2      # relative, GPU nondeterminism allowance (~1% observed drift)
DET_ES_TOL = 0.02    # relative, deterministic N=1 anchor (GPU nondeterminism vs backfilled MAE)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def validate(orig_path: str, esfix_path: str) -> dict:
    orig = json.load(open(orig_path))
    new = json.load(open(esfix_path))
    report = {}
    ok_all = True
    for case in ("s0", "s1"):
        for method, blk in orig.get(case, {}).items():
            nb = new.get(case, {}).get(method)
            if nb is None:
                report[f"{case}/{method}"] = {"status": "MISSING"}
                ok_all = False
                continue
            rmse_ok = _rel(nb["mean"], blk["mean"]) < RMSE_TOL
            ev_ok = all(
                _rel(nb["ev"]["groups"].get(g, float("nan")),
                     blk["ev"]["groups"].get(g, float("nan"))) < RMSE_TOL
                for g in ("slow", "obs_fast", "all_obs")
            )
            es_old_raw = blk.get("es")
            es_new_raw = nb.get("es")
            det_anchor = method == "Strong-4DVar"
            if es_old_raw is None:
                es_ok = True
                es_old = None
                es_new = es_new_raw["groups"]["all_obs"] if es_new_raw else None
            elif es_new_raw is None:
                es_ok = False
                es_old = es_old_raw["groups"]["all_obs"]
                es_new = None
            else:
                es_old = es_old_raw["groups"]["all_obs"]
                es_new = es_new_raw["groups"]["all_obs"]
                es_ok = (
                    abs(es_new - es_old) < DET_ES_TOL * abs(es_old)
                    if det_anchor else abs(es_new - es_old) > 1e-4
                )
            status = "OK" if (rmse_ok and ev_ok and es_ok) else "FAIL"
            if status == "FAIL":
                ok_all = False
            report[f"{case}/{method}"] = {
                "status": status,
                "rmse_old": round(blk["mean"], 6), "rmse_new": round(nb["mean"], 6),
                "ev_old": round(blk["ev"]["groups"]["all_obs"], 6),
                "ev_new": round(nb["ev"]["groups"]["all_obs"], 6),
                "es_old": round(es_old, 6) if es_old is not None else None,
                "es_new": round(es_new, 6) if es_new is not None else None,
            }
    return {"ok": ok_all, "checks": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, choices=sorted(SPECS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    esfix_name, extra_argv, data_tag = SPECS[args.cache]
    cmd = [sys.executable, os.path.join(BASE, "evaluate_all_l96.py"),
           "--device", args.device, *extra_argv]
    if data_tag:
        cmd += ["--data-cache-tag", data_tag]
    print(f"[esfix] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    if args.skip_validate:
        return
    orig_path = os.path.join(EXP_DIR, args.cache)
    esfix_path = os.path.join(EXP_DIR, esfix_name)
    rep = validate(orig_path, esfix_path)
    out = os.path.join(EXP_DIR, esfix_name.replace(".json", "_validation.json"))
    with open(out, "w") as f:
        json.dump({"orig": args.cache, "esfix": esfix_name, **rep}, f, indent=2)
    print(f"\n[esfix] validation {'PASS' if rep['ok'] else 'FAIL'} -> {out}")
    for key, chk in rep["checks"].items():
        print(f"  {key:<22} {chk['status']:<7} RMSE {chk['rmse_old']}->{chk['rmse_new']} "
              f"ES {chk.get('es_old', 'N/A')}->{chk.get('es_new', 'N/A')}")
    if not rep["ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
