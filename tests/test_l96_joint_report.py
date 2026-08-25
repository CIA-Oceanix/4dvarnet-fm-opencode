import json
import subprocess
import sys

import pytest

REPORT = "reports/l96/generate_l96_joint_report.py"
PARAMS = ["F", "c1", "hx", "eps", "w1", "w2", "w3", "w4"]


def _da_entry(joint: bool):
    e = {
        "state_rmse": {"slow": 0.7, "obs_fast": 1.0, "mean": 0.85},
        "ev": {"slow": 0.6, "obs_fast": 0.4, "mean": 0.5},
        "es": {"slow": 0.4, "obs_fast": 0.6, "mean": 0.5},
    }
    e["param_rmse"] = {k: 0.5 for k in PARAMS} if joint else None
    return e


def _write_da(path):
    da = {}
    for case in ("S0", "S1"):
        da[case] = {m: _da_entry(m.startswith("Joint-"))
                    for m in ("EnKF", "Joint-EnKF", "ETKF", "Joint-ETKF",
                              "Strong-4DVar", "Joint-Strong-4DVar")}
    with open(path, "w") as f:
        json.dump(da, f, indent=2)
    return da


def _write_neural(exp_dir, rmse=0.62, prmse=0.3):
    exp_dir.mkdir(parents=True, exist_ok=True)
    mets = {}
    for c in ("s0", "s1"):
        mets[c] = {
            "rmse": rmse,
            "groups": {"slow": 0.4, "obs_fast": 0.8, "all_obs": rmse},
            "ev": {"groups": {"all_obs": 0.6}},
            "es": {"groups": {"all_obs": 0.4}},
            "param_rmse": {k: prmse for k in PARAMS},
            "param_rmse_mean": prmse,
        }
    with open(exp_dir / "joint_neural_eval.json", "w") as f:
        json.dump({"metrics": mets}, f, indent=2)


def test_report_merges_neural_and_da(tmp_path):
    da_path = tmp_path / "da.json"
    _write_da(da_path)
    for exp in ("L7_joint_cfm_s0s1", "L8_joint_direct_unet_s0s1",
                "L9_joint_cfm_s0s1_multitau"):
        _write_neural(tmp_path / exp)

    out_dir = tmp_path / "out"
    subprocess.run(
        [sys.executable, REPORT, "--da-json", str(da_path),
         "--experiments-dir", str(tmp_path), "--out-dir", str(out_dir)],
        check=True, capture_output=True,
    )
    md = (out_dir / "l96_joint_benchmark.md").read_text()

    assert "JointCFM τ=0" in md and "JointDirectUNet" in md and "JointCFM multi-τ" in md
    assert "Joint-EnKF" in md and "Joint-Strong-4DVar" in md
    assert "## Parameter RMSE" in md
    # every metric cell rendered as a number (not missing)
    assert "0.6200" in md and "0.8500" in md
    assert "--" not in md.split("| Method |")[0]  # header intact


def test_report_missing_artifacts_renders_dashes(tmp_path):
    da_path = tmp_path / "da.json"
    _write_da(da_path)
    # no neural JSONs written
    out_dir = tmp_path / "out"
    subprocess.run(
        [sys.executable, REPORT, "--da-json", str(da_path), "--out-dir", str(out_dir)],
        check=True, capture_output=True,
    )
    md = (out_dir / "l96_joint_benchmark.md").read_text()
    # neural rows exist but missing values render as '--'
    assert "| **JointCFM τ=0** | neural | -- | -- |" in md
