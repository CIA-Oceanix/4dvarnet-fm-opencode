#!/usr/bin/env python3
"""Generate a benchmark table for L96 joint state-parameter neural models.

Dir structure:
  {EXP_DIR}/L7/L8/L9/joint_neural_eval*.json (mm), joint_estimates*.npz

Outputs:
  reports/l96/outputs/l96_joint_neural_benchmark.md

DA rows shown as '--' (deferred joint DA baseline regeneration)."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EXP_DIR = ROOT / "experiments"

L96_JOINT_PARAM_NAMES = ("F", "c1", "hx", "eps", "w1", "w2", "w3", "w4")
PD = len(L96_JOINT_PARAM_NAMES)


def load_joint_eval(exp_dir: Path, case: str, m: int, k: int):
    """Load joint neural eval JSON for a given model/case/ensemble."""
    patterns = [
        exp_dir / "joint_neural_eval.json",
        exp_dir / f"joint_neural_eval_ens{m}_k{k}.json",
    ]
    for pat in patterns:
        if pat.exists():
            with open(pat) as f:
                return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate L96 joint neural benchmark report")
    parser.add_argument("--exp-dir", type=str, default=str(EXP_DIR),
                        help="Experiments directory")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "reports/l96/outputs/l96_joint_neural_benchmark.md"),
                        help="Output markdown report")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_defs = {
        "L7_joint_cfm_s0s1": {"type": "JointCFM (tau=0)", "desc": "CFM with tau=0 training"},
        "L8_joint_direct_unet_s0s1": {"type": "JointDirectUNet", "desc": "Single-stage UNet"},
        "L9_joint_cfm_s0s1_multitau": {"type": "JointCFM (multi-tau)", "desc": "CFM with random tau"},
    }

    markdown = []
    markdown.append("# L96 Joint State-Parameter Neural Estimation Benchmark")
    markdown.append("")
    markdown.append("**System:** Lorenz-96 (two-scale, state_dim=24, Obs30)")
    markdown.append("**Models:** JointCFM + JointDirectUNet (state + 8 params)")
    markdown.append("")
    markdown.append("DA baselines (EnKF/ETKF/Strong-4DVar + joint variants) were **not** run in this generation cycle.")
    markdown.append("* DA rows shown as --")
    markdown.append("")
    markdown.append("---")
    markdown.append("")
    markdown.append("## Benchmarked models")
    markdown.append("")
    markdown.append("| ID | Type | Description | Stage |")
    markdown.append("|---|---|---|---|")
    for exp_name, model_def in model_defs.items():
        markdown.append(f"| {exp_name} | {model_def['type']} | {model_def['desc']} | Stage 1 |")

    markdown.append("")
    markdown.append("---")
    markdown.append("")
    markdown.append("## Single-sample results (n_members=1, k=1)")
    markdown.append("")
    markdown.append("| ID | S0 RMSE | S1 RMSE |")
    markdown.append("|---|---|---|")
    for exp_name in model_defs.keys():
        exp_dir = EXP_DIR / exp_name
        if not exp_dir.exists():
            continue
        for case in ["s0", "s1"]:
            data = load_joint_eval(exp_dir, case, 1, 1)
            if data is None:
                markdown.append(f"| {exp_name} {case.upper()} | -- | -- |")
                continue
            m = data["metrics"][case]
            markdown.append(f"| {exp_name} {case.upper()} | {m['rmse']:.6f} | -- |")

    markdown.append("")
    markdown.append("*RMSE/EV shown for S0 only; S1 requires multi-member inference (not shown here).*")

    with open(output_path, "w") as f:
        f.write("\n".join(markdown))

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
