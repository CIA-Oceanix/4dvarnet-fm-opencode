#!/usr/bin/env python3
"""L96 fast-Y observation-density generalization report builder.

Consumes ``eval_obs_density_l96.py``'s results JSON (default:
``experiments/l96_obs_density_generalization/results.json``), schema
``results[method][case][str(keep_k)] = {rmse: {mean,std,n}, ev: {all_obs:
{mean,std,n}, slow: {...}, obs_fast: {...}}}``.

Generates ``reports/l96/outputs/l96_obs_density_generalization.md``: a
RMSE/EV table (rows = keep_k, columns = method x case, mean +/- std across
the independent mask-redraw repeats) plus a degradation table (RMSE at each
keep_k relative to the keep_k=16 full-density baseline) for each method.

Distinct from the pre-existing ``generate_l96_obs_density_report.py``, which
covers the unrelated DA-baseline slow-only-vs-obsj2 study (a fixed reduced
density, not a randomly-redrawn one, and DA baselines rather than the 4
best-of-subcategory neural/SDA schemes).
"""
import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "experiments/l96_obs_density_generalization/results.json"
DEFAULT_OUT = ROOT / "reports/l96/outputs/l96_obs_density_generalization.md"

METHOD_LABELS = {
    "directunet": "DirectUNet-L(monai,cos)",
    "cfm": "CFM-M(monai,flat)",
    "sda3": "SDA3(monai)",
    "directunet_sda3": "DirectUNet+SDA3",
}
METHOD_ORDER = ["directunet", "cfm", "sda3", "directunet_sda3"]
CASES = ["s0", "s1"]


def load_json(path: Path):
    if not path.exists():
        logger.warning("JSON not found: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def fmt_mean_std(entry, missing="--", ndigits=3):
    if not entry:
        return missing
    mean, std = entry.get("mean"), entry.get("std")
    if mean is None or (isinstance(mean, float) and (math.isnan(mean) or math.isinf(mean))):
        return missing
    return f"{mean:.{ndigits}f}±{std:.{ndigits}f}"


def _cell(data, method, case, keep_k, metric_path):
    entry = (((data or {}).get(method) or {}).get(case) or {}).get(str(keep_k))
    if entry is None:
        return None
    node = entry
    for key in metric_path:
        node = (node or {}).get(key)
    return node


def build_rmse_ev_table(data, methods, cases, keep_k_values):
    lines = []
    header = ["keep_k"] + [
        f"{METHOD_LABELS.get(m, m)} ({c.upper()})" for m in methods for c in cases
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for keep_k in keep_k_values:
        row = [str(keep_k)]
        for m in methods:
            for c in cases:
                rmse = fmt_mean_std(_cell(data, m, c, keep_k, ["rmse"]))
                ev = fmt_mean_std(_cell(data, m, c, keep_k, ["ev", "all_obs"]))
                row.append(f"{rmse} / EV {ev}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_degradation_table(data, methods, cases, keep_k_values):
    """RMSE at each keep_k relative to the keep_k=max(keep_k_values) (full
    density) baseline, per method/case -- ``1.00x`` at the baseline row,
    growing as fast-Y density drops."""
    baseline_k = max(keep_k_values)
    lines = []
    header = ["keep_k"] + [
        f"{METHOD_LABELS.get(m, m)} ({c.upper()})" for m in methods for c in cases
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for keep_k in keep_k_values:
        row = [str(keep_k)]
        for m in methods:
            for c in cases:
                base_rmse = _cell(data, m, c, baseline_k, ["rmse", "mean"])
                cur_rmse = _cell(data, m, c, keep_k, ["rmse", "mean"])
                if base_rmse is None or cur_rmse is None or base_rmse == 0:
                    row.append("--")
                else:
                    row.append(f"{cur_rmse / base_rmse:.3f}x")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build the L96 obs-density generalization report")
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    payload = load_json(Path(args.json))
    data = (payload or {}).get("results", {})
    methods = [m for m in METHOD_ORDER if m in data] or list(data.keys())
    keep_k_values = sorted(
        (payload or {}).get("keep_k_values", [16, 8, 4, 0]), reverse=True
    )
    cases = (payload or {}).get("cases", CASES)

    n_repeats = (payload or {}).get("n_repeats", "?")
    sampling = (payload or {}).get("sampling", {})

    lines = [
        "# L96 Fast-Y Observation-Density Generalization",
        "",
        "Inference-time-only generalization test (no retraining) for the 4",
        "best-of-subcategory L96 monai-backbone schemes -- DirectUNet-L(cos),",
        "CFM-M(flat), SDA3, DirectUNet+SDA3 -- under randomly reduced fast-Y",
        "observation density. Of the 16 canonical fast-Y channels (2 per slow",
        "node), only `keep_k` are kept, **redrawn independently at every",
        "observation time** within each window; the 8 slow-X channels always",
        "stay fully observed. `keep_k=16` is the full-density sanity check and",
        "should reproduce the canonical `l96_consolidated_benchmark.md` numbers.",
        "",
        "**Caveat:** DirectUNet/CFM consume `obs` only via",
        "`torch.nan_to_num(obs, nan=0.0)`, with no separate mask channel --",
        "trained only on whole-timestep NaN blocks, never partial-channel NaN",
        "within an observed timestep. A dropped fast-Y channel is therefore",
        "indistinguishable from a genuine near-zero observation for these two",
        "schemes: this is genuinely out-of-distribution, not fixable without",
        "retraining. SDA's guided-sampling cost, by contrast, never conditions",
        "the network on raw obs -- the dropped channels are cleanly excluded",
        "from the guidance cost term, with no zero-imputation ambiguity.",
        "",
        f"Each (method, case, keep_k) cell is `n_repeats={n_repeats}` independent",
        "seed reruns (fresh mask redraw, and fresh sampling noise for the",
        "stochastic methods); values below are mean +/- std across those repeats.",
        f"Sampling settings: `{sampling}`.",
        "",
        "## RMSE / EV(all_obs) vs. fast-Y density",
        "",
        build_rmse_ev_table(data, methods, cases, keep_k_values),
        "",
        "## RMSE degradation vs. full density (keep_k=16)",
        "",
        build_degradation_table(data, methods, cases, keep_k_values),
        "",
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    logger.info(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
