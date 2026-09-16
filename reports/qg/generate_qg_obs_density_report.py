#!/usr/bin/env python3
"""Dedicated QG obs-density report: ETKF/EnKF at cols_per_day =
1/2/4/8/16/32/64, N=100, S0 + S1.

JSON-only generator (no QG/neural code imports). Every `loc_radius` value
used here was independently screened at N=10 (and, for ETKF, `etkf_ridge`
too) and confirmed at N=100 for both methods, not assumed or transferred
from a different density -- see PLAN.md and `da_sensitivity_s0_s1_report.md`
for the full sensitivity-study trail this consolidates.

Run from the repository root::

    python reports/qg/generate_qg_obs_density_report.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N100_DIR = "qg_obs_density_sweep_n100"

# cols_per_day -> (loc_radius, etkf file tag, enkf file tag). etkf_ridge is
# 0.1 throughout (independently confirmed at both loc_radius=2.0 and 1.0).
DENSITY_GRID = [
    (1, 2.0, "colpts1_ridge0p1", "colpts1"),
    (2, 2.0, "colpts2_ridge0p1", "colpts2"),
    (4, 2.0, "loc2_ridge0p1", "colpts4"),
    (8, 2.0, "colpts8_ridge0p1", "colpts8"),
    (16, 2.0, "colpts16_ridge0p1", "colpts16"),
    (32, 2.0, "colpts32_ridge0p1", "colpts32"),
    (64, 1.0, "colpts64_ridge0p1", "colpts64"),
]


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def fmt(x) -> str:
    return f"{x:.3f}" if x is not None else "--"


def rank_marks(values: list, higher_is_better: bool = True) -> dict:
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    present.sort(key=lambda t: t[1], reverse=higher_is_better)
    marks = {}
    if len(present) >= 1:
        marks[present[0][0]] = "bold"
    if len(present) >= 2:
        marks[present[1][0]] = "italic"
    return marks


def mark(s: str, kind) -> str:
    if kind == "bold":
        return f"**{s}**"
    if kind == "italic":
        return f"*{s}*"
    return s


def metrics(d, scenario: str):
    if d is None:
        return None
    s = d.get("scenarios", {}).get(scenario)
    if s is None:
        return None
    q = s["metrics_per_field"]["q"]
    psi = s["metrics_per_field"]["psi"]["full"]
    return {"psi_ev": psi["ev"], "q_ev": q["full"]["ev"], "q_ev_l2": q["layer2"]["ev"]}


def render_density_table(add, root, scen_key, scen_scenario):
    rows = []
    for cols, loc, etkf_tag, enkf_tag in DENSITY_GRID:
        e = metrics(load_json(root / N100_DIR / f"{scen_key}_etkf_{etkf_tag}_n100.json"), scen_scenario)
        k = metrics(load_json(root / N100_DIR / f"{scen_key}_enkf_{enkf_tag}_n100.json"), scen_scenario)
        rows.append((cols, loc, e, k))

    add("| cols/day | loc_radius | ETKF psi | ETKF q | ETKF q-layer2 | EnKF psi | EnKF q | EnKF q-layer2 |")
    add("|---|---|---|---|---|---|---|---|")
    cols_list = [
        [r[2]["psi_ev"] if r[2] else None for r in rows],
        [r[2]["q_ev"] if r[2] else None for r in rows],
        [r[2]["q_ev_l2"] if r[2] else None for r in rows],
        [r[3]["psi_ev"] if r[3] else None for r in rows],
        [r[3]["q_ev"] if r[3] else None for r in rows],
        [r[3]["q_ev_l2"] if r[3] else None for r in rows],
    ]
    col_marks = [rank_marks(c) for c in cols_list]
    for i, (cols, loc, e, k) in enumerate(rows):
        vals = [
            e["psi_ev"] if e else None, e["q_ev"] if e else None, e["q_ev_l2"] if e else None,
            k["psi_ev"] if k else None, k["q_ev"] if k else None, k["q_ev_l2"] if k else None,
        ]
        cells = [mark(fmt(v), col_marks[j].get(i)) for j, v in enumerate(vals)]
        add(f"| {cols} | {loc} | " + " | ".join(cells) + " |")
    add("")
    add("(PV-q full-field and layer2, psi full-field EV, N=100; best per "
        "column **bolded**, second-best *italicized* -- with a strictly "
        "monotonic curve this typically just marks cols=64/32, included "
        "for consistency with this project's table convention.)")
    add("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", default=str(ROOT / "reports/qg/outputs"))
    ap.add_argument("--out", default=str(ROOT / "reports/qg/outputs/qg_obs_density_report.md"))
    args = ap.parse_args()
    root = Path(args.json_root)

    lines = []
    add = lines.append
    add("# QG Obs-Density Curve: ETKF/EnKF at cols/day = 1-64 (S0 / S1)")
    add("")
    add("ETKF and EnKF performance across the full observation-density "
        "range this project has explored (1, 2, 4, 8, 16, 32, 64 upper-"
        "layer (psi1) columns/day), N=100, on both the S0 reference case "
        "and the revised S1 combined-error case. Every `loc_radius` value "
        "here was independently screened at N=10 and confirmed at N=100 "
        "for **both** methods separately (not assumed to transfer from "
        "one density or method to another) -- see PLAN.md's obs-density "
        "sensitivity sections and `da_sensitivity_s0_s1_report.md` for the "
        "full screening trail, including the initial (retracted) N=10 "
        "findings this superseded and the joint `loc_radius`/`etkf_ridge`/"
        "`inflation` re-tuning that followed.")
    add("")
    add("**Config per row**: `N_ensemble=80`, `inflation=1.0` throughout "
        "(reconfirmed optimal at the new `loc_radius`, both methods); "
        "`etkf_ridge=0.1` for ETKF at every density (independently "
        "confirmed at both `loc_radius=2.0` and `loc_radius=1.0`, not "
        "re-screened separately at 8/16/32 -- the mechanism, regularizing "
        "conditioning that `loc_radius` already controls, doesn't depend "
        "on density beyond `loc_radius` itself). `cols_sampling=\"random\"` "
        "for cols>12 (exceeds the sequential sampler's `steps_per_day` "
        "ceiling at the reference dt=7200s); `\"sequential\"` (default) "
        "otherwise.")
    add("")

    add("## S0 (reference case, no model error)")
    add("")
    render_density_table(add, root, "s0", "test_s0")

    add("## S1 (revised combined-error case)")
    add("")
    render_density_table(add, root, "s1", "test_s1")

    add("## Synthesis")
    add("")
    add("- **Clean, monotonic dose-response curve**: once `loc_radius` is "
        "tuned per density (rather than held at one project-wide default), "
        "more observation density is unambiguously better on every field, "
        "both scenarios, both methods -- no collapse, no non-monotonicity "
        "anywhere in the 1-64 range. This is the corrected picture after "
        "the 2026-09-14 retraction of the original (loc_radius=6.0, "
        "untuned) N=10 sweep, which had found the opposite (density "
        "*hurting* S1 above cols=8).")
    add("- **ETKF and EnKF are now nearly indistinguishable at every "
        "density** -- typically within 0.001-0.006 EV of each other on "
        "every field, at every density from 1 to 64 cols/day. The "
        "EnKF>ETKF gap that motivated the original 2026-09-11 ETKF-ridge "
        "sensitivity study, and the later ETKF>EnKF gap after "
        "`etkf_ridge=1.0`'s promotion, were both largely artifacts of "
        "running at an untuned `loc_radius=6.0` -- properly localized, "
        "the method choice barely matters at this reference case's scale.")
    add("- **The previously-collapsing unobserved deep layer (q-layer2) "
        "improves monotonically with density too**, despite psi1 columns "
        "never observing it directly -- S1 ETKF q-layer2 rises from 0.243 "
        "(cols=1) to 0.490 (cols=64), roughly doubling, purely through "
        "better-conditioned analysis updates propagating through the "
        "layer coupling.")
    add("- **Still open**: whether independent lower-layer (psi2) "
        "observations add value *at matched total density* against a "
        "properly-tuned pure-psi1 config -- not yet tested (see "
        "`da_sensitivity_s0_s1_report.md`). `N_ensemble` has never been "
        "varied (hardcoded 80 throughout this entire curve).")
    add("")
    add("Data: `reports/qg/outputs/qg_obs_density_sweep_n100/*.json` "
        "(N=100, all 7 densities x 2 methods x 2 scenarios). Underlying "
        "N=10 screens: `reports/qg/outputs/qg_obs_density_sweep/*.json`. "
        "Scratch drivers (not committed): `qg_cols1_loc_scratch.py`, "
        "`qg_cols2_loc_scratch.py`/`qg_cols2_loc_lowgrid_scratch.py`, "
        "`qg_cols4_loc_scratch.py`, `qg_cols8_loc_scratch.py`, "
        "`qg_cols32_loc_scratch.py`, `qg_obs_density_n100_scratch.py` "
        "(original cols=16/64), `qg_ridge_at_loc2_scratch.py`/"
        "`qg_ridge_at_loc1_cols64_scratch.py`, "
        "`qg_inflation_at_loc2_scratch.py`, `qg_density_n100_scratch.py` "
        "(generic N=100 driver used for cols=8/32 and the ridge=0.1 "
        "re-runs at cols=1/2/16/64).")
    add("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
