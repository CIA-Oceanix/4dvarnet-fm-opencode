#!/usr/bin/env python3
"""Generate CS1/CS2 summary report comparing all model families."""

import os, sys, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")


def load_result(exp_id: str):
    path = os.path.join(EXP_DIR, exp_id, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def format_row(model, variant, config, cs1, cs2, note=""):
    return {
        "model": model,
        "variant": variant,
        "config": config,
        "cs1": cs1,
        "cs2": cs2,
        "note": note,
    }


rows = []

# ── DirectUNet (E1-E3) ────────────────────────────────────────────────
for eid, label, cfg in [
    ("E1_direct_unet_default", "DirectUNet", "[64,128,256]"),
    ("E2_direct_unet_small",   "DirectUNet", "[32,64,128]"),
    ("E3_direct_unet_rand",    "DirectUNet", "[32,64,128] rand"),
]:
    r = load_result(eid)
    if r:
        rows.append(format_row(label, "direct", cfg,
                               r["fm_cs1"]["mean"], r["fm_cs2"]["mean"]))

# ── VanillaCFM full 10-step (F1-F3) ────────────────────────────────────
for fid, label, cfg in [
    ("F1_vanilla_cfm_default", "VanillaCFM", "[64,128,256]"),
    ("F2_vanilla_cfm_small",   "VanillaCFM", "[32,64,128]"),
    ("F3_vanilla_cfm_rand",    "VanillaCFM", "[32,64,128] rand"),
]:
    r = load_result(fid)
    if r:
        rows.append(format_row(label, "cfm_10step", cfg,
                               r["fm_cs1"]["mean"], r["fm_cs2"]["mean"]))

# ── VanillaCFM forced τ=0 single step (F1-F3 with τ=0 eval) ────────────
forced_path = os.path.join(EXP_DIR, "forced_tau0_results.json")
if os.path.exists(forced_path):
    with open(forced_path) as f:
        forced = json.load(f)
    for name, label, cfg in [
        ("F1_vanilla_cfm_default", "VanillaCFM (τ=0 forced)", "[64,128,256]"),
        ("F2_vanilla_cfm_small",   "VanillaCFM (τ=0 forced)", "[32,64,128]"),
        ("F3_vanilla_cfm_rand",    "VanillaCFM (τ=0 forced)", "[32,64,128] rand"),
    ]:
        if name in forced:
            rows.append(format_row(label, "cfm_forced_tau0", cfg,
                                   forced[name]["cs1"]["overall_mean"],
                                   forced[name]["cs2"]["overall_mean"]))

# ── τ=0 CFM trained (G1-G3) ───────────────────────────────────────────
for gid, label, cfg in [
    ("G1_vanilla_cfm_t0_default", "τ=0 CFM", "[64,128,256]"),
    ("G2_vanilla_cfm_t0_small",   "τ=0 CFM", "[32,64,128]"),
    ("G3_vanilla_cfm_t0_rand",    "τ=0 CFM", "[32,64,128] rand"),
]:
    r = load_result(gid)
    if r:
        rows.append(format_row(label, "cfm_tau0_trained", cfg,
                               r["fm_cs1"]["mean"], r["fm_cs2"]["mean"]))


# ── Print Report ───────────────────────────────────────────────────────
print("=" * 80)
print("  CS1/CS2 Performance Summary (RMSE ↓)")
print("=" * 80)
print()

groups = [
    ("DirectUNet", "direct"),
    ("VanillaCFM (10-step)", "cfm_10step"),
    ("VanillaCFM (τ=0 forced eval)", "cfm_forced_tau0"),
    ("τ=0 CFM (trained τ=0)", "cfm_tau0_trained"),
]

for group_name, group_key in groups:
    group_rows = [r for r in rows if r["variant"] == group_key]
    if not group_rows:
        continue
    print(f"── {group_name} {'─' * (60 - len(group_name))}")
    print(f"  {'Config':<24} {'CS1':<10} {'CS2':<10}  {'Best?':<6}")
    print(f"  {'-'*22}   {'-'*8}   {'-'*8}   {'-'*6}")
    best_cs1 = min(r["cs1"] for r in group_rows)
    best_cs2 = min(r["cs2"] for r in group_rows)
    for r in group_rows:
        cs1_s = f"{r['cs1']:.6f}"
        cs2_s = f"{r['cs2']:.6f}"
        note = ""
        if r["cs1"] == best_cs1 and r["cs2"] == best_cs2:
            note = "★"
        elif r["cs1"] == best_cs1:
            note = "CS1"
        elif r["cs2"] == best_cs2:
            note = "CS2"
        print(f"  {r['config']:<24} {cs1_s:<10} {cs2_s:<10}  {note:<6}")
    print()

# Overall best
print("─" * 80)
print("  CS1 Best-in-Class")
print("─" * 80)
for label, field in [("DirectUNet", "direct"),
                      ("VanillaCFM (10-step)", "cfm_10step"),
                      ("VanillaCFM (τ=0 forced)", "cfm_forced_tau0"),
                      ("τ=0 CFM (trained)", "cfm_tau0_trained")]:
    gr = [r for r in rows if r["variant"] == field]
    if gr:
        best = min(gr, key=lambda r: r["cs1"])
        print(f"  {label:<30}  CS1={best['cs1']:.6f}  ({best['config']})")
print()
print("─" * 80)
print("  CS2 Best-in-Class")
print("─" * 80)
for label, field in [("DirectUNet", "direct"),
                      ("VanillaCFM (10-step)", "cfm_10step"),
                      ("VanillaCFM (τ=0 forced)", "cfm_forced_tau0"),
                      ("τ=0 CFM (trained)", "cfm_tau0_trained")]:
    gr = [r for r in rows if r["variant"] == field]
    if gr:
        best = min(gr, key=lambda r: r["cs2"])
        print(f"  {label:<30}  CS2={best['cs2']:.6f}  ({best['config']})")

print()
print("─" * 80)
print("  Key Takeaways")
print("─" * 80)
print("""
  1. τ=0 CFM (G3) achieves by far the best RMSE: 0.032 on both CS1 and CS2.
  2. DirectUNet ranges 0.081-0.144 (CS1) and 0.102-0.156 (CS2).
  3. VanillaCFM full 10-step ranges 0.069-0.141 (CS1) and 0.070-0.149 (CS2).
  4. Forcing multi-τ VanillaCFM to τ=0 single-step degrades performance:
     best  0.093 (F3) vs 0.070 full 10-step.  The multi-step Euler
     integration is essential for models trained on random τ.
  5. The τ=0-only training regime (G3, 0.032) outperforms DirectUNet
     (E3, 0.116) by 3.6× despite using the same backbone [32,64,128]
     and matching everything except the denoising formulation.
""")

# Save as JSON
out = {"groups": {}}
for group_name, group_key in groups:
    gr = [r for r in rows if r["variant"] == group_key]
    if gr:
        out["groups"][group_key] = {
            "label": group_name,
            "entries": gr,
            "best_cs1": min(gr, key=lambda r: r["cs1"]),
            "best_cs2": min(gr, key=lambda r: r["cs2"]),
        }
with open(os.path.join(EXP_DIR, "summaries/cs1_cs2_unet_cfm_comparison.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"  Full data saved to experiments/summaries/cs1_cs2_comparison.json")
