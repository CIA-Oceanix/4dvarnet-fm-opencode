#!/usr/bin/env python3
"""L96 per-channel normalization ablation report builder.

Consumes ``neural_eval.json`` (the eval JSON written by ``eval_neural_l96.py``,
schema: ``metrics[case] = {rmse, groups: {slow, obs_fast, all_obs}, ev:
{groups}, es: {groups}, num_samples}``, plus ``metrics["degradation"]``) from
three experiment dirs per model:

- **Original**   -- the existing benchmark checkpoint, no normalization.
- **Phase 1**    -- same frozen checkpoint, inference-only normalize-input/
  denormalize-output (``--normalize-stats``); expected to look worse (OOD
  input for weights tuned on raw-scale obs).
- **Phase 2**    -- a checkpoint retrained from scratch with ``data.normalize:
  true`` (the actual research question).

Missing files (e.g. Phase 2 before its training run completes) render as
``--`` without crashing.
"""
import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# model id -> (Original exp dir override, Phase 1 exp dir, Phase 2 exp dir, description)
#
# L3's top-level ``neural_eval.json`` is an older single-sample run; the
# model's actual benchmark number (and what Phase 1/2 are sampled as) is the
# ens30 x 10 Euler-step ensemble, so its Original row is pointed at a
# freshly-regenerated ens30 run (same invocation as Phase 1/2, just without
# --normalize-stats) for an apples-to-apples comparison.
MODEL_DEFS = {
    "L1b_direct_unet_s0s1": {
        "original_dir": "L1b_direct_unet_s0s1",
        "phase1_dir": "L1b_direct_unet_s0s1_phase1",
        "phase2_dir": "L1b_direct_unet_s0s1_norm",
        "desc": "DirectUNet, single-pass obs -> state regression. Hidden [64,128,256], 200 epochs.",
    },
    "L1b_monai_unet_s0s1": {
        "original_dir": "L1b_monai_unet_s0s1__no_baseline",
        "phase1_dir": "L1b_monai_unet_s0s1__no_baseline",
        "phase2_dir": "L1b_monai_unet_s0s1_norm",
        "desc": "MonaiUNet1D-backed analogue of L1b: models.direct_unet.DirectUNet -> "
                "models.monai_unet_adapter.MonaiDirectUNet (wraps MONAI's DiffusionModelUNet, "
                "FiLM-style timestep conditioning + real ResBlocks, vs UNet1D's additive-only "
                "conditioning). Identical data setup, normalization, and hyperparameters as "
                "L1b_direct_unet_s0s1_norm (single-pass obs -> state regression, hidden "
                "[64,128,256], 200 epochs, lr=0.001, gradient_clip_val=10.0) -- only the backbone "
                "differs. No unnormalized baseline was trained for this model (testing "
                "normalization-robustness specifically was the point), so Original/Phase 1 are --.",
    },
    "L3_vanilla_cfm_s0s1": {
        "original_dir": "L3_vanilla_cfm_s0s1_ens30_original",
        "phase1_dir": "L3_vanilla_cfm_s0s1_phase1",
        "phase2_dir": "L3_vanilla_cfm_s0s1_norm",
        "desc": "VanillaCFM, standard multi-tau conditional flow matching, ens30 x 10 Euler steps. "
                "Hidden [64,128,256], 400 epochs.",
    },
}

CASES = ["s0", "s1"]
GROUPS = ("all_obs", "slow", "obs_fast")
PHASES = ("Original", "Phase 1", "Phase 2")


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def fmt_num(x, missing="--", ndigits=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return missing
    return f"{x:.{ndigits}f}"


def metrics_case(data, case):
    if data is None:
        return None
    return (data.get("metrics") or {}).get(case)


def phase_dirs(model_id: str, defs: dict, exp_root: Path) -> dict:
    return {
        "Original": exp_root / defs.get("original_dir", model_id),
        "Phase 1": exp_root / defs["phase1_dir"],
        "Phase 2": exp_root / defs["phase2_dir"],
    }


def write_model_section(md: list, model_id: str, defs: dict, exp_root: Path) -> None:
    md.append(f"## {model_id}")
    md.append("")
    md.append(defs["desc"])
    md.append("")

    dirs = phase_dirs(model_id, defs, exp_root)
    data = {phase: load_json(d / "neural_eval.json") for phase, d in dirs.items()}

    for case in CASES:
        md.append(f"### {case.upper()}")
        md.append("")
        md.append("| Phase | RMSE (all) | RMSE (slow) | RMSE (fast) | EV (all) | ES (all) | S1/S0 |")
        md.append("|---|---|---|---|---|---|---|")
        for phase in PHASES:
            m = metrics_case(data[phase], case)
            deg = (data[phase] or {}).get("metrics", {}).get("degradation") if data[phase] else None
            if m is None:
                md.append(f"| {phase} | -- | -- | -- | -- | -- | {fmt_num(deg)} |")
                continue
            md.append(
                f"| {phase} | {fmt_num(m['rmse'])} | {fmt_num(m['groups']['slow'])} | "
                f"{fmt_num(m['groups']['obs_fast'])} | {fmt_num(m['ev']['groups']['all_obs'])} | "
                f"{fmt_num(m['es']['groups']['all_obs'])} | {fmt_num(deg)} |"
            )
        md.append("")


def write_report(exp_root: Path, output_path: Path) -> None:
    md = []
    md.append("# L96 Per-Channel Normalization Ablation")
    md.append("")
    md.append("**Question:** no model or data-pipeline code applies normalization to L96 "
              "states/observations anywhere -- raw physical units flow through the whole "
              "pipeline unchanged. The shared `UNet1D` backbone's `LayerNorm` normalizes "
              "jointly across channels *per timestep*, not per-channel, so it doesn't "
              "compensate. This ablation tests whether adding real per-channel z-score "
              "normalization changes performance, for the two simplest/most standard "
              "neural baselines: **DirectUNet** and **VanillaCFM** -- plus a third leg, "
              "**MonaiDirectUNet**, that swaps DirectUNet's `UNet1D` backbone for MONAI's "
              "`DiffusionModelUNet` (same data/hyperparameters otherwise) to test whether "
              "DirectUNet's collapse under normalization is fixable with a better-conditioned "
              "backbone rather than by retuning DirectUNet itself.")
    md.append("")
    md.append("**Normalization scheme:** per-channel z-score over the 24D observed subspace "
              "(8 slow + 16 fast), `x_norm[d] = (x_raw[d] - mean[d]) / std[d]`. Stats computed "
              "once from a freshly-generated L96 train split (1000 windows) and shared across "
              "every model/phase (`experiments/l96_norm_stats_obsj2.pt`, see "
              "`precompute_l96_norm_stats.py`). Both `states` and `obs` use the same stats.")
    md.append("")
    md.append("**Phases:**")
    md.append("- **Original** -- existing benchmark checkpoint, no normalization (baseline).")
    md.append("- **Phase 1** -- same frozen checkpoint, inference-only normalize-input/"
              "denormalize-output. Sanity check: expected to look *worse* than Original "
              "(the frozen weights were tuned for raw-scale obs, so normalized-scale obs is "
              "an out-of-distribution input). If Phase 1 looks unchanged from Original, that's "
              "a sign the wiring isn't actually being applied, not a genuine finding.")
    md.append("- **Phase 2** -- retrained from scratch with `data.normalize: true`. The actual "
              "research question: does normalization help, hurt, or wash out once the model is "
              "trained on normalized-scale data end to end?")
    md.append("")
    md.append("**FDV1 deferred:** FDV1 (`models/fourdvarnet.py`, `eval_fdv1_l96.py`) is not yet "
              "merged to `master` (still on `feature/l96-fdv1-sda-hybrid`), so this ablation "
              "covers DirectUNet + VanillaCFM only; FDV1's leg is a follow-up once that branch merges.")
    md.append("")
    md.append("---")
    md.append("")

    for model_id, defs in MODEL_DEFS.items():
        write_model_section(md, model_id, defs, exp_root)
        md.append("---")
        md.append("")

    md.append("## Findings")
    md.append("")
    md.append("Phase 1 confirms the wiring works both ways: normalizing a frozen checkpoint's "
              "input badly degrades it (RMSE ~5x worse, EV strongly negative for both models), "
              "as expected for feeding an out-of-distribution input scale into weights tuned "
              "for the other scale.")
    md.append("")
    md.append("Phase 2 (retrained from scratch, identical hyperparameters otherwise) gives "
              "**opposite outcomes for the two models**:")
    md.append("")
    md.append("- **L1b (DirectUNet): normalization breaks training.** `train_loss` flat-lined "
              "at ~1.000 (the no-skill MSE for unit-variance normalized targets) from epoch ~9 "
              "through epoch 199 -- the model collapsed to predicting the per-channel mean and "
              "never recovered. The resulting checkpoint scores RMSE 1.73/1.72 (S0/S1) vs the "
              "Original's 0.62/0.63 -- a ~2.8x degradation -- and EV collapses from 0.86 to "
              "~0.03 (essentially no skill). This was checked against a mid-training false-alarm "
              "risk (verified: the no-op regression tests pass, S0/S1 are handled by identical "
              "code paths, and NaN-masked/unobserved timesteps pass through normalize() as NaN "
              "unchanged, so the masking mechanism is unaffected) -- it is a real, reproducible "
              "training failure with these exact hyperparameters (`lr=0.001`, "
              "`gradient_clip_val=10.0`), not a bug in the normalize/denormalize wiring.")
    md.append("- **L3 (VanillaCFM): normalization helps slightly.** Training converged normally "
              "(loss 0.71 -> 0.046 over 400 epochs). The retrained checkpoint scores RMSE "
              "0.552/0.552 (S0/S1) vs the Original's 0.564/0.566 -- a small but consistent "
              "improvement, with EV up from 0.879/0.877 to 0.884/0.883.")
    md.append("")
    md.append("**Takeaway:** normalization is not a free win here -- it is architecture-"
              "sensitive. VanillaCFM (which already conditions on a time/noise-scale signal and "
              "has to learn a well-behaved velocity field regardless of input scale) tolerates "
              "and mildly benefits from normalization. DirectUNet (a single-pass regression with "
              "no such structure) failed to train at all under the exact hyperparameters tuned "
              "for the raw-scale version -- normalization changed its loss landscape enough to "
              "need re-tuning (e.g. learning rate) that was deliberately out of scope for this "
              "controlled ablation. Whether DirectUNet can be recovered with a lower learning "
              "rate is an open follow-up, not answered by this ablation.")
    md.append("")
    md.append("- **MonaiDirectUNet (L1b + MONAI's DiffusionModelUNet backbone): normalization "
              "not only doesn't collapse it, it outperforms L1b's own unnormalized baseline.** "
              "Same data setup and hyperparameters as L1b's Phase 2 (lr=0.001, "
              "gradient_clip_val=10.0, 200 epochs) -- the only change is swapping "
              "`models.unet.UNet1D` for MONAI's `DiffusionModelUNet` (proper timestep-embedding "
              "MLP + FiLM-style scale-shift conditioning + real ResBlocks, vs UNet1D's "
              "additive-only conditioning; see the architecture gap analysis this model was "
              "built to test, `/homes/rfablet/.claude/plans/monai-diffunet-prototype.md`). "
              "(Note: an earlier run of this experiment had a bug where `dropout` was silently "
              "a no-op -- see CHANGELOG 2026-09-07 -- the numbers below are from the corrected "
              "retrain with real dropout active, which improved the result further.) "
              "Training loss descended smoothly the whole run (0.73 -> 0.029 train), and unlike "
              "the buggy no-dropout run, val loss kept falling rather than plateauing early "
              "(0.115 at epoch 40 -> 0.105 at epoch 80 -> 0.096 at epoch 199) -- real dropout "
              "delaying overfitting as expected, with no flatlining. Final RMSE 0.501/0.502 "
              "(S0/S1) -- *better* than L1b's own unnormalized Original (0.62/0.63) by a clear "
              "margin and ~3.4x better than L1b's collapsed normalized Phase 2 (1.73/1.72). EV "
              "0.902/0.901, now clearly ahead of VanillaCFM's normalized result (RMSE "
              "0.552/0.552, EV 0.884/0.883) despite being architecturally a single-pass "
              "deterministic regressor like DirectUNet, not a flow-matching model with an "
              "inherent noise-scale signal. This is consistent with the hypothesis that "
              "DirectUNet's collapse under normalization was a "
              "backbone-conditioning limitation, not something fundamental to single-pass "
              "regression: a backbone with real scale-aware conditioning tolerates the "
              "normalized input distribution just as well as (here, clearly better than) a "
              "model that conditions on an explicit noise/time scale. Caveat: MonaiDirectUNet "
              "has 3.11x more parameters than DirectUNet at matching `hidden_channels` "
              "(5,889,048 vs 1,896,600 at [64,128,256]), so this is not a controlled "
              "like-for-like capacity comparison -- it shows the normalization failure is "
              "avoidable with a richer backbone, not that UNet1D's conditioning scheme "
              "specifically is the sole cause.")
    md.append("")
    md.append("---")
    md.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md) + "\n")
    logger.info(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build the L96 normalization ablation report")
    parser.add_argument("--exp-root", default=str(ROOT / "experiments"))
    parser.add_argument("--output", default=str(ROOT / "reports/l96/outputs/l96_normalization_ablation.md"))
    args = parser.parse_args()
    write_report(Path(args.exp_root), Path(args.output))


if __name__ == "__main__":
    main()
