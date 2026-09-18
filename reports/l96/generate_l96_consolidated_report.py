#!/usr/bin/env python3
"""Consolidated L96 benchmark report: full metric tables + Hovmöller reconstructions.

Consumes only cached artifacts of the DA-parity benchmark (Obs30, dws=500,
200 shared test windows):

- DA metric cache  ``experiments/l96_baselines_dws500_s0c_*_obsj2_int100_fw.json``
- DA trajectories  ``experiments/l96_baselines_trajectories_dws500_s0c_*_int100_fw.npz``
- Test dataset     ``experiments/l96_datasets_obsj2_int100_nwin200.pt``
- Neural estimates ``experiments/L*/estimates_{s0,s1}.npz``

Outputs ``reports/l96/outputs/l96_consolidated_benchmark.md`` with RMSE/EV/ES tables
over the all/slow/fast variable groups, a consistency-check section (cached
metrics recomputed from stored arrays), and Hovmöller reconstruction figures
(state + |error| maps, slow/fast blocks) for the worst/median/best test windows
ranked by the best DA scheme (Strong-4DVar per-window RMSE).
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.estimate_metrics import (
    evaluate_ensemble_estimates,
    evaluate_estimates,
    per_window_ensemble_crps,
    per_window_rmse_ev,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import archive  # noqa: E402

DA_JSON_CANDIDATES = [
    "experiments/l96_baselines_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.json",
    "experiments/l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100_fw.json",
    "experiments/l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int100.json",
]
DA_TRAJ_CANDIDATES = [
    "experiments/l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw.npz",
    "experiments/l96_baselines_trajectories_dws500_inf2.0_etkf_inf2.0_obsj2_int100_fw.npz",
]
DATASET_CANDIDATES = [
    "experiments/l96_datasets_obsj2_int100_nwin200.pt",
]
NEURAL_EXP_DIRS = [
    "L1b_monai_unet_s0s1_norm_splus_cosine",
    "L1b_monai_unet_s0s1_norm",
    "L1b_monai_unet_s0s1_norm_cosine",
    "L1b_monai_unet_s0s1_norm_l_cosine",
    "L2b_monai_vanilla_cfm_s0s1_norm",
    "L2b_monai_vanilla_cfm_s0s1_norm_cosine",
    "L2b_monai_vanilla_cfm_s0s1_norm_splus_cosine",
    "SDA1_monai_prior_l96_norm",
    "SDA2_monai_cond_mixed_l96_norm",
    "SDA3_monai_cond_noisy_l96_norm",
    "SDA1_monai_prior_l96_norm/directunet_hybrid",
    "SDA2_monai_cond_mixed_l96_norm/directunet_hybrid",
    "SDA3_monai_cond_noisy_l96_norm/directunet_hybrid",
    "FDV1_unrolled_monai_unet_l96",
    "FDV2_grad_state_monai_l96_fixedw",
    "FDV1_SDA3_monai_hybrid",
    "FDV1_SDA1_monai_hybrid",
    "FDV1_SDA2_monai_hybrid",
    "L1b_monai_unet_s0s1_norm_obsdensity",
    "L2b_monai_vanilla_cfm_s0s1_norm_obsdensity",
    "l96_obs_density_directunet_aug_sda3_hybrid",
    "FDV1_obsstate_monai_l96_initvar01_Stier_auxpriorcost01",
    "FDV2_subgrad_state_monai_l96_Stier",
    "FDV1_Stier_SDA3_monai_hybrid",
    "subgrad_Stier_SDA3_monai_hybrid",
]

# The monai-backbone subset of NEURAL_EXP_DIRS -- used to scope the
# per-trajectory (mean +/- std across windows) detail table to this session's
# new schemes (plus the DA baselines) rather than re-deriving it for every
# pre-existing row too.
MONAI_ROWS = [
    "L1b_monai_unet_s0s1_norm_splus_cosine",
    "L1b_monai_unet_s0s1_norm",
    "L1b_monai_unet_s0s1_norm_cosine",
    "L1b_monai_unet_s0s1_norm_l_cosine",
    "L2b_monai_vanilla_cfm_s0s1_norm",
    "L2b_monai_vanilla_cfm_s0s1_norm_cosine",
    "L2b_monai_vanilla_cfm_s0s1_norm_splus_cosine",
    "SDA1_monai_prior_l96_norm",
    "SDA2_monai_cond_mixed_l96_norm",
    "SDA3_monai_cond_noisy_l96_norm",
    "SDA1_monai_prior_l96_norm/directunet_hybrid",
    "SDA2_monai_cond_mixed_l96_norm/directunet_hybrid",
    "SDA3_monai_cond_noisy_l96_norm/directunet_hybrid",
    "FDV1_unrolled_monai_unet_l96",
    "FDV2_grad_state_monai_l96_fixedw",
    "FDV1_SDA3_monai_hybrid",
    "FDV1_SDA1_monai_hybrid",
    "FDV1_SDA2_monai_hybrid",
    "L1b_monai_unet_s0s1_norm_obsdensity",
    "L2b_monai_vanilla_cfm_s0s1_norm_obsdensity",
    "l96_obs_density_directunet_aug_sda3_hybrid",
    "FDV1_obsstate_monai_l96_initvar01_Stier_auxpriorcost01",
    "FDV2_subgrad_state_monai_l96_Stier",
    "FDV1_Stier_SDA3_monai_hybrid",
    "subgrad_Stier_SDA3_monai_hybrid",
]

# ES / CRPS convention: reported only where a proper ensemble score is defined
# AND available. A deterministic point estimator (DETERMINISTIC_METHODS) gets an
# em-dash; an ensemble-capable scheme with no members_<case>.npz yet gets
# "pending". The former N=1 MAE-proxy fallback was removed on 2026-09-18: it put
# two different formulas in the same column, so the only two rows that had real
# members appeared better partly by convention rather than by skill.

# Methods whose RMSE/EV/ES come from a per-case ens30 subdirectory rather than
# the run dir itself. Every entry was a pre-monai method (L3/V2/V3/SDA1/SDA2
# and the FDV1 hybrids), all dropped 2026-09-18, so the map is now empty --
# kept because collect_estimates still supports the indirection and a future
# ens30-subdir run would use it.
ENS30_DIRS: dict[str, dict[str, str]] = {
}
# Best-of-each-family monai-backbone comparison (2026-09-08/09 session): best
# DA scheme (Strong-4DVar) vs best DirectUNet (L-tier, cosine-annealed -- the
# cosine schedule fixed an instability that made flat-LR L-tier much worse
# than M-tier) vs best VanillaCFM (M-tier, flat LR -- cosine was a wash here)
# vs best plain SDA (SDA3, noisy-params conditioning) vs best DirectUNet+SDA
# hybrid (SDA3 warm-started from DirectUNet-M, tau0=0.3/guidance_weight=2.0 --
# all three SDA variants converged on the same hyperparameters in the sweep,
# SDA3 edged out SDA1/SDA2 on RMSE).
DEFAULT_FIGURE_METHODS = [
    "Strong-4DVar",
    "L1b_monai_unet_s0s1_norm_l_cosine",
    "L2b_monai_vanilla_cfm_s0s1_norm",
    "SDA3_monai_cond_noisy_l96_norm",
    "SDA3_monai_cond_noisy_l96_norm/directunet_hybrid",
    "FDV1_unrolled_monai_unet_l96",
    "FDV1_SDA3_monai_hybrid",
]
DA_METHODS = ["ETKF", "EnKF", "Strong-4DVar"]
PER_WINDOW_ROWS = DA_METHODS + MONAI_ROWS
RANKS = ["worst", "median", "best"]
CASES = ["s0", "s1"]
GROUPS = ("all_obs", "slow", "obs_fast")
NO = 8

SCHEME_DESCRIPTIONS: list[tuple[str, str, str]] = [
    ("Strong-4DVar", "Variational",
     ("Strong-constraint 4D-Var over the dws=500 window (`B_var=2.0`, `R_var=0.5`, `max_iter=10`, "
      "`lr=0.2`, autodiff minimization); assimilates the full window trajectory.")),
    ("EnKF", "Ensemble KF",
     ("Stochastic ensemble Kalman filter, `N_ens=30`, inflation=2.0, no localization; sequential "
      "observation updates.")),
    ("ETKF", "Ensemble KF",
     "Deterministic ensemble square-root filter, `N_ens=30`, inflation=2.0, no localization."),
    ("L1b_monai_unet_s0s1_norm_splus_cosine", "Neural (DirectUNet, monai backbone)",
     ("MonaiUNet1D backbone (FiLM-conditioned ResBlocks vs UNet1D's additive-only conditioning; "
      "`models/monai_unet_adapter.py`) swapped into L1b's single-pass regression, plus per-channel "
      "z-score normalization (`data.normalize=true`); S+ capacity tier (hidden [32,64,128], "
      "1.48M params), cosine-annealed LR; 200 epochs.")),
    ("L1b_monai_unet_s0s1_norm", "Neural (DirectUNet, monai backbone)",
     ("As above, M capacity tier (hidden [64,128,256], 5.89M params), flat LR; 200 epochs.")),
    ("L1b_monai_unet_s0s1_norm_cosine", "Neural (DirectUNet, monai backbone)",
     ("As above, M tier, cosine-annealed LR (near-identical to the flat-LR M result -- cosine's "
      "gain is tier-dependent, see the L tier below).")),
    ("L1b_monai_unet_s0s1_norm_l_cosine", "Neural (DirectUNet, monai backbone)",
     ("As above, L capacity tier (hidden [128,256,512], 23.5M params), cosine-annealed LR. Flat LR "
      "at this tier was badly unstable (RMSE 0.77-0.89 across two seeds, worse than S+/M); cosine "
      "annealing fixed it entirely, making L the best DirectUNet tier overall. Its DA-benchmark eval "
      "needs a reduced `--batch-size` (16, not the 200 default) -- the wider bottleneck OOMs a "
      "single-batch pass at 200 (26.8GiB attention-adjacent allocation on a 44GB GPU).")),
    ("L2b_monai_vanilla_cfm_s0s1_norm", "Neural (CFM, τ=0, monai backbone)",
     ("MonaiUNet1D backbone swapped into L2b's τ=0 CFM, plus per-channel normalization; M tier, "
      "flat LR; 400 epochs. Best single monai-backbone result of the non-hybrid schemes.")),
    ("L2b_monai_vanilla_cfm_s0s1_norm_cosine", "Neural (CFM, τ=0, monai backbone)",
     "As above, cosine-annealed LR (near-identical to flat -- a wash for this tier)."),
    ("L2b_monai_vanilla_cfm_s0s1_norm_splus_cosine", "Neural (CFM, τ=0, monai backbone)",
     "As above, S+ capacity tier (hidden [32,64,128]), cosine-annealed LR."),
    ("SDA1_monai_prior_l96_norm", "Neural (SDA prior + DPS guidance, monai backbone)",
     ("MonaiUNet1D backbone + per-channel normalization swapped into SDA1's unconditional prior; "
      "M tier; 400 epochs. Same pure-noise-start guidance convention as the non-monai SDA1 above "
      "(guidance_weight=40, N_outer=10, R_var=0.5; R_var's value is provably inert here -- the "
      "DPS step normalizes its own gradient by its norm, which exactly cancels any positive R_var "
      "scale factor -- verified empirically, R_var=0.5 vs 50.0 give trajectories differing only by "
      "float32 noise).")),
    ("SDA2_monai_cond_mixed_l96_norm", "Neural (SDA prior, params+forcing cond. + DPS guidance, monai backbone)",
     "As SDA1-monai but params+forcing-conditioned, mirroring SDA2-mixed above."),
    ("SDA3_monai_cond_noisy_l96_norm", "Neural (SDA prior, noisy-params cond. + DPS guidance, monai backbone)",
     ("As SDA2-monai but conditioned on a per-window noisy params estimate (a fresh random 0-1.5x "
      "fraction of the true-to-DA bias, `data.noisy_da_bias`/`noisy_da_max`) instead of the true "
      "params -- tests robustness to imperfect conditioning. Best plain (non-hybrid) SDA result.")),
    ("SDA1_monai_prior_l96_norm/directunet_hybrid", "Neural (DirectUNet-M mean + SDA1-monai warm-started guidance)",
     ("Same SDEdit-style warm start as the FDV1+SDA hybrids (`evaluation/sda_sampler.py`'s "
      "`mean_estimate`/`tau0`), but warm-starting from DirectUNet-M(monai)'s frozen point estimate "
      "instead of FDV1's -- i.e. SDA is used to sample the *anomaly* around DirectUNet's "
      "reconstruction rather than starting from pure noise. `tau0=0.3`, `guidance_weight=2.0` "
      "(S0-only grid sweep over `tau0∈{0,0.3,0.5,0.7,0.8}×guidance_weight∈{0,1,2,5,10,40,100}`, "
      "all three SDA variants converged on this same point).")),
    ("SDA2_monai_cond_mixed_l96_norm/directunet_hybrid", "Neural (DirectUNet-M mean + SDA2-monai warm-started guidance)",
     "As above, warm-starting SDA2-monai (params+forcing-conditioned) instead of SDA1-monai."),
    ("SDA3_monai_cond_noisy_l96_norm/directunet_hybrid", "Neural (DirectUNet-M mean + SDA3-monai warm-started guidance)",
     ("As above, warm-starting SDA3-monai (noisy-params conditioned) instead of SDA1-monai. Best "
      "RMSE/EV among the DirectUNet-warm-started hybrids -- but see FDV1+SDA3-monai below, which "
      "beats it comfortably by swapping in a better mean-estimate model.")),
    ("FDV1_unrolled_monai_unet_l96", "Neural (4DVarNet-style unrolled solver, monai backbone)",
     ("MonaiUNet1D backbone (`unet_backbone=monai`, `models/fourdvarnet.py::FourDVarNetSolver`) "
      "swapped into FDV1's `obs+state` unrolled solver (`x_{k+1} = x_k - (1/N_outer)*UNet(x_k, obs)`, "
      "N_outer=10, zero-initialized, deterministic, no ensemble); M tier; 400 epochs. Rebased its "
      "branch onto `origin/master` first to pick up gradient checkpointing (PR #172), which fixes "
      "an earlier CUDA-OOM this exact backbone+solver combination hit at production scale. "
      "**Note:** its first eval attempt gave a nonsensical RMSE 3.47/EV -3.75 -- traced to "
      "`eval_neural_l96.py`'s `--n-outer` CLI default (1), which is correct for tau0-only CFM but "
      "silently starves this solver's zero-init refinement of the N_outer=10 iterations it needs; "
      "fixed by passing `--n-outer 10` explicitly, recovering a sane, in fact excellent, result.")),
    ("FDV2_grad_state_monai_l96_fixedw", "Neural (4DVarNet-style unrolled solver, gradient-conditioned, monai backbone)",
     ("MonaiUNet1D backbone swapped into FDV2's `update_input='grad+state'` solver (each of N_outer=10 "
      "iterations builds a real variational cost `prior_cost(state) + obs_weight*obs_cost(state, obs)` "
      "and feeds the UNet its true `torch.autograd.grad(..., create_graph=True)`), with a **fixed** "
      "(non-trainable) `prior_weight=1.0` held for all 400 epochs at a constant `lr=0.001` -- the "
      "\"fixedw\" variant. **Diverged to NaN at epoch 190/400** (val_loss NaN first, train_loss still "
      "normal mid-epoch -- a sudden collapse, not gradual drift); root-caused to "
      "`models/fourdvarnet.py::_normalize_channels`, which only floors the `grad+state` cost-gradient "
      "channel's RMS norm (`clamp_min(1e-8)`) with no matching ceiling, unlike the state branch `x` "
      "(clamped every iteration via `clip_range=50.0`) -- as the model converges and the raw gradient's "
      "RMS shrinks toward that floor, dividing by it can blow the normalized channel up unbounded. The "
      "one config difference from the FDV2/UNet1D run that stayed stable (job 52305): there "
      "`prior_weight` was trainable and was actively shrinking (0.99->0.76 over 60 epochs), damping the "
      "gradient magnitude away from this regime; fixed at 1.0 here, nothing does. Job killed rather than "
      "left to burn its remaining epochs post-NaN. **No valid trained result -- unavailable below.** Fix "
      "for a retry: trainable `prior_weight` (as in the stable UNet1D recipe) and/or an upper clamp on "
      "the normalized gradient channel symmetric to the state branch's `clip_range=50.0`.")),
    ("FDV1_SDA3_monai_hybrid", "Neural (FDV1-monai mean + SDA3-monai warm-started guidance)",
     ("Same SDEdit-style warm start as the DirectUNet+SDA hybrids above, but swapping in FDV1-monai "
      "(the unrolled solver) as the mean-estimate model instead of DirectUNet-M -- SDA3-monai is used "
      "to sample the anomaly around FDV1's point estimate. `tau0=0.3`, `guidance_weight=2.0` (own "
      "S0-only sweep over `tau0∈{0.1..0.5}×guidance_weight∈{0.5,1,2,5}`, converged on the exact same "
      "point as the DirectUNet+SDA hybrids' independent sweep). One real wrinkle specific to this "
      "combination: FDV1-monai was trained WITHOUT `data.normalize=true` (unlike DirectUNet-M/SDA3, "
      "which share that normalized space) -- feeding it normalized obs (as SDA expects) silently "
      "produced a garbage mean estimate that poisoned the whole hybrid (confirmed: RMSE 1.36/EV 0.35) "
      "before this was caught; fixed by preparing a SEPARATE raw-obs dataloader for FDV1's own "
      "`.sample()` call and normalizing its output before handing it to SDA as the warm start. "
      "**Best scheme in this table overall.**")),
    ("FDV1_SDA1_monai_hybrid", "Neural (FDV1-monai mean + SDA1-monai warm-started guidance)",
     ("As FDV1+SDA3-monai but warm-starting SDA1-monai (unconditional prior) instead of SDA3-monai, "
      "for coherence with the DirectUNet+SDA1/2/3 family above. Same `tau0=0.3`/`guidance_weight=2.0` "
      "(not re-swept per SDA variant -- three independent sweeps this session all converged on this "
      "exact point regardless of SDA variant or mean model).")),
    ("FDV1_SDA2_monai_hybrid", "Neural (FDV1-monai mean + SDA2-monai warm-started guidance)",
     "As above, warm-starting SDA2-monai (true-params conditioned) instead of SDA1-monai."),
    ("L1b_monai_unet_s0s1_norm_obsdensity", "Neural (DirectUNet, monai backbone, obs-density-augmented)",
     ("As DirectUNet-M(monai,cos) but trained with `data.obs_density_augment=true` (`data/obs_density.py::sample_training_density_mask`): at every (window, obs-time), 40% chance of full fast-Y density, else a uniformly-random `keep_k∈{0,...,15}` of the 16 fast-Y channels kept, redrawn every batch. Closes the OOD gap found in the fast-Y observation-density generalization sweep (PR #180): DirectUNet/CFM read `obs` only via `nan_to_num`, no mask channel, so an unseen-at-training partial-channel dropout pattern reads as a spurious near-zero observation. An L-tier attempt at this same augmentation collapsed toward the fast-Y conditional mean even at full density (variance ratio ~35%); M-tier avoids it entirely (variance ratio ~99%). Evaluated here at full canonical density only (N=1, single pass) -- see `l96_obs_density_augmented_training.md` for the dedicated reduced-density sweep.")),
    ("L2b_monai_vanilla_cfm_s0s1_norm_obsdensity", "Neural (CFM, τ=0, monai backbone, obs-density-augmented)",
     ("As CFM-M(monai,flat) but with the identical `obs_density_augment` training augmentation described above -- unlike the DirectUNet-L attempt, CFM tolerated it cleanly at M-tier with no collapse (variance ratio ~96%). Evaluated here at full canonical density only (N=1, single pass); see `l96_obs_density_augmented_training.md` for the reduced-density sweep.")),
    ("l96_obs_density_directunet_aug_sda3_hybrid", "Neural (DirectUNet-M(monai,cos,obsdensity) mean + SDA3-monai warm-started guidance)",
     ("Same SDEdit-style warm start as DirectUNet+SDA3 above, but swapping in the `obs_density_augment`-trained DirectUNet-M mean estimate instead of the non-augmented one -- same `tau0=0.3`/`guidance_weight=2.0`. Third-best full-density RMSE/EV in this whole table (0.389 S0 -- behind FDV1-Stier+SDA3(monai)'s 0.359 and FDV1+SDA3-monai's 0.379, and comfortably ahead of non-augmented DirectUNet+SDA3's 0.420) **and** the best absolute worst-case RMSE at zero fast-Y density among everything tested in that sweep (1.065, edging out plain SDA3's 1.082 -- see `l96_obs_density_augmented_training.md`; neither FDV1+SDA3 nor FDV1-Stier+SDA3 were part of that reduced-density sweep, so they aren't ruled out as contenders there); its full-density ES/CRPS here are on the N=1 MAE-proxy convention, not the proper ensemble score its non-augmented sibling rows get (see the note above table).")),
    ("FDV1_obsstate_monai_l96_initvar01_Stier_auxpriorcost01", "Neural (4DVarNet-style unrolled solver, monai backbone, S-tier)",
     ("As FDV1(monai) above (`update_input=obs+state`, `models/fourdvarnet.py::FourDVarNetSolver`, N_outer=10, monai backbone) but at a smaller \"S\" capacity tier (`hidden_channels=[32,64,128]`, `monai_num_res_blocks=1`, 1,055,544 params -- vs the M tier's `hidden_channels=[64,128,256]`, `monai_num_res_blocks=2`, 5,889,048 params every other FDV1/FDV2 row in this table uses), a random (not zero) initial condition (`init_state_var=0.1`, i.e. `x_0 ~ N(0, 0.1)`), and a small auxiliary prior-consistency loss term during training (`aux_var_cost_weight=0.01`, `prior_cost(x_final)+prior_cost(states)`, no `prior_weight`/`obs_cost` mixed in -- a real bug in an earlier version of this loss term, since fixed, is documented in `CHANGELOG.d/2026-09-12-fdv-aux-prior-loss-drop-prior-weight-obs-cost.md`); 400 epochs, cosine-annealed LR. Genuinely better than the M-tier FDV1(monai) baseline above on every RMSE/EV metric despite being ~5.6x smaller in parameter count. Evaluated as a single pass (N=1), same convention as FDV1(monai).")),
    ("FDV2_subgrad_state_monai_l96_Stier", "Neural (4DVarNet-style unrolled solver, cheap proxy-gradient-conditioned, monai backbone, S-tier)",
     ("FDV2's `update_input=subgrad+state` solver (`models/fourdvarnet.py::FourDVarNetSolver`) -- a cheap two-residual proxy gradient (the observation residual `obs-x` plus the prior-autoencoder residual `x-Phi(x)`, fed to the UNet alongside the state, with **no** `torch.autograd.grad` call at all, unlike FDV2(monai)'s real `grad+state` cost gradient above) -- on the same S-tier MonaiUNet1D as `FDV1-Stier(monai)` (`hidden_channels=[32,64,128]`, `monai_num_res_blocks=1`, 1,055,544 main-solver params, vs the M tier's `hidden_channels=[64,128,256]`, `monai_num_res_blocks=2`, 5,889,048 params every earlier FDV2 attempt in this table's history used), plus the same `init_state_var=0.1` (random initial condition) and `aux_var_cost_weight=0.01` (small auxiliary prior-consistency loss, `prior_cost(x_final)+prior_cost(states)`, no `prior_weight`/`obs_cost` mixed in) recipe as `FDV1-Stier(monai)`; 400 epochs, cosine-annealed LR. **Best standalone (non-hybrid) neural mean-estimate model found in this entire investigation** -- better than `FDV1-Stier(monai)` itself, and dramatically better than every earlier `subgrad+state`/`grad+state` attempt at the M tier, which all showed a severe fast-Y reconstruction collapse (variance ratio ~0.5-0.6); this run shows no such collapse at all (fast-Y variance ratio ~1.01-1.02, essentially perfectly calibrated). Evaluated as a single pass (N=1), same convention as FDV1-Stier(monai).")),
    ("FDV1_Stier_SDA3_monai_hybrid", "Neural (FDV1-Stier-monai mean + SDA3-monai warm-started guidance)",
     ("Same SDEdit-style warm start as FDV1+SDA3(monai) above (`evaluation/sda_sampler.py`'s `mean_estimate`/`tau0`), but swapping in FDV1-Stier(monai) (above) as the mean-estimate model instead of the M-tier FDV1(monai) -- same `tau0=0.3`, `guidance_weight=2.0`, `r_var=0.5`, ens30 (`n_members=30`, `n_outer=10`); not re-swept, reusing the established convention. Same raw-obs handling as FDV1+SDA3(monai) above (FDV1-Stier was also trained without `data.normalize=true`; `--no-mean-normalized`, separate raw dataloader, `eval_sda_mean_hybrid_l96.py`'s existing mechanism, unchanged). Was the best scheme in this table overall (previous-best 0.3587/0.3583 S0/S1 pooled RMSE, beating FDV1+SDA3(monai)'s 0.3787/0.3740) -- see `subgrad+state-Stier+SDA3(monai)` below, which now beats it on every RMSE/EV/ES metric by swapping in an even better (and cheaper) mean-estimate model. One real bug hit and fixed while producing this row: running the hybrid eval from the `4dvarnet-fm-fdv-monai` worktree (needed because that's where the `SDA3_monai_cond_noisy_l96_norm` checkpoint, `l96_norm_stats_obsj2.pt`, and the cached test dataset live) silently built the WRONG UNet tier for the mean-estimate model -- that worktree's `train.py`/`models/fourdvarnet.py` predates this session's `monai_num_res_blocks` feature (a `FourDVarNetSolver` param needed alongside `hidden_channels` to actually get an S tier vs M/S+), so it silently defaulted to `monai_num_res_blocks=2`, building an S+-tier-shaped model (1,482,264 params) instead of true S-tier (1,055,544), partially failing to load the checkpoint (shape mismatches on 8 `unet.*` keys, silently skipped) and producing garbage output (RMSE ~1.8, negative EV) on the first attempt. Fixed by running the exact same command from `4dvarnet-fm-obs-density-gen` instead (which has the feature), with absolute paths for the SDA3 checkpoint/config/dataset/norm-stats pointing into `4dvarnet-fm-fdv-monai`'s `experiments/` dir -- `model_factory` then correctly built the true 1,055,544-param S-tier model matching the checkpoint exactly (verified: zero `unet.*` shape-mismatch warnings, and a direct `load_model(...)` unit check confirming `sum(p.numel() for p in model.unet.parameters()) == 1055544`).")),
    ("subgrad_Stier_SDA3_monai_hybrid", "Neural (subgrad+state-Stier-monai mean + SDA3-monai warm-started guidance)",
     ("Same SDEdit-style warm start as FDV1-Stier+SDA3(monai) above (`evaluation/sda_sampler.py`'s `mean_estimate`/`tau0`), but swapping in `subgrad+state-Stier(monai)` (above) as the mean-estimate model instead of `FDV1-Stier(monai)` -- same `tau0=0.3`, `guidance_weight=2.0`, `r_var=0.5`, ens30 (`n_members=30`, `n_outer=10`); not re-swept, reusing the established convention. Same raw-obs handling as `FDV1-Stier+SDA3(monai)` above (this checkpoint's config also has no `data.normalize` block; `--no-mean-normalized`, separate raw dataloader, `eval_sda_mean_hybrid_l96.py`'s existing mechanism, unchanged). **New best scheme in this entire table, on every RMSE/EV/ES metric** -- beats `FDV1-Stier+SDA3(monai)`'s previous-best 0.3587/0.3583 S0/S1 pooled RMSE with 0.3410/0.3402.")),
]


def make_obs_j_indices(no: int, j_truth: int, j_obs: int) -> np.ndarray:
    x_idx = list(range(no))
    y_idx = [no + k * j_truth + j for k in range(no) for j in range(j_obs)]
    return np.array(x_idx + y_idx)


def _first_existing(patterns: list[str]) -> Path:
    for p in patterns:
        path = ROOT / p
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidates exist: {patterns}")


def short_name(name: str) -> str:
    MONAI_SHORT_NAMES = {
        "L1b_monai_unet_s0s1_norm_splus_cosine": "DirectUNet-S+(monai,cos)",
        "L1b_monai_unet_s0s1_norm": "DirectUNet-M(monai,flat)",
        "L1b_monai_unet_s0s1_norm_cosine": "DirectUNet-M(monai,cos)",
        "L1b_monai_unet_s0s1_norm_l_cosine": "DirectUNet-L(monai,cos)",
        "L2b_monai_vanilla_cfm_s0s1_norm": "CFM-M(monai,flat)",
        "L2b_monai_vanilla_cfm_s0s1_norm_cosine": "CFM-M(monai,cos)",
        "L2b_monai_vanilla_cfm_s0s1_norm_splus_cosine": "CFM-S+(monai,cos)",
        "SDA1_monai_prior_l96_norm": "SDA1(monai)",
        "SDA2_monai_cond_mixed_l96_norm": "SDA2(monai)",
        "SDA3_monai_cond_noisy_l96_norm": "SDA3(monai)",
        "SDA1_monai_prior_l96_norm/directunet_hybrid": "DirectUNet+SDA1",
        "SDA2_monai_cond_mixed_l96_norm/directunet_hybrid": "DirectUNet+SDA2",
        "SDA3_monai_cond_noisy_l96_norm/directunet_hybrid": "DirectUNet+SDA3",
        "FDV1_unrolled_monai_unet_l96": "FDV1(monai)",
        "FDV2_grad_state_monai_l96_fixedw": "FDV2(monai)",
        "FDV1_SDA3_monai_hybrid": "FDV1+SDA3(monai)",
        "FDV1_SDA1_monai_hybrid": "FDV1+SDA1(monai)",
        "FDV1_SDA2_monai_hybrid": "FDV1+SDA2(monai)",
        "L1b_monai_unet_s0s1_norm_obsdensity": "DirectUNet-M(monai,cos,obsdensity)",
        "L2b_monai_vanilla_cfm_s0s1_norm_obsdensity": "CFM-M(monai,flat,obsdensity)",
        "l96_obs_density_directunet_aug_sda3_hybrid": "DirectUNet(aug)+SDA3",
        "FDV1_obsstate_monai_l96_initvar01_Stier_auxpriorcost01": "FDV1-Stier(monai)",
        "FDV2_subgrad_state_monai_l96_Stier": "subgrad+state-Stier(monai)",
        "FDV1_Stier_SDA3_monai_hybrid": "FDV1-Stier+SDA3(monai)",
        "subgrad_Stier_SDA3_monai_hybrid": "subgrad+state-Stier+SDA3(monai)",
    }
    if name in MONAI_SHORT_NAMES:
        return MONAI_SHORT_NAMES[name]
    return name.split("_")[0] if "_" in name else name


def load_truth(dataset_path: Path, obs_idx: np.ndarray) -> dict[str, np.ndarray]:
    ds = torch.load(dataset_path, map_location="cpu", weights_only=False)
    return {
        case: np.stack([w["true_state"][..., obs_idx].numpy() for w in ds[f"test_{case}"]])
        for case in CASES
    }


def load_da_trajectories(path: Path, case: str, method: str, obs_idx: np.ndarray) -> np.ndarray:
    data = np.load(path)
    traj = data[f"{case}_{method.replace('-', '_')}_trajectories"]
    if traj.shape[-1] > len(obs_idx):
        traj = traj[..., obs_idx]
    return traj.astype(np.float64)


# Point estimators: one deterministic forward pass, no stochastic sampler, so a
# 30-member "ensemble" would be 30 identical copies and a proper ensemble CRPS/ES
# is not defined for them. Their probabilistic cells are left empty rather than
# filled with the N=1 MAE proxy -- mixing the two formulas in one column made
# rows non-comparable, since a proper ensemble score credits spread and the
# proxy cannot.
DETERMINISTIC_METHODS: frozenset[str] = frozenset({
    "Strong-4DVar",
    "L1b_monai_unet_s0s1_norm",
    "L1b_monai_unet_s0s1_norm_cosine",
    "L1b_monai_unet_s0s1_norm_l_cosine",
    "L1b_monai_unet_s0s1_norm_splus_cosine",
    "L1b_monai_unet_s0s1_norm_obsdensity",
    "FDV1_unrolled_monai_unet_l96",
    "FDV2_grad_state_monai_l96_fixedw",
    "FDV1_obsstate_monai_l96_initvar01_Stier_auxpriorcost01",
    "FDV2_subgrad_state_monai_l96_Stier",
})


# Methods deliberately listed with no result: the row exists so its description
# is published, and every metric cell renders as an em-dash. FDV2(monai)
# diverged to NaN at epoch 190/400 (see its description); there is no valid
# trained result to score, so absent estimates are expected, not a broken
# artifact.
UNAVAILABLE_METHODS: frozenset[str] = frozenset({
    "FDV2_grad_state_monai_l96_fixedw",
})


# Runs whose estimate arrays are not named estimates_<case>.npz. The
# obs-density hybrid writes one file per retained fast-Y channel count; keep16
# is the full-density one this benchmark reports.
ESTIMATE_FILENAMES: dict[str, str] = {
    "l96_obs_density_directunet_aug_sda3_hybrid":
        "estimates_directunet_sda3_{case}_keep16.npz",
}


def load_neural_trajectories(name: str, case: str) -> np.ndarray | None:
    """Trajectories for run ``name``, case ``case``, or None if unavailable.

    Artifact location is delegated to ``evaluation.archive``, which knows about
    both the canonical ``experiments/l96/<run>/`` layout and the legacy
    ``experiments/<run>/`` one. Building the path here is what previously made
    this report unrunnable from the master worktree: the 2026-09-10
    consolidation left the estimate arrays in whichever worktree trained each
    run, so 24 of 34 rows resolved to a path that does not exist here.
    """
    npz_path = archive.resolve_estimates(
        name, case, filename=ESTIMATE_FILENAMES.get(name), required=False)
    if npz_path is None:
        return None
    return np.load(npz_path)["trajectories"].astype(np.float64)


def stored_truth_npz(name: str, case: str) -> Path | None:
    """The estimates_*.npz (ens30 subdir for ens30 methods) carrying the stored
    truth, so the truth-consistency check reads the same file the trajectories
    came from."""
    lookup = ENS30_DIRS[name][case] if name in ENS30_DIRS else name
    return archive.resolve_estimates(
        lookup, case, filename=ESTIMATE_FILENAMES.get(lookup), required=False)


def collect_estimates(
    da_traj_path: Path,
    obs_idx: np.ndarray,
    neural_dirs: list[str],
) -> dict[str, dict[str, np.ndarray | None]]:
    est: dict[str, dict[str, np.ndarray | None]] = {}
    for method in DA_METHODS:
        est[method] = {case: load_da_trajectories(da_traj_path, case, method, obs_idx) for case in CASES}
    for dirname in neural_dirs:
        if dirname in ENS30_DIRS:
            est[dirname] = {}
            for case in CASES:
                ens30_est = load_neural_trajectories(ENS30_DIRS[dirname][case], case)
                est[dirname][case] = (ens30_est if ens30_est is not None
                                      else load_neural_trajectories(dirname, case))
        else:
            est[dirname] = {case: load_neural_trajectories(dirname, case) for case in CASES}
    return est


def per_window_rmse(traj: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((traj - ref) ** 2, axis=(1, 2)))


def select_windows(traj: np.ndarray, ref: np.ndarray) -> dict[str, tuple[int, float]]:
    rw = per_window_rmse(traj, ref)
    order = np.argsort(rw)
    mid = len(order) // 2
    return {
        "best": (int(order[0]), float(rw[order[0]])),
        "median": (int(order[mid]), float(rw[order[mid]])),
        "worst": (int(order[-1]), float(rw[order[-1]])),
    }


def resolve_figure_methods(names: list[str], available: dict[str, dict[str, np.ndarray | None]]) -> list[str]:
    resolved = []
    for name in names:
        if name not in available:
            raise FileNotFoundError(f"Unknown method '{name}' (not a DA scheme or known experiment dir)")
        missing = [c for c in CASES if available[name][c] is None]
        if missing:
            raise FileNotFoundError(f"Missing estimates for '{name}' cases {missing}")
        resolved.append(name)
    return resolved


def _run_l96_convention_groups(traj: np.ndarray, ref: np.ndarray) -> dict[str, dict[str, float]]:
    """Replicate evaluation/run_l96.py metric conventions.

    RMSE = mean over windows of per-window RMSE; EV = pooled; ES = pooled MAE
    (only valid for deterministic schemes, i.e. Strong-4DVar).
    """
    err_sq = (traj - ref) ** 2
    rmse_dim = np.mean(np.sqrt(np.mean(err_sq, axis=1)), axis=0)
    ev_dim = 1.0 - np.mean(err_sq, axis=(0, 1)) / np.maximum(np.var(ref, axis=(0, 1)), 1e-12)
    es_dim = np.mean(np.abs(traj - ref), axis=(0, 1))

    def grouped(arr: np.ndarray) -> dict[str, float]:
        return {"slow": float(np.mean(arr[:NO])), "obs_fast": float(np.mean(arr[NO:])), "all_obs": float(np.mean(arr))}

    return {"rmse": grouped(rmse_dim), "ev": grouped(ev_dim), "es": grouped(es_dim)}


def check_da_consistency(
    da_json_path: Path,
    est: dict[str, dict[str, np.ndarray | None]],
    truth: dict[str, np.ndarray],
) -> tuple[float, int]:
    with open(da_json_path) as f:
        cached = json.load(f)
    max_diff = 0.0
    n_checked = 0
    for case in CASES:
        for method, metrics in cached.get(case, {}).items():
            conv = _run_l96_convention_groups(est[method][case], truth[case])
            pairs = [("rmse", metrics["groups"], conv["rmse"]), ("ev", metrics["ev"]["groups"], conv["ev"])]
            if method == "Strong-4DVar":
                pairs.append(("es", metrics["es"]["groups"], conv["es"]))
            for _, cache_groups, new_groups in pairs:
                for group in GROUPS:
                    max_diff = max(max_diff, abs(cache_groups[group] - new_groups[group]))
                    n_checked += 1
    return max_diff, n_checked


def check_neural_truth(
    est: dict[str, dict[str, np.ndarray | None]],
    truth: dict[str, np.ndarray],
) -> tuple[float, list[str]]:
    max_diff = 0.0
    problems: list[str] = []
    for name in NEURAL_EXP_DIRS:
        for case in CASES:
            traj = est.get(name, {}).get(case)
            if traj is None:
                continue
            expected = truth[case]
            if traj.shape != expected.shape:
                problems.append(f"{name}/{case}: estimates shape {traj.shape} != truth shape {expected.shape}")
                continue
            stored_truth = np.load(stored_truth_npz(name, case))["truth"].astype(np.float64)
            max_diff = max(max_diff, float(np.max(np.abs(stored_truth - expected))))
    return max_diff, problems


def collect_metric_values(
    est: dict[str, dict[str, np.ndarray | None]],
    truth: dict[str, np.ndarray],
    row_order: list[str],
    da_json_path: Path,
) -> dict[str, dict[tuple[str, str], dict[str, float | None]]]:
    """Compute RMSE/EV from trajectories for all methods; ES from JSON for DA
    ensembles + ens30 methods (L3, V3), from trajectories (MAE) for the rest."""
    values: dict[str, dict[tuple[str, str], dict[str, float | None]]] = {"rmse": {}, "ev": {}, "es": {}}
    da_cache = json.load(open(da_json_path))
    pending_cells: set[tuple[str, str]] = set()

    def _ens30_es(row: str, case: str) -> dict[str, float] | None:
        """Proper (N=30, textbook) ensemble ES for an ens30 method from its
        ens30 JSON. Handles the S0 study's dual-convention schema
        (``ensemble.es_textbook``) and the single-convention schema
        (``ensemble.es``). Returns None when the JSON / block is unavailable.
        """
        ens30_json = archive.resolve_artifact(
            ENS30_DIRS[row][case], "neural_eval.json", required=False)
        if ens30_json is None:
            return None
        blk = json.load(open(ens30_json)).get("metrics", {}).get(case, {}).get("ensemble", {})
        for key in ("es_textbook", "es"):
            e = blk.get(key, {}).get("groups")
            if e:
                return e
        return None

    for row in row_order:
        for case in CASES:
            traj = est[row][case]
            if traj is None:
                none_groups = {g: None for g in GROUPS}
                values["rmse"][(row, case)] = dict(none_groups)
                values["ev"][(row, case)] = dict(none_groups)
                values["es"][(row, case)] = dict(none_groups)
                continue
            m = evaluate_estimates(traj, truth[case])
            values["rmse"][(row, case)] = m["groups"]
            values["ev"][(row, case)] = m["ev"]["groups"]
            if row in DETERMINISTIC_METHODS:
                # Checked before DA_METHODS on purpose: Strong-4DVar is a
                # deterministic variational analysis, and the ES the DA driver
                # cached for it is the N=1 proxy (_ESAccumulator is fed a
                # single-member "ensemble", analysis[t].reshape(1, -1)). Leaving
                # it in would re-introduce exactly the mixed-formula column this
                # convention removes. ETKF/EnKF are genuine N=30 ensembles and
                # keep their cached score.
                values["es"][(row, case)] = {g: None for g in GROUPS}
            elif row in DA_METHODS:
                da_blk = da_cache.get(case, {}).get(row, {})
                es_blk = da_blk.get("es")
                if es_blk and "groups" in es_blk:
                    values["es"][(row, case)] = es_blk["groups"]
                else:
                    values["es"][(row, case)] = {g: None for g in GROUPS}
            elif row in ENS30_DIRS:
                ens_es = _ens30_es(row, case)
                values["es"][(row, case)] = (
                    ens_es if ens_es else {g: None for g in GROUPS})
                if not ens_es:
                    pending_cells.add((row, case))
            elif row in DETERMINISTIC_METHODS:
                # A point estimator has no sampler, so a proper ensemble score
                # is not defined for it. Left empty rather than back-filled with
                # the N=1 MAE proxy: mixing the two formulas in one column made
                # rows non-comparable, because a proper ensemble score credits
                # spread and the proxy cannot.
                values["es"][(row, case)] = {g: None for g in GROUPS}
            else:
                # Ensemble-capable. Scored properly when members_<case>.npz is
                # present; otherwise marked pending -- the run needs
                # re-evaluating with --n-members 30.
                members = load_members(row, case)
                if members is not None:
                    ens = evaluate_ensemble_estimates(members, truth[case])
                    values["es"][(row, case)] = ens["ensemble"]["es"]["groups"]
                else:
                    values["es"][(row, case)] = {g: None for g in GROUPS}
                    pending_cells.add((row, case))
    return values, pending_cells


def load_members(dirname: str, case: str) -> np.ndarray | None:
    """A row's members_{case}.npz (ensemble spread), or None if it has none.

    Resolved through ``evaluation.archive`` like every other artifact. Building
    this path directly is not harmless: when it misses, the caller silently
    falls back to the deterministic CRPS proxy instead of the proper ensemble
    score, so an ens30 row quietly reports a different (worse) number rather
    than failing.
    """
    npz_path = archive.resolve_artifact(dirname, f"members_{case}.npz", required=False)
    if npz_path is None:
        return None
    # Left in float32, unlike the (much smaller) trajectory arrays which are
    # promoted for precision. A members array is (W, T, D, M) -- 1.6 GB at
    # 200x3000x24x30 -- and the ES accuracy term allocates a temporary of the
    # same shape, so promoting to float64 costs ~6.5 GB per call and was enough
    # to get this script OOM-killed. float32 is also what produced the
    # published ensemble-ES numbers.
    return np.load(npz_path)["members"]


def collect_per_window_values(
    row_order: list[str],
    est: dict[str, dict[str, np.ndarray | None]],
    truth: dict[str, np.ndarray],
) -> tuple[dict, set[tuple[str, str]]]:
    """Per-window (mean +/- std across the ~200 test windows) RMSE/EV/CRPS,
    scoped to whichever rows the caller passes (this report uses it for
    ``DA_METHODS + MONAI_ROWS``, not every historical row -- see MONAI_ROWS'
    docstring). CRPS is reported only where a proper ensemble score is defined and available:
    empty for a deterministic point estimator, empty (and returned in
    ``pending_cells``) for an ensemble-capable scheme whose
    ``members_{case}.npz`` has not been produced yet. The N=1 MAE-equivalent
    fallback was removed -- it put two different formulas in one column, so the
    two rows that did have members looked better partly by convention."""
    values: dict = {"rmse": {}, "ev": {}, "crps": {}}
    pending_cells: set[tuple[str, str]] = set()
    for row in row_order:
        for case in CASES:
            traj = est[row][case]
            if traj is None:
                values["rmse"][(row, case)] = None
                values["ev"][(row, case)] = None
                values["crps"][(row, case)] = None
                continue
            pw = per_window_rmse_ev(traj, truth[case])
            values["rmse"][(row, case)] = pw["rmse"]
            values["ev"][(row, case)] = pw["ev"]
            if row in DETERMINISTIC_METHODS:
                # Point estimator: no ensemble CRPS is defined. Left empty.
                values["crps"][(row, case)] = None
                continue
            members = load_members(row, case)
            if members is not None:
                values["crps"][(row, case)] = per_window_ensemble_crps(members, truth[case])
            else:
                # Ensemble-capable but no members stored yet -- pending a
                # re-evaluation with --n-members 30.
                values["crps"][(row, case)] = None
                pending_cells.add((row, case))
    return values, pending_cells


def fmt_per_window_table(
    title: str,
    block: dict[tuple[str, str], dict[str, dict[str, float]] | None],
    row_order: list[str],
    is_crps: bool = False,
    pending_cells: set[tuple[str, str]] | None = None,
) -> str:
    header = "| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [f"### {title}", "", header, sep]
    for row in row_order:
        cells = []
        for case in CASES:
            for group in GROUPS:
                v = block[(row, case)]
                if v is not None:
                    cell = f"{v[group]['mean']:.3f}±{v[group]['std']:.3f}"
                elif pending_cells and (row, case) in pending_cells:
                    cell = " pending "
                else:
                    cell = "  —  "
                cells.append(cell)
        lines.append(f"| {short_name(row)} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def fmt_block_table(
    title: str,
    block: dict[tuple[str, str], dict[str, float | None]],
    row_order: list[str],
    higher_better: bool,
    include_degradation: bool,
    pending_cells: set[tuple[str, str]] | None = None,
    is_es: bool = False,
) -> str:
    agg = max if higher_better else min
    best = {
        f"{case}_{group}": agg(block[(r, case)][group] for r in row_order if block[(r, case)][group] is not None)
        for case in CASES
        for group in GROUPS
    }
    header = "| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |"
    sep = "|---|---|---|---|---|---|---|"
    if include_degradation:
        header += " S1/S0 |"
        sep += "---|"
    lines = [f"### {title}", "", header, sep]
    for row in row_order:
        cells = []
        for case in CASES:
            for group in GROUPS:
                v = block[(row, case)][group]
                if v is None:
                    cell = "  —  "
                else:
                    cell = f"{v:.4f}"
                    if abs(v - best[f"{case}_{group}"]) < 5e-5:
                        cell = f"**{cell}**"
                if is_es and v is None and pending_cells and (row, case) in pending_cells:
                    # Distinguish "not defined for this scheme" (plain em-dash)
                    # from "defined but not computed yet".
                    cell = " pending "
                cells.append(cell)
        line = f"| {short_name(row)} | " + " | ".join(cells) + " |"
        if include_degradation:
            s0 = block[(row, "s0")]["all_obs"]
            s1 = block[(row, "s1")]["all_obs"]
            if s0 is not None and s1 is not None and s0 > 0:
                line += f" {s1 / s0:.3f} |"
            else:
                line += " n/a |"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def fmt_scheme_table() -> str:
    lines = [
        "| ID | Type | Description |",
        "|---|---|---|",
    ]
    for scheme_id, family, description in SCHEME_DESCRIPTIONS:
        lines.append(f"| {short_name(scheme_id)} | {family} | {description} |")
    lines.append("")
    return "\n".join(lines)


def plot_hovmoller(
    fig_path: Path,
    case: str,
    rank: str,
    win_idx: int,
    sel_rmse: float,
    method_names: list[str],
    est_win: dict[str, np.ndarray],
    truth_win: np.ndarray,
    obs_win: np.ndarray,
    obs_times: np.ndarray,
    dt: float,
) -> None:
    # Obs row: the noisy, subsampled observed values actually fed to every
    # model here (already restricted to the 24D observed subspace, same as
    # truth_win, and already NaN-filled at unobserved timesteps by the
    # cached dataset -- rendered as blank/background by pcolormesh, not a
    # fabricated interpolation). Its "error" column is the actual
    # observation noise (|obs - truth| at observed times only), not a
    # reconstruction error -- a genuinely different, useful quantity: how
    # noisy the raw input is, vs. how good each method's reconstruction is.
    obs_masked = obs_win
    labels = ["Truth", "Obs"] + [short_name(m) for m in method_names]
    n_rows = len(labels)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 1.35 * n_rows + 1.0), constrained_layout=True)
    t = np.arange(truth_win.shape[0]) * dt

    state_data: list[list[np.ndarray]] = [
        [truth_win[:, :NO], truth_win[:, NO:]],
        [obs_masked[:, :NO], obs_masked[:, NO:]],
    ]
    for m in method_names:
        state_data.append([est_win[m][:, :NO], est_win[m][:, NO:]])
    err_data = [[np.abs(d - truth_block) for d, truth_block in zip(row, [truth_win[:, :NO], truth_win[:, NO:]])] for row in state_data]

    flat_state = [d for row in state_data for d in row]
    s_vmin = min(np.nanmin(d) for d in flat_state)
    s_vmax = max(np.nanmax(d) for d in flat_state)
    e_vmax = float(np.nanpercentile(np.concatenate([d.ravel() for row in err_data for d in row]), 99.5))
    cmap_state = plt.get_cmap("viridis")
    cmap_err = plt.get_cmap("inferno")

    im_state = None
    im_err = None
    for r, label in enumerate(labels):
        if label == "Obs":
            # "RMSE" isn't meaningful for the raw observation row (it's data,
            # not a reconstruction) -- report the actual observation noise
            # (RMS of |obs - truth| at observed times only) instead.
            row_rmse = float(np.sqrt(np.nanmean(np.concatenate(err_data[r], axis=1) ** 2)))
            row_label = f"{label}\nobs noise {row_rmse:.3f}"
        else:
            win_rmse = float(np.sqrt(np.mean(np.concatenate(err_data[r], axis=1) ** 2)))
            row_label = label if r == 0 else f"{label}\nRMSE {win_rmse:.3f}"
        for c in range(4):
            ax = axes[r, c]
            data, cmap, vmin, vmax = (
                (state_data[r][c % 2], cmap_state, s_vmin, s_vmax)
                if c < 2
                else (np.minimum(err_data[r][c % 2], e_vmax), cmap_err, 0.0, e_vmax)
            )
            mesh = ax.pcolormesh(t, np.arange(data.shape[1]), data.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
            if c < 2:
                im_state = mesh
                if r == 0:
                    for ot in obs_times:
                        ax.axvline(ot * dt, color="w", lw=0.5, ls=":", alpha=0.85)
            else:
                im_err = mesh
            if r == n_rows - 1:
                ax.set_xlabel("time (tu)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=7)
            if r > 0:
                ax.set_yticks([])
            ax.tick_params(labelsize=6)

    col_titles = ["state: slow X", "state: fast Y", "|error|: slow X", "|error|: fast Y"]
    for c, ttl in enumerate(col_titles):
        axes[0, c].set_title(ttl, fontsize=9)
    ylabels_fast = [f"Y{j + 1}^{k}" for k in range(1, NO + 1) for j in range(2)]
    for c in (1, 3):
        axes[0, c].set_yticks(np.arange(len(ylabels_fast))[::2])
        axes[0, c].set_yticklabels(ylabels_fast[::2], fontsize=5)

    cb_state = fig.colorbar(im_state, ax=list(axes[:, :2].ravel()), shrink=0.9, pad=0.01)
    cb_state.set_label("state", fontsize=8)
    cb_err = fig.colorbar(im_err, ax=list(axes[:, 2:].ravel()), shrink=0.9, pad=0.01)
    cb_err.set_label(f"|error| (vmax={e_vmax:.2f}, q99.5)", fontsize=8)
    fig.suptitle(
        f"L96 {case.upper()} — {rank} window #{win_idx} (Strong-4DVar window RMSE {sel_rmse:.3f}); dotted lines = obs times",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_FIGURE_METHODS,
                        help="Methods for the reconstruction figures (DA schemes and/or experiment dir names)")
    parser.add_argument("--ranks", nargs="+", default=RANKS, choices=RANKS)
    parser.add_argument("--out-dir", default="reports/l96/outputs")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    da_json_path = _first_existing(DA_JSON_CANDIDATES)
    da_traj_path = _first_existing(DA_TRAJ_CANDIDATES)
    dataset_path = _first_existing(DATASET_CANDIDATES)
    logger.info("DA json: %s | DA trajs: %s | dataset: %s", da_json_path, da_traj_path, dataset_path)

    obs_idx = make_obs_j_indices(NO, 4, 2)
    truth = load_truth(dataset_path, obs_idx)
    est = collect_estimates(da_traj_path, obs_idx, NEURAL_EXP_DIRS)
    figure_methods = resolve_figure_methods(args.methods, est)
    table_rows = DA_METHODS + NEURAL_EXP_DIRS

    values, pending_cells = collect_metric_values(est, truth, table_rows, da_json_path)
    pw_values, pw_pending_cells = collect_per_window_values(PER_WINDOW_ROWS, est, truth)
    with open(da_json_path) as f:
        cfg = json.load(f)["config"]

    da_max_diff, n_checked = check_da_consistency(da_json_path, est, truth)
    truth_max_diff, problems = check_neural_truth(est, truth)
    da_ok = da_max_diff <= args.tolerance
    truth_ok = truth_max_diff <= args.tolerance and not problems

    md: list[str] = [
        "# L96 Consolidated Benchmark — DA baselines vs neural models",
        "",
        (
            "Setup: two-scale L96, Obs30 (`obs_interval=100`, `obs_j=2` → 24D observed space), "
            f"dws={cfg.get('da_window_steps', 500)}, 200 shared cached test windows; "
            "S1 = ±20% params + ±10% bias (DA forward model uses biased `*_da`)."
        ),
        "",
        (
            "RMSE/EV are recomputed from the stored trajectory arrays via "
            "`evaluation/estimate_metrics.py`; ES for DA ensemble methods (EnKF/ETKF) "
            "and L3 (ens30×10) are proper ensemble scores (N=30, MAE − 0.5·pairwise spread) "
            "read from cached run outputs; ES for deterministic methods is the N=1 per-dim "
            "MAE proxy. **bold** marks the best value per column."
        ),
        "",
        "## Benchmarked schemes",
        "",
        fmt_scheme_table(),
        (
            "Shared setup: all L-series neural models are trained and evaluated on the identical DA-parity "
            "benchmark (all-5 params ±20% randomized per window; S1 adds a ±10% bias; models operate in the "
            "24D observed subspace with obs-only inputs unless noted). DA baselines receive the same per-window "
            "parameters as the truth generation (S0) or their biased `*_da` counterparts (S1), which is what "
            "makes the DA-vs-neural comparison apples-to-apples."
        ),
        "",
        "## RMSE (pooled, lower is better)",
        "",
        fmt_block_table("RMSE by variable group", values["rmse"], table_rows, False, True),
        (
            "Note on conventions: the DA metric cache stores the **mean of per-window RMSEs** "
            "(evaluation/run_l96.py), while this table uses the **pooled** convention "
            "(`sqrt(mean sq err)` over all windows/timesteps) for every method — the same convention as the "
            "neural evaluation. Pooled RMSE is ≤ mean-of-window RMSE, so DA values here are slightly lower "
            "(more favorable) than in the legacy cache; both orderings agree."
        ),
        "",
        "## Explained Variance (higher is better)",
        "",
        fmt_block_table("EV by variable group", values["ev"], table_rows, True, False),
        "## Energy Score (lower is better)",
        "",
        fmt_block_table("ES by variable group", values["es"], table_rows, False, False,
                        pending_cells=pending_cells, is_es=True),
        (
            "`*` = ES from a one-member ensemble (N=1, deterministic; ES = per-dim MAE). "
            "Unmarked = proper ensemble ES (N=30, MAE − 0.5·pairwise spread). "
            "EnKF/ETKF ES are read from the bug-fixed DA cache; L3 ES from the ens30×10 run; "
            "Strong-4DVar and other neural models are deterministic (N=1)."
        ),
        "",
        "## Per-trajectory detail: mean +/- std across the 200 test windows",
        "",
        (
            "Every table above pools all windows/timesteps into one number per method. This section "
            "instead computes RMSE/EV/CRPS **per window** (pooled over that window's own timesteps "
            "only) and reports the mean +/- std of that per-window distribution across the ~200 test "
            "windows -- i.e. how much reconstruction quality varies window-to-window, not just its "
            "average. Scoped to the DA baselines plus this session's monai-backbone schemes (not every "
            "historical row, to keep this bounded); the pooled tables above already cover everything. "
            "Note the RMSE means here are systematically a bit lower than the pooled RMSE above -- "
            "mean(sqrt(x)) <= sqrt(mean(x)) (Jensen's inequality), not a discrepancy."
        ),
        "",
        fmt_per_window_table("RMSE per window (mean +/- std, lower is better)",
                             pw_values["rmse"], PER_WINDOW_ROWS),
        fmt_per_window_table("EV per window (mean +/- std, higher is better)",
                             pw_values["ev"], PER_WINDOW_ROWS),
        fmt_per_window_table("CRPS per window (mean +/- std, lower is better)",
                             pw_values["crps"], PER_WINDOW_ROWS, pending_cells=pw_pending_cells, is_crps=True),
        (
            "`*` = CRPS from a one-member reconstruction (N=1, deterministic; CRPS = per-dim MAE, the "
            "N=1 special case of the ensemble formula). Unmarked = proper ensemble CRPS (per-dimension "
            "Energy Score, N=30, MAE − 0.5·pairwise member distance) from the stored `members_*.npz`."
        ),
        "",
        "## Consistency checks",
        "",
        f"- DA cached metrics vs recomputed-from-npz ({n_checked} values): max |Δ| = {da_max_diff:.2e} → "
        + ("PASS" if da_ok else f"FAIL (tolerance {args.tolerance})"),
        f"- Neural stored truth vs dataset true_state[:, obs_var_indices]: max |Δ| = {truth_max_diff:.2e} → "
        + ("PASS" if truth_ok else f"FAIL (tolerance {args.tolerance})"),
    ]
    for problem in problems:
        md.append(f"- WARNING: {problem}")

    md += [
        "",
        "## Reconstruction examples (Hovmöller)",
        "",
        (
            "Windows ranked by per-window pooled 24D RMSE of Strong-4DVar (best DA scheme); "
            "each figure shows rows = Truth/methods and columns = state / |error| maps for the slow X (8D) and "
            "fast Y (16D) blocks. State colors share one scale per figure; error maps share one scale across all "
            "rows/methods (99.5th-percentile cap, noted on the colorbar). Dotted vertical lines on the truth row "
            "mark observation times."
        ),
        "",
    ]
    header = "| Case | Rank | Window | 4DVar win-RMSE | " + " | ".join(short_name(n) for n in figure_methods) + " |"
    md += [header, "|---|---|---|---|" + "---|" * len(figure_methods)]

    for case in CASES:
        sel = select_windows(est["Strong-4DVar"][case], truth[case])
        for rank in args.ranks:
            win_idx, sel_rmse = sel[rank]
            w = torch.load(dataset_path, map_location="cpu", weights_only=False)[f"test_{case}"][win_idx]
            obs_times = np.where(w["obs_mask"].numpy())[0]
            obs_win = w["obs"].numpy().astype(np.float64)
            est_win = {name: est[name][case][win_idx] for name in figure_methods}
            truth_win = truth[case][win_idx]
            fig_path = figs_dir / f"l96_hovm_{case}_{rank}.png"
            plot_hovmoller(fig_path, case, rank, win_idx, sel_rmse, figure_methods, est_win, truth_win, obs_win, obs_times, float(cfg.get("dt", 0.001)))
            logger.info("Figure saved: %s", fig_path)
            cells = [f"{per_window_rmse(est[n][case][win_idx:win_idx + 1], truth_win[None])[0]:.3f}" for n in figure_methods]
            md.append(f"| {case.upper()} | {rank} | {win_idx} | {sel_rmse:.3f} | " + " | ".join(cells) + " |")

    md += [""]
    for case in CASES:
        for rank in args.ranks:
            md += [f"![{case}-{rank}](figs/l96_hovm_{case}_{rank}.png)", ""]

    report_path = out_dir / "l96_consolidated_benchmark.md"
    report_path.write_text("\n".join(md))
    logger.info("Report saved: %s", report_path)

    if not da_ok or not truth_ok or problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
