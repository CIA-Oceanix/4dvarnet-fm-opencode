# L63 Monai Hybrid Benchmark (DirectUNet-M / CFM-L / PredictStateCFM-L)

State estimation results for the three Monai-backbone L63 mean models —
**DirectUNet-M (flat)**, **CFM-L (cos)**, **PredictStateCFM-L (flat)** — on
their own, and combined with two different DA/guidance schemes: **SDA**
(score-based data assimilation) and **FlowPrior** (ICTM).

## Background: what SDA1/2/3 and FlowPrior are

**SDA (score-based data assimilation)** — Rozet & Louppe, *"Score-based Data
Assimilation"* (NeurIPS 2023). A flow-matching **prior** `p(x1)` is trained
*without ever seeing the observations* (only forcing/params, or nothing at
all); at inference time, its Euler sampling trajectory is nudged toward the
observations at every step via a DPS-style normalized-gradient guidance term
on the observation cost (`evaluation/sda_sampler.py::sda_guided_sample`).
Three variants, differing only in what the prior conditions on during
training:

- **SDA1** — fully unconditional prior (`monai_sda_prior.yaml`): never reads
  obs, forcing, or params.
- **SDA2** — conditioned on the window's true forcing + physical params
  (`monai_sda_prior_cond.yaml`).
- **SDA3** — conditioned on a *noisy* per-window estimate of forcing/params
  instead of the true values (`monai_sda_prior_cond_noisy.yaml`,
  `data.noisy_da_bias`), testing robustness to imperfect conditioning info.

Each SDA prior can also be run as a **hybrid**: instead of starting the
guided Euler trajectory from pure noise, it's warm-started (`tau0=0.3`, an
SDEdit-style mix of noise and a mean model's own point estimate) from one of
the three mean models' frozen checkpoints. This is what the "Mean model ×
SDA{1,2,3}" cells below report.

**FlowPrior (ICTM)** — Zhang et al., *"Flow Priors for Linear Inverse
Problems via Iterative Corrupted Trajectory Matching"* (NeurIPS 2024,
`reports/FlowPrior.pdf`). Unlike SDA, this is applied *directly on top of an
already-observation-conditioned flow model's own Euler trajectory*
(CFM-L(cos) or PredictStateCFM-L(flat) — both read obs as a network input
during training already). At each outer Euler step, `K` inner Adam steps
refine the current state against a local MAP objective combining the
observation cost, the flow's own closed-form prior score, and a
log-density (Hutchinson-trace) correction term
(`evaluation/ictm_sampler.py::ictm_map_sample`). It's a deterministic
point-estimate refinement, not an ensemble method. DirectUNet has no
tau-parameterized Euler trajectory for ICTM to refine, so it has no
FlowPrior row — and combining DirectUNet's estimate with an *already*
obs-conditioned flow's own refinement wouldn't add a complementary signal
the way warm-starting an *unconditional* SDA prior does, so it isn't a
meaningful combination to build either.

## 1. Mean models on their own

No guidance/refinement — each model's single-shot (DirectUNet) or
one-shot-ensemble (CFM-family, N_ensemble=50) estimate straight from its own
checkpoint, plus the classical DA baselines (N_ensemble=50 for EnKF/ETKF,
N=1 for the 4DVar variants) for reference. Full S+/M/L capacity tier ×
flat/cosine LR-schedule sweep for the neural models; `hidden_channels` are
`[32,64,128]` (S+), `[64,128,256]` (M), `[128,256,512]` (L). The three rows
marked **★** are the mean models used in the hybrid matrix below (picked as
each family's best- or near-best-performing variant).

| Model | Tier | Schedule | s0 RMSE | s0 R² | s1 RMSE | s1 R² |
|---|---|---|---:|---:|---:|---:|
| Weak-4DVar | — | — | 0.874 | 0.972 | 2.171 | 0.910 |
| Strong-4DVar | — | — | 0.928 | 0.962 | 2.586 | 0.863 |
| EnKF | — | — | 1.216 | 0.977 | 2.741 | 0.882 |
| ETKF | — | — | 1.217 | 0.976 | 2.762 | 0.880 |
| DirectUNet | S+ | flat | 0.715 | 0.993 | 0.677 | 0.993 |
| DirectUNet | S+ | cos | 0.694 | 0.993 | 0.665 | 0.994 |
| **DirectUNet ★** | **M** | **flat** | **0.644** | **0.994** | **0.600** | **0.995** |
| DirectUNet | M | cos | 0.649 | 0.994 | 0.619 | 0.995 |
| DirectUNet | L | flat | 0.857 | 0.989 | 0.836 | 0.990 |
| DirectUNet | L | cos | 0.673 | 0.993 | 0.678 | 0.993 |
| CFM | S+ | flat | 0.702 | 0.993 | 0.671 | 0.994 |
| CFM | S+ | cos | 0.699 | 0.993 | 0.699 | 0.993 |
| CFM | M | flat | 0.696 | 0.993 | 0.684 | 0.993 |
| CFM | M | cos | 0.658 | 0.994 | 0.759 | 0.992 |
| CFM | L | flat | 0.700 | 0.993 | 0.622 | 0.995 |
| **CFM ★** | **L** | **cos** | **0.611** | **0.995** | **0.621** | **0.994** |
| PredictStateCFM | S+ | flat | 0.668 | 0.994 | 0.696 | 0.993 |
| PredictStateCFM | S+ | cos | 0.683 | 0.993 | 0.660 | 0.994 |
| PredictStateCFM | M | flat | 0.650 | 0.994 | 0.624 | 0.994 |
| PredictStateCFM | M | cos | 0.663 | 0.993 | 0.612 | 0.995 |
| **PredictStateCFM ★** | **L** | **flat** | **0.658** | **0.993** | **0.602** | **0.995** |
| PredictStateCFM | L | cos | 0.672 | 0.993 | 0.601 | 0.995 |

## 2. Mean model × {SDA1, SDA2, SDA3, FlowPrior}

Best classical baseline (Weak-4DVar) and each mean model's own standalone
(no guidance/refinement) result included for reference.

| Mean model | Prior | s0 RMSE | s0 R² | s1 RMSE | s1 R² |
|---|---|---:|---:|---:|---:|
| — | Weak-4DVar (best baseline) | 0.874 | 0.972 | 2.171 | 0.910 |
| DirectUNet-M (flat) | none | 0.644 | 0.994 | 0.600 | 0.995 |
| CFM-L (cos) | none | 0.611 | 0.995 | 0.621 | 0.994 |
| PredictStateCFM-L (flat) | none | 0.658 | 0.993 | 0.602 | 0.995 |
| DirectUNet-M (flat) | SDA1 | 0.816 | 0.991 | 0.783 | 0.992 |
| DirectUNet-M (flat) | SDA2 | 0.797 | 0.991 | 0.747 | 0.992 |
| DirectUNet-M (flat) | SDA3 | 0.791 | 0.991 | 0.767 | 0.992 |
| CFM-L (cos) | SDA1 | 0.791 | 0.991 | 0.777 | 0.992 |
| CFM-L (cos) | SDA2 | 0.792 | 0.991 | 0.761 | 0.992 |
| CFM-L (cos) | SDA3 | 0.769 | 0.992 | 0.767 | 0.992 |
| PredictStateCFM-L (flat) | SDA1 | 0.813 | 0.991 | 0.770 | 0.992 |
| PredictStateCFM-L (flat) | SDA2 | 0.784 | 0.991 | 0.744 | 0.992 |
| PredictStateCFM-L (flat) | SDA3 | 0.786 | 0.991 | 0.758 | 0.992 |
| CFM-L (cos) | FlowPrior | 0.937 | 0.987 | 0.769 | 0.991 |
| PredictStateCFM-L (flat) | FlowPrior | 0.746 | 0.991 | 0.638 | 0.994 |
| DirectUNet-M (flat) | FlowPrior | — | — | — | — |

DirectUNet-M(flat) × FlowPrior intentionally not run (see background above).

**Notes:**
- All SDA cells use `guidance_weight=20`, `N_outer=10`, `N_ensemble=50`,
  `tau0=0.3` (`guidance_weight=20` confirmed near-optimal for all three
  priors by a sweep over `{0,1,3,10,20,40,80,150}`).
- Fixed during this run: the hybrid warm-start previously fed the mean
  model's *normalized*-space output straight into the (raw-space-trained)
  SDA prior without denormalizing it first, and the guidance-cost
  comparison had a matching normalize/raw mismatch — both corrected in
  `eval_sda_l63.py`.
- Also fixed: `PredictStateCFM` predicts the direct state `μ = E[x1|x_tau,y]`,
  not a velocity, but `ictm_map_sample` treated its output as a velocity
  field throughout — corrected by converting `μ → v = (μ - x)/(1-τ)`
  (`evaluation/ictm_sampler.py`), matching `PredictStateCFM.sample()`'s own
  convention. This is what brought PredictStateCFM-L(flat)+FlowPrior from an
  outlier (RMSE≈5.85) into line with the rest of the table.
