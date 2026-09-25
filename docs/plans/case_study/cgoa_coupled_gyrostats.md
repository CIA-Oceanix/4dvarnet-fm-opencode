# Coupled ocean–atmosphere case study from coupled gyrostats (CGOA)

**Status:** SCOPING v2 (2026-09-24). Nothing implemented. Proposes a fourth
case study, after L63, L96 and QG. It is built **in this repo**, fenced by a
`cgoa_*` prefix; §7 gives the reasons. **Which paper it serves (P1, P2 or its
own) is open** — see §9, decision 1.

> **What changed in v2.** Only §5 (implementation), §8 (work packages) and
> §9 decision 5. The science (§1–§4) is unchanged. v1 mirrored the L96 stack
> file for file (its own dataset classes, DA driver and sbatch per run). v2 builds
> CGOA on a few shared pieces instead, extracted in one no-op PR (§5.2), so
> the next case study does not start by copying L96 either. v2 also corrects
> v1's claim that `Weak4DVar` "applies unchanged": on two-scale dynamics it
> is the robust `L96Weak4DVar` design that applies, and even that needs a
> per-component whitening scale.

---

## 1. The question

L96 and QG test our methods on one fluid with one dominant time scale. Coupled
ocean–atmosphere DA adds a difficulty neither has: **two components whose time
scales differ by one to three orders of magnitude, observed very unequally**.
The atmosphere is observed densely and often; the ocean sparsely, rarely, or
only through its surface.

For our methods this raises three questions the existing benchmarks cannot
ask:

- **Q1 — windows across time scales.** A DA window that is long enough for the
  atmosphere barely moves the ocean. Can an amortized solver (4DVarNet, CFM,
  SDA) constrain the slow component from a window in which it is almost
  constant? Or does the slow part of the posterior stay at the prior?
- **Q2 — strongly vs weakly coupled DA.** Classical coupled DA distinguishes
  strongly coupled DA (SCDA: cross-component covariances used in the update),
  weakly coupled DA (WCDA: coupled forecast, separate updates) and uncoupled
  DA. A learned solver is strongly coupled by construction. Does it recover
  the SCDA gain when only one component is observed? That is the regime where
  SCDA beats WCDA classically (Garcia-Oliva et al. 2025).
- **Q3 — structural model error in the coupling.** The air–sea coupling is
  where real coupled models are least trusted. Dropping or parameterizing
  a coupling term gives a **clean, controllable structural model error**. It
  sits on the P1 model-error axis, and is more physical than L96's
  `param_bias`.

## 2. Existing resources

### 2.1 Gyrostat low-order models (theory; no public code)

- **Obukhov; Gluhovsky & Tong (1999), *Phys. Fluids* 11, 334.** Galerkin
  low-order models that keep the energy conservation of the parent PDE can be
  written as **coupled Volterra gyrostats** (three-mode rigid-body systems)
  plus forcing and friction.
- **Gluhovsky (2006/2007), *J. Nonlinear Sci.***
  ([link](https://link.springer.com/article/10.1007/s00332-007-9006-6)). The
  equivalence runs both ways, with an algorithm to convert an energy-conserving
  low-order model into coupled generalized gyrostats. This is what links Tier A
  to Tier B below.
- **Gluhovsky (2017), *Complexity*, 6176045**
  ([link](https://onlinelibrary.wiley.com/doi/10.1155/2017/6176045)). The
  Vallis El Niño (ENSO) coupled ocean–atmosphere model admits a gyrostatic
  form. **This is the seed for the ocean component.**
- **Seshadri & Lakshmivarahan (2023), *Physica D* 133948**
  ([arXiv 2307.12337](https://arxiv.org/abs/2307.12337)). The minimal chaotic
  models built from gyrostats. Whether the gyrostat *core* conserves energy
  decides whether the forced–dissipative system can be chaotic. This is a
  design constraint for §3.
- **Seshadri & Lakshmivarahan (arXiv 2503.10782, v2 2026-05)**
  ([link](https://arxiv.org/html/2503.10782)). Invariants of coupled gyrostat
  hierarchies:
  - *sparse* nested chains: K gyrostats, M = 2K+1 modes, each gyrostat shares
    one mode with its predecessor;
  - *dense* nested chains: M = K+2, two modes shared.
  - Without linear feedback, a sparse chain has exactly (M+1)/2 quadratic
    invariants, recovered as Casimirs of a non-canonical Hamiltonian structure.

  Their code is only a MATLAB symbolic supplement.

### 2.2 Coupled ocean–atmosphere low-order codes

| resource | what it gives us | fit |
|---|---|---|
| **[Climdyn/qgs](https://github.com/Climdyn/qgs)** (MIT; numba, sympy) | MAOOAM: two-layer quasi-geostrophic atmosphere coupled mechanically and thermally to a shallow-water ocean. `f, Df = create_tendencies(params)` gives tendencies and Jacobian. Symbolic export to Python, Julia and Fortran. Lyapunov and covariant-Lyapunov-vector estimators. The tendencies are a sparse quadratic tensor. | **Backbone for Tier B.** |
| [Climdyn/MAOOAM](https://github.com/Climdyn/MAOOAM) (Lua, Fortran, Python) | The original MAOOAM ([GMD 2016](https://arxiv.org/pdf/1603.06755)). DAPPER ships its Python version. | Reference for parity only. |
| [nansencenter/DAPPER](https://github.com/nansencenter/DAPPER) | DA test bed that includes MAOOAM. | Cross-check for our EnKF numbers. |
| VDDG / OA-QG-WS v2 ([GMD 2014](https://ui.adsabs.harvard.edu/abs/2014GMD.....7..649V/abstract)) | 24-variable predecessor of MAOOAM. | Not needed if qgs is used. |
| Peña & Kalnay (2004) | Three coupled Lorenz-63 systems (extratropical atmosphere, tropical atmosphere, ocean), 9 variables. The coupling is linear and does not conserve energy. | The design Tier A improves on. |

### 2.3 Coupled-DA benchmarks to compare against

- **Tondeur, Carrassi, Vannitsem & Bocquet (2020)**
  ([arXiv 2003.03333](https://arxiv.org/abs/2003.03333)). EnKF on MAOOAM,
  6 configurations varying coupling strength and mode count. Cross-component
  effects are strong from slow to fast. The fast component **must be observed
  frequently**, or its error contaminates the slow one.
- **Garcia-Oliva, Carrassi & Counillon (2025), *NPG* 32, 439**
  ([link](https://npg.copernicus.org/articles/32/439/2025/)). A two-component
  Lorenz-63 with separate spatial (`S`) and temporal (`τ`) scale parameters,
  comparing SCDA, WCDA and uncoupled DA. SCDA improves over WCDA **when only
  one component is observed**, most of all in the unobserved component. Code
  is available on request only.

## 3. Tier A — the coupled gyrostat model

### 3.1 Building block

Gyrostat `k`, modes `(y₁, y₂, y₃)`:

```
ẏ₁ = p y₂y₃ + b y₃ − c y₂
ẏ₂ = q y₃y₁ + c y₁ − a y₃
ẏ₃ = r y₁y₂ + a y₂ − b y₁
```

Forcing `F` and linear damping `−κ ⊙ y` are added on top.

With inertia weights `I` and energy `E = ½ Σᵢ Iᵢ yᵢ²`, the core conserves `E`
if and only if

- `I₁p + I₂q + I₃r = 0` for the triad (the unit-inertia case is
  `p + q + r = 0`), and
- the linear terms are skew in the `I`-weighted inner product.

**Lorenz-63 is one gyrostat.** Shift `Z = z − (ρ + σ)`. Then:

- the core is `(p, q, r) = (0, −1, 1)`, `(a, b, c) = (0, 0, −σ)`;
- the damping is `κ = (σ, 1, β)`;
- the forcing is `F = (0, 0, −β(ρ + σ))`.

This is the first unit test: the gyrostat builder must reproduce the L63
right-hand side exactly.

### 3.2 Components

- **Atmosphere (fast):** `K_a` Lorenz-like gyrostats with unit inertia and time
  scale 1. Two topologies:
  - a sparse nested chain (the Seshadri–Lakshmivarahan hierarchy);
  - a **ring** in which each gyrostat shares one mode with each neighbour. The
    ring has an L96-like topology, so the state has a spatial neighbourhood
    (see §6, risk R3).
- **Ocean (slow):** `K_o` gyrostats seeded by the Vallis/ENSO gyrostat. The
  time-scale separation is carried by **inertia**, `I_o = 1/τ` with
  `τ ∈ [0.01, 0.1]`, rather than by multiplying the ocean tendencies by `τ`.
  This keeps cross-component energy exchange conservative. In rigid-body terms
  the ocean is the heavy rotor.

### 3.3 Air–sea coupling — three switchable mechanisms

1. **Conservative cross triads.** Gyrostats whose three modes span both
   components (for example one atmosphere mode and two ocean modes), with
   `I_a p + I_o q + I_o r = 0`. They move energy between components without
   creating or destroying it.
2. **Shared interface modes.** One mode belongs to both chains: an SST-like
   variable, the nested-hierarchy construction applied across components. It
   is the natural target for "ocean observed only at the surface".
3. **Dissipative drag and heat flux.**

   ```
   ẏ_a += −γ (y_a − y_o)
   ẏ_o += +γ (I_a / I_o) (y_a − y_o)
   ```

   This conserves the "momentum" `I_a y_a + I_o y_o` and dissipates
   `γ I_a (y_a − y_o)²`. It mimics MAOOAM's wind stress and heat exchange.

**Energy budget.** `dE/dt = forcing − damping − drag`. Mechanisms 1 and 2 are
exactly neutral. This is the test for §5.

### 3.4 Proposed sizes (to be calibrated; see WP2)

| config | atmosphere | ocean | cross | state dim | role |
|---|---|---|---|---|---|
| **A0** | 1 gyrostat (L63) | 1 gyrostat (ENSO) | 1 triad + drag | 6 | Energy-consistent Peña–Kalnay analogue. Also the unit-test system. |
| **A1** | sparse chain `K_a=4` (9 modes) | chain `K_o=2` (5 modes) | 2 triads + 1 shared mode | ~14 | Main exploration config. |
| **A2** | ring `K_a=16` (32 modes) | ring `K_o=4` (8 modes) | 4 triads + drag | ~40 | Matches L96's `state_dim=40` so the backbones and budgets carry over. |

### 3.5 What Tier A buys over MAOOAM

- **Separate controls** for dimension (`2K+1`), scale separation (`τ`) and
  coupling (`γ`, triad strength).
- **Bounded by construction**, since the core conserves energy.
- **Invariants known analytically** (Casimirs). This gives an exact test for
  hard versus soft physical constraints.
- **Model-error knobs that are structural, not parametric** (§4.2).

Tier A has no community reference configuration, so the calibration in WP2 is
mandatory before any benchmark is frozen.

## 4. Tier B — MAOOAM through qgs

- Build the MAOOAM tensor once with qgs. Target the 36-variable configuration
  used by Tondeur et al.: atmosphere 2×10 modes (ψ, θ), ocean 2×8 (ψ, T).
- Save the sparse tensor as a small `.npz` committed to the repo. qgs stays an
  **optional, build-time-only** dependency, so the `fdv-monai-proto` env and
  CI are unchanged.
- Run it on the same torch engine as Tier A (§5).
- A parity test compares our torch tendencies and autograd Jacobian with
  qgs's `f` and `Df` on random states. It is skipped when qgs is absent.
- Time unit `1/f₀ ≈ 2.7 h`. Low-frequency variability is decadal, so spinup
  runs to about 10⁵–10⁷ time units. Cache the truth trajectories the way QG
  caches its per-window truth.
- MAOOAM's energy-conserving part is itself a coupled-gyrostat system
  (Gluhovsky 2007). Tier A is therefore a controlled caricature of Tier B, not
  an unrelated toy.

### 4.1 Observation scenarios (both tiers)

| scenario | atmosphere | ocean | tests |
|---|---|---|---|
| **O-full** | every `Δt_a`, all modes | every `n·Δt_a`, all modes | reference |
| **O-atm** | every `Δt_a` | none | SCDA vs WCDA; Q2 |
| **O-sst** | every `Δt_a` | interface / SST modes only, every `n·Δt_a` | the realistic case |
| **O-sparse-atm** | every `m·Δt_a` | as O-sst | Tondeur's "fast must be observed often" |

For MAOOAM, observations can be taken on a physical grid (qgs reconstructs
fields from the modes). That makes `H` non-trivial and gives the state a
spatial layout. See R3.

### 4.2 Model-error scenarios (S0 / S1 analogue)

- **S0:** perfect model.
- **S1-param:** biased `τ`, `γ` or forcing.
- **S1-struct-drop:** the DA model omits the conservative cross triads.
- **S1-struct-param:** the DA model replaces the cross triads with drag, i.e.
  a bulk parameterization.

The standing caveat in `docs/README.md` applies unchanged: these
scenarios price model error **only for methods that run a forward model at
inference**. A model-free neural row's S1/S0 ratio is definitional.

## 5. Implementation in this repo

*Revised in v2. v1 mirrored the L96 layout file for file; see "What changed in
v2" under the status line.*

**Principle: CGOA is the first case study built on shared pieces, not a fifth
copy of the per-case stack.** Each existing case study has its own dataset
module and its own DA driver: `evaluation/run.py` (L63), `evaluation/run_l96.py`
(L96) and `evaluation/run_qg_baselines.py` (QG, 1337 lines). The three drivers
repeat the same function names. L96 constants also sit in modules that look
generic:
- `evaluation/estimate_metrics.py:12` hard-codes `NO = 8`;
- `evaluation/run_l96.py`'s `fmt_*` helpers default to `NO=8`;
- `train.py`'s `_per_group_rmse` hard-codes `NO=8, J=4`.

Mirroring `data/lorenz96.py` and `run_l96.py` would add a fourth copy of each.
Instead, CGOA adds only what is specific to it. The few generic pieces it needs
are extracted first (§5.2), in their own PR, so that L96 and QG can adopt them
later. **No existing L63/L96/QG file moves** — that stays ruled out by
`docs/plans/tech/multi_refactor_plan.md` ("What this plan is not").

**Acceptance criterion:** CGOA adds no top-level `eval_*_cgoa.py`, no
per-experiment sbatch, and no copy of a function that already exists in
`run_l96.py` or `run_qg_baselines.py`. If it seems to need one, a shared piece is
missing: extract it rather than copy it.

### 5.1 Case-specific files

**One engine for both tiers.** Both are quadratic ODEs,
`dx/dt = Σ T_ijk x_j x_k` with `x₀ = 1` carrying the linear and constant terms.

| file | content |
|---|---|
| `models/quadratic_ode.py` | `QuadraticTensorDynamics(DynamicsBase)`: sparse `T`, batched `einsum`, RK4, GPU. **Generic:** L63, single-scale L96 and MAOOAM can all be written in it, and nothing in it is CGOA-specific. `step(state, forcing, **params)` must accept and ignore unknown keywords, because the generic `ETKF`/`EnKF`/`Weak4DVar`/`Strong4DVar` inject L63 defaults (`sigma, rho, beta, c1`) into every call (`evaluation/baselines.py:427`). |
| `models/cgoa_dynamics.py` | Gyrostat builder: blocks, topologies (chain / ring / dense), inertia, and the three coupling mechanisms → `T`. Energy and Casimir diagnostics. Also exposes **`component_slices`** (`{"atm": slice, "ocean": slice}`, plus `"interface"` for O-sst). Scoring, localization and whitening in §5.2 all read this one piece of metadata. |
| `models/maooam_dynamics.py` | Loads the cached qgs tensor into `QuadraticTensorDynamics`, with the same `component_slices`. |
| `scripts/export_maooam_tensor.py` | One-off qgs → `.npz` export (needs qgs, run offline). |
| `data/cgoa.py` | `CGOAConfig`, window generation, the observation scenarios of §4.1 and the S0/S1 scenarios of §4.2. **A contract, not a mirror.** It emits the window dict that the DA and neural paths already consume: `true_state`, `obs`, `obs_mask`, `forcing_true`, `forcing_corrupted`, and per-parameter `<k>` / `true_<k>` / `<k>_da` keys (the format of `data/lorenz96.py:303` `_generate_window_dict`). It exposes a single `make_cgoa_s0_s1_datasets(cfg)` and does not copy L96's `RandomParam*`/`RandomBias*` class pair. |
| `models/dynamics.py` | `get_dynamics()` gains `cgoa` and `maooam`. Today nothing outside `tests/` calls `get_dynamics()`, and QG is absent from it. CGOA should be the first case study whose training and DA **both** build dynamics through it, so the factory stops being dead code. This closes the "SW/MAOOAM deferred" note at `PLAN.md` (L96 section). |

### 5.2 Shared pieces CGOA needs — extracted, not copied

| piece | today | change | why CGOA needs it |
|---|---|---|---|
| **Robust weak-constraint 4D-Var** | Two copies of one design: `L96Weak4DVar` (`evaluation/baselines.py:775`) and `QG4DVar` (`evaluation/run_qg_baselines.py:707`). The plain `Weak4DVar` diverges on two-scale L96, as the `L96Weak4DVar` docstring records. | Rename `L96Weak4DVar` → `WhitenedWeak4DVar` and keep the old name as an alias, since docs and changelog cite it. Its body is already generic; only the L96 defaults in `assimilate(F=8.0, c1=…)` move to the caller. Replace the scalar whitening scale `sigma = xb.std()` (line 915) with a **per-component** scale taken from `component_slices`. With one component this equals the scalar, so L96 is unchanged. | CGOA is two-scale by design, so the plain `Weak4DVar` should be expected to fail as it did on L96. v1's "applies unchanged" was wrong on this point. A single scalar scale is also wrong when atmosphere and ocean amplitudes differ by orders of magnitude: the ocean control is either frozen or blown up. |
| **Grouped scoring** | `evaluation/estimate_metrics.py` groups by the hard-coded L96 split (`slow` = first `NO = 8` dims, `obs_fast` = the rest). | `evaluate_estimates(..., groups: dict[str, slice] \| None)`. The default is the current L96 split, so every published L96 number is bit-identical. | CGOA reports atmosphere, ocean and interface separately. R2 needs the slow component's numbers on their own, not pooled with the fast ones. |
| **SCDA vs WCDA** | — | **Not a new class.** `ETKF`/`EnKF` already accept explicit per-time localization matrices (`loc_Lx_t`, `loc_Ly_t`), which is how QG passes its column localization. Add one helper, `block_localization(component_slices, obs_components)`. For WCDA it zeroes every state–observation pair across components; SCDA passes no cross-component mask. | Q2 becomes a one-argument difference on the same class and the same code path, so the comparison is like-for-like by construction. |
| **DA driver loop** | Written three times (`run.py`, `run_l96.py`, `run_qg_baselines.py`). | `evaluation/run_cgoa_baselines.py` stays **thin**: it builds windows and methods and nothing else. The loop (assimilate per window → cache → write `estimates_s0/s1.npz` in the archive layout) goes into one generic module, which CGOA uses first. L96 and QG migrate later, in `docs/plans/tech/multi_refactor_plan.md` Phase 3. | DA rows are then scored by `estimate_metrics` exactly like neural rows, which is already the L96 rule. |
| **Archive** | `evaluation/archive.py` has `RunArchive` entries for `system="l96"` and `"qg"`. | Add `system="cgoa"`. No artifact path is built anywhere else (`docs/plans/tech/archive.md`). | Reports regenerate from the canonical archive (`--check --portable`) from the first run, not after a later consolidation. |
| **Training** (WP7) | `train.py` handles L96 with `if system == "lorenz96"` branches (lines 667, 784). QG has its own argparse script. | Add CGOA through a dataloader-builder lookup keyed on `data.system`, not a third `if` branch. The model comes from `model_factory`, or from the registry once `docs/plans/tech/multi_refactor_plan.md` Phase 1 lands. **No `train_cgoa.py`.** | Keeps CGOA on the Hydra path, so it gets resume, cosine LR, `resolved_config.yaml` and monai backbones for free. |

The first four rows share one property: each change is **a no-op for the
existing case studies** (one component, default groups, no mask), so the
extraction PR is gated by unchanged L96/QG outputs, not by new behaviour.

### 5.3 Configs, scripts, reports

- **Configs:** `config/cgoa_default.yaml`, playing the role `lorenz96_default.yaml` plays for L96, plus Hydra experiments under `config/experiment/cgoa/` (monai backbones by default). Existing flat configs stay where they are.
- **Batch scripts:** two parameterized scripts, not one per experiment: `batch/cgoa/train.sbatch` (reads `EXPERIMENT=`) and `batch/cgoa/da.sbatch` (reads scenario and method). For contrast, L96 has 138 sbatch files, most of which differ only in `EXPERIMENT=` and log names.
- **Reports:** `reports/cgoa/`. Its generator reads estimates through `evaluation.archive` only.

### 5.4 Tests

- `tests/test_cgoa_dynamics.py` (fast):
  - the gyrostat-built L63 matches `models/lorenz63_dynamics.py`'s right-hand
    side;
  - `QuadraticTensorDynamics.step` accepts and ignores the L63 default keywords;
  - `dE/dt = 0` for the unforced, undamped core;
  - Casimirs conserved;
  - the drag term dissipates exactly `γ I_a (y_a − y_o)²`.
- **Extraction PR (§5.2), gated by no-op checks:**
  - `tests/test_da_golden_l96.py` passes unchanged.
  - `WhitenedWeak4DVar` with one component reproduces `L96Weak4DVar` locally. That class is in the golden test's `NO_CROSS_PLATFORM_VALUE`, so CI alone does not pin it.
  - `evaluate_estimates` with default groups is bit-identical on an archived L96 `estimates_s0.npz`.
- `block_localization` (fast): with atmosphere-only observations, **WCDA leaves
  the ocean analysis equal to its forecast** and SCDA does not. This is the
  cheapest falsifiable check that Q2 compares what it claims to.
- `tests/test_maooam_parity.py`: skipped without qgs.
- DA golden values on the Tier A config chosen in WP2, following
  `tests/test_da_golden_l96.py`.

## 6. Risks

| # | risk | mitigation |
|---|---|---|
| R1 | A gyrostat hierarchy can end up regular or weakly chaotic. Chaos needs an energy-conserving core *and* suitable forcing (Seshadri & Lakshmivarahan 2023). | WP2 calibrates the Lyapunov spectrum and time-scale ratio before any config is frozen. A0 starts from known-chaotic L63 and ENSO parameters. |
| R2 | Too much scale separation makes the ocean unidentifiable within a window, so Q1 has a trivial answer. | Sweep `τ`. Report the posterior spread of the slow component, not just RMSE. |
| R3 | **Mode-space states have no spatial neighbourhood.** The UNet1D and monai 1D backbones assume locality along the state axis (true for L96's ring). A chain of gyrostats or a vector of spectral coefficients lacks it. | Tier A: prefer the ring topology (A2). Tier B: observe and learn on qgs's physical-grid reconstruction, or use an MLP / attention backbone for mode space. Decision 3. |
| R4 | qgs pins (numba, sympy) could conflict with `fdv-monai-proto`. | qgs runs only in the offline export script, in a throwaway env if needed. The committed tensor is the interface. |
| R5 | MAOOAM spinup and decadal variability make truth generation slow. | Truth cache keyed by config, as QG does. The torch engine batches trajectories on GPU. |

## 7. Why this repo and not a new one

- The case study reuses the whole method core: `DynamicsBase`, the
  ETKF/EnKF/4D-Var baselines, FDV/CFM/SDA, the samplers,
  `evaluation/estimate_metrics.py`, and the training setup (resume, cosine LR,
  monai backbones).
- Its state is a low-dimensional vector, the closest possible match to the L96
  code path.
- The core still changes weekly. A copy would drift and break like-for-like
  cross-case tables.
- The fences are the `cgoa_*` / `maooam_*` prefixes, the dedicated worktree
  `4dvarnet-fm-cgoa` (see `docs/plans/tech/worktrees.md`), and this note instead of a new
  `PLAN.md` section.
- Revisit if the case needs a different stack or becomes a separate project.

## 8. Work packages

| WP | content | PR |
|---|---|---|
| WP0 | This note. | this PR |
| WP1 | `quadratic_ode.py` + `cgoa_dynamics.py` (with `component_slices`) + fast tests; config A0. | 1 |
| WP2 | Calibration of A1/A2: Lyapunov spectrum, time-scale ratio, energy spectra, attractor statistics. Freeze configs. Record in `docs/results/`. | 1 |
| WP3 | **Shared-piece extraction (§5.2)**: `WhitenedWeak4DVar` (per-component scale), grouped `evaluate_estimates`, `block_localization`, the generic DA loop, and the `cgoa` archive entry. A no-op for L96/QG, gated by §5.4. Independent of WP1/WP2, so it can run in parallel. | 1 |
| WP4 | `data/cgoa.py` (window-dict contract), `get_dynamics("cgoa")`, observation and model-error scenarios. | 1 |
| WP5 | DA baselines through the thin `run_cgoa_baselines.py`: SCDA vs WCDA, golden values, first report in `reports/cgoa/`. Needs WP3 and WP4. | 1 |
| WP6 | Tier B: tensor export, parity test, `maooam` dataset, baselines. Cross-check against Tondeur et al.'s EnKF regime. | 1–2 |
| WP7 | Neural rows (DirectUNet, FDV, CFM, SDA; monai) through `train.py`, same protocol as L96. | 1+ |

WP1–WP5 do not depend on any decision in §9 except decisions 2 and 5.

## 9. Open decisions

1. **Paper fit.** Three options:
   - P1 (model error): Q3 and the S1-struct scenarios fit its axis directly.
   - P2 (operator family): Q1 and the observation-sparsity axis.
   - A standalone coupled-DA paper built on Q2.

   This decides which WP7 runs are priority.
2. **Tier order.** Tier A first (recommended: controllable, cheap, tests the
   engine), or Tier B first (a published reference, no calibration risk)?
3. **Backbone for mode-space states** (R3): ring topology plus conv backbones,
   physical-grid observations, or a non-convolutional backbone?
4. **Scale of A2.** Match L96's `state_dim=40` (recommended, budgets carry
   over), or go larger to stress the method?
5. **Where the §5.2 extraction lands.** In WP3, as a standalone no-op PR
   (recommended: reviewable on its own and gated by unchanged L96/QG outputs),
   or folded into WP5 (fewer PRs, but a behaviour change and a refactor in the
   same diff)?

## References

- Gluhovsky & Tong (1999), *Phys. Fluids* 11(2), 334 —
  [link](https://pubs.aip.org/aip/pof/article-abstract/11/2/334/254000/The-structure-of-energy-conserving-low-order)
- Gluhovsky (2007), *J. Nonlinear Sci.* —
  [link](https://link.springer.com/article/10.1007/s00332-007-9006-6)
- Gluhovsky (2017), *Complexity* 6176045 —
  [link](https://onlinelibrary.wiley.com/doi/10.1155/2017/6176045)
- Seshadri & Lakshmivarahan (2023), *Physica D* 133948 —
  [arXiv 2307.12337](https://arxiv.org/abs/2307.12337)
- Seshadri & Lakshmivarahan, arXiv 2503.10782 v2 (2026) —
  [link](https://arxiv.org/html/2503.10782)
- De Cruz, Demaeyer & Vannitsem (2016), MAOOAM v1.0, *GMD* —
  [arXiv 1603.06755](https://arxiv.org/pdf/1603.06755)
- Demaeyer, De Cruz & Vannitsem, qgs, *JOSS* —
  [repo](https://github.com/Climdyn/qgs),
  [docs](https://qgs.readthedocs.io/en/latest/files/user_guide.html)
- Vannitsem (2014), OA-QG-WS v2, *GMD* 7, 649 —
  [link](https://ui.adsabs.harvard.edu/abs/2014GMD.....7..649V/abstract)
- Tondeur, Carrassi, Vannitsem & Bocquet (2020) —
  [arXiv 2003.03333](https://arxiv.org/abs/2003.03333)
- Garcia-Oliva, Carrassi & Counillon (2025), *NPG* 32, 439 —
  [link](https://npg.copernicus.org/articles/32/439/2025/)
- Peña & Kalnay (2004), *NPG* 11, 319 (coupled Lorenz-63).
