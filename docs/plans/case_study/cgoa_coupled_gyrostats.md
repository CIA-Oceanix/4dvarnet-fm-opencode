# Coupled ocean–atmosphere case study from coupled gyrostats (CGOA)

**Status:** SCOPING v1 (2026-09-24). Nothing implemented. Proposes a fourth
case study, after L63, L96 and QG. It is built **in this repo**, fenced by a
`cgoa_*` prefix; §7 gives the reasons. **Which paper it serves (P1, P2 or its
own) is open** — see §9, decision 1.

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

**One engine for both tiers.** Both are quadratic ODEs,
`dx/dt = Σ T_ijk x_j x_k` with `x₀ = 1` carrying the linear and constant terms.

| file | content |
|---|---|
| `models/quadratic_ode.py` | `QuadraticTensorDynamics(DynamicsBase)`: sparse `T`, batched `einsum`, RK4, GPU. Autograd supplies the adjoint, so `Strong4DVar` and `Weak4DVar` in `evaluation/baselines.py` apply unchanged. |
| `models/cgoa_dynamics.py` | Gyrostat builder: blocks, topologies (chain / ring / dense), inertia, the three coupling mechanisms → `T`. Energy and Casimir diagnostics. |
| `models/maooam_dynamics.py` | Loads the cached qgs tensor into `QuadraticTensorDynamics`. |
| `scripts/export_maooam_tensor.py` | One-off qgs → `.npz` export (needs qgs, run offline). |
| `data/cgoa.py` | Config, datasets, window generation, the observation scenarios of §4.1. Mirrors `data/lorenz96.py` (`make_*_s0_s1_datasets`). |
| `models/dynamics.py` | `get_dynamics()` gains `cgoa` and `maooam`. This closes the "SW/MAOOAM deferred" note at `PLAN.md` (L96 section). |
| `evaluation/run_cgoa_baselines.py` | ETKF/EnKF/4D-Var, plus a **WCDA variant**: the analysis zeroes the cross-component covariance blocks. |
| `config/experiment/cgoa/`, `batch/cgoa/`, `reports/cgoa/` | Hydra configs (monai backbones by default), SLURM scripts, reports. |

**Tests:**

- `tests/test_cgoa_dynamics.py` (fast):
  - the gyrostat-built L63 matches `models/lorenz63_dynamics.py`'s right-hand
    side;
  - `dE/dt = 0` for the unforced, undamped core;
  - Casimirs conserved;
  - the drag term dissipates exactly `γ I_a (y_a − y_o)²`.
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
| WP1 | `quadratic_ode.py` + `cgoa_dynamics.py` + fast tests; config A0. | 1 |
| WP2 | Calibration of A1/A2: Lyapunov spectrum, time-scale ratio, energy spectra, attractor statistics. Freeze configs. Record in `docs/results/`. | 1 |
| WP3 | `data/cgoa.py`, `get_dynamics("cgoa")`, observation and model-error scenarios. | 1 |
| WP4 | DA baselines, SCDA vs WCDA, golden values. First report, `reports/cgoa/`. | 1 |
| WP5 | Tier B: tensor export, parity test, `maooam` dataset, baselines. Cross-check against Tondeur et al.'s EnKF regime. | 1–2 |
| WP6 | Neural rows (DirectUNet, FDV, CFM, SDA; monai), same protocol as L96. | 1+ |

WP1–WP4 do not depend on any decision in §9 except decision 2.

## 9. Open decisions

1. **Paper fit.** Three options:
   - P1 (model error): Q3 and the S1-struct scenarios fit its axis directly.
   - P2 (operator family): Q1 and the observation-sparsity axis.
   - A standalone coupled-DA paper built on Q2.

   This decides which WP6 runs are priority.
2. **Tier order.** Tier A first (recommended: controllable, cheap, tests the
   engine), or Tier B first (a published reference, no calibration risk)?
3. **Backbone for mode-space states** (R3): ring topology plus conv backbones,
   physical-grid observations, or a non-convolutional backbone?
4. **Scale of A2.** Match L96's `state_dim=40` (recommended, budgets carry
   over), or go larger to stress the method?

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
