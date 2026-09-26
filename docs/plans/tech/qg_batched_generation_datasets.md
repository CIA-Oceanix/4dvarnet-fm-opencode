# Batched GPU generation for QG Options B and C, and a train/val/test dataset framework

**Status:** DRAFT v2 (2026-09-26): the six decisions of §8 are settled. Nothing implemented. It covers two things:

1. how to generate QG windows **in batches on GPU** for the one-way spectral
   wind drivers (Option B) and the two-way coupled system (Option C);
2. a **dataset framework**: train and val sets diverse enough, and a test set
   fully independent from both.

It builds on the Option B plan, `docs/plans/analysis/qg_gyrostat_wind_forcing.md`,
and on its PR-1 (#271: Fourier basis, gyrostat driver). It also builds on the
Option C description in `docs/plans/case_study/qg_gyrostat_synoptic_forcing.md`.
All timings are measured on one Quadro RTX 8000 (64 × 64 grid, float32), on a
machine other jobs were also using. Read them as ratios and orders of magnitude.

---

## 1. Summary

| | today | this plan |
|---|---|---|
| generation | serial, one window at a time, 28–32 s per window on GPU (~4.5 min on CPU) | batches of 256 windows, **0.25 s per window** one-way, **0.33 s** coupled |
| 1000/100/100 split | ~10 h on GPU | **~5 min** one-way, **~7 min** coupled (×1.5 margin: 8–10 min) |
| storage | 33.9 MB per window, **41 GB** for 1000/100/100 | ~3 MB per train/val window, ~16 MB per test window: **~5 GB** |
| seeds | ad hoc arithmetic per split | a hierarchical `SeedSequence`: splits disjoint by construction, test independent of train size |
| independence | one spinup per window (good); nothing checks it | the same rule plus automated leakage checks, a frozen and separately generated test set, and train-only statistics |

- **Batching is the dominant lever.** It gives about 100× and applies to
  every driver. The choice between drivers changes the cost by at most
  1.4×.
- **Two-way coupling costs 1.2–1.4× a one-way step once batched.** At batch
  size 1 it costs 2.25×; the difference is kernel-launch overhead.
- **Storage limits dataset size more than compute does.**

## 2. Measurements

### 2.1 Time per ocean step (batch size 1)

| step | time | vs unforced |
|---|---|---|
| unforced | 3.05 ms | 1.00× |
| current storm (grid Mexican hat + FFT at each RK4 stage) | 10.3 ms | 3.40× |
| spectral forcing (Option B) | 3.50 ms | 1.15× |
| two-way coupled prototype (Option C: gyrostat co-stepped on GPU, ψ₁ projection, eddy drag) | 6.87 ms | 2.25× |

### 2.2 Batching (time per window-step)

| batch size | unforced | spectral | coupled prototype | peak GPU memory |
|---|---|---|---|---|
| 1 | 3.05 ms | 3.46 ms | 6.80 ms | — |
| 16 | 0.197 ms | 0.224 ms | 0.438 ms | — |
| 64 | 0.048 ms | 0.055 ms | 0.111 ms | 0.05 GB |
| 128 | 0.025 ms | 0.029 ms | 0.055 ms | 0.08 GB |
| 256 | 0.027 ms | 0.027 ms | 0.036 ms | 0.15 GB |
| 512 | 0.026 ms | 0.026 ms | 0.032 ms | 0.29 GB |

What the table shows:
- Throughput **saturates around batch size 128–256**, at about 0.026 ms per
  window-step, so the batched step is over 100× cheaper than a serial one.
- Once saturated, the spectral forcing costs nothing extra. The coupled step
  settles at 1.2–1.4× unforced.
- Memory is not a constraint.

**Caveat.** The benchmark shares scalar ocean parameters across the batch.
Per-window parameters (§3.1) keep the FLOP count the same, but add small
per-window broadcast tensors. §5 applies a ×1.5 margin for this and for
post-processing.

### 2.3 Wind generation per window (CPU, numpy, serial)

| driver | time |
|---|---|
| current storm | 47 ms |
| spectral OU | 4 ms |
| gyrostat, 50-unit burn-in | 1.06 s |
| gyrostat, 5-unit burn-in | 0.15 s |
| surrogate | 1.22 s |

Once the ocean is batched, serial gyrostat generation on CPU would cost as
much as the ocean itself (§3.2 moves it to GPU).

### 2.4 Storage (a current `qg_truth_*.pt`, per window)

| field | shape | MB | needed? |
|---|---|---|---|
| `true_state` | 360 × 8192 | 11.80 | yes, but possibly thinned |
| `target_state_psi`, `target_state_q` | 360 × 4096 each | 11.80 | **no**: recomputable from `true_state` |
| `wind_curl` | 360 × 64 × 64 | 5.90 | **no**: recomputable from the wind state (3 or 12 numbers per step) |
| `init_lead_truth` | 121 × 8192 | 3.96 | for initial-condition lag sampling |
| obs, masks, columns, wind state | — | 0.4 | yes |
| **total** | | **33.9** | |

That makes 1000/100/100 = 41 GB and 5000/500/500 = 203 GB. After the
24 September disk-full incident (149 GB deleted), this is the binding
constraint.

### 2.5 Measured after G1 (`scripts/bench_qg_batched.py`, RTX 8000)

With `BatchedQGDynamics`, per-window `rd/U1/rek`, the spectral forcing and
the eddy drag, at batch size 256:

| quantity | value |
|---|---|
| ocean time per window-step | **0.028 ms** (the plan assumed 0.027: risk R2 cleared) |
| full batch: forced 2-year spin-up + 481 window steps (9241 steps) | wind 29 s + ocean 66 s = **0.37 s per window** |
| wind generation (gyrostat, 50-unit burn-in + 9241 steps) | 29 s **per batch**, launch-bound, so independent of batch size |
| batch dependence of the ocean on GPU | none: a window alone and in a batch of 256 are bit-identical after 50 steps |

Generating a split's wind in one large batch, the cost is about 0.26 s per
window (ocean) + about 30 s per split (wind). **5000/500/500 takes about
26 min on this GPU.**

## 3. Part A: batched GPU generation

### 3.1 `BatchedQGDynamics` (new class; `QGDynamics` untouched)

`QGDynamics` stores its parameters as scalars. `rd` enters the inversion
matrices `a11 … a22` and `F1, F2`; `U1, U2, beta, rek` are floats passed to
`_tendency`. The new class holds them as tensors of shape `(B,)` and
broadcasts them:

- inversion matrices `(B, 1, ny, nk)`, which is tiny;
- `Ubg` `(B, 2, 1, 1)`, `ikQy` `(B, 2, 1, nk)`, `rek` `(B, 1, 1, 1)`;
- the same RK4, filter and clip as `QGDynamics`, reusing its code paths
  wherever the math is identical.

It is a separate class, so the legacy benchmark path stays bit-identical.

**Parity test:** a batch of B windows with different parameters matches B
independent `QGDynamics` rollouts to float32 tolerance. It cannot be bitwise,
because cuFFT results depend on the batch size.

### 3.2 Batched wind on GPU (`models/qg_wind_modes.py`, `models/gyrostat_driver.py`)

- **Gyrostat:** a torch right-hand side over `(B, M)` using `index_add`; the
  Option C prototype already has it. It integrates on the GPU in the same
  loop as the ocean, or ahead of it for Option B.
- **Burn-in (revised in G1):** the attractor-state cache is dropped. Each
  window gets its **own** 50-unit burn-in (≈ 62 Lyapunov times, λ = 1.25)
  from an initial state drawn from its own seed. All windows burn in
  together as one batch on the GPU, so the cost does not grow with the
  number of windows (§2.5), and windows are independent by construction,
  which is stronger than well-separated offsets on a shared trajectory.
- **Reproducibility:** the gyrostat right-hand side uses gathers and
  explicit elementwise adds over a fixed, padded role table, with no atomic
  scatter and no reductions. Noise and surrogate phases come from
  per-window generators, and the surrogate FFT is done per window. A
  window's wind is therefore **bit-identical whatever batch it is generated
  in** (tested). This matters because the burn-in amplifies roundoff by
  about e⁶⁰.
- **OU and surrogate:** OU is batched as an exact AR(1). The surrogate uses
  batched `torch.fft`, with the same random phases across the modes of a
  window.
- **Current storm:** batched through `wind_curl_field`, which already
  accepts batch dimensions. It stays expensive (3.4×), so it is only used
  for comparison datasets.

### 3.3 Option B path

The amplitudes `(B, T, 12)` are precomputed, then written straight into the
spectrum at each stage (`curl_spectral`, a K-term product). This is the
PR-2 hook, applied to `BatchedQGDynamics`.

### 3.4 Option C path

The gyrostat state `y (B, 12)` is **co-stepped inside the ocean's RK4
stages**. Each stage computes:

1. **ocean → atmosphere:** the projection of ψ₁ onto the 12 modes, which
   means reading the 6 complex Fourier coefficients of ψ̂₁ (no FFT);
2. **atmosphere → ocean:** the spectral PV source `curl_spectral(a(y))`;
3. **exchange terms**, in the momentum-conserving, dissipative form of
   `docs/plans/case_study/cgoa_coupled_gyrostats.md` §3.3:
   - relative-wind eddy drag on the ocean, `−r_cf ζ̂₁`, which is diagonal in
     spectral space;
   - drag between `y` and the projected ψ̂₁ on the atmosphere.

**Tests:** the energy budget (the exchange dissipates, never creates); the
Option B path is recovered when the coupling is 0; and the tangent-linear
map is available through autograd for 4D-Var.

**Physics decisions left open** (§8): the values of `r_cf` and γ, and how
ψ̂₁ is scaled into the atmospheric modes.

**Spinup.** Feedback only matters in coupled equilibrium, so Option C
datasets spin up **coupled**. One-way datasets can use either an unforced
spin-up (today's convention) or a forced one (decision 2). With batching,
the cost difference is small (§5).

> **Implemented in G3** (`models/qg_coupled.py`: `CoupledSpectralQG`,
> `generate_coupled_windows`). Measured on an RTX 8000 at batch 256:
> - the coupled step costs **0.046 ms per window-step**, 1.63× the one-way
>   step (0.028). The prototype's 1.2–1.4× left out the per-stage frame
>   rotation and projection;
> - feedback ÷ the gyrostat's own tendency (median / p90):
>   **κ_fb = 1: 0.75% / 2.3%**, κ_fb = 10: 7.5% / 22%, κ_fb = 100:
>   57% / 119%;
> - after 60 days, the gyrostat state differs from κ_fb = 0 by 4% / 41% /
>   173% (relative).
>
> With κ_fb = 0 the run reproduces the one-way path bit for bit (tested).

### 3.5 Sharding and determinism

- **Sharding:** per-window seeds (§4.2) make sharding free. A SLURM array
  splits the window indices, each task generates its shard in batches, and a
  final step checks the manifest.
- **Determinism:** batched float32 GPU results depend on the batch size and
  the hardware (cuFFT). The manifest records the device, batch size and code
  version. **Test sets are generated once and frozen on disk**, never
  regenerated. Train sets may be regenerated (§4.6).

## 4. Part B: the dataset framework

### 4.1 Spec and manifest

`QGDatasetSpec` is a versioned dataclass. It holds:
- the driver (`ou`, `spectral_ou`, `gyrostat`, `gyrostat_surrogate`,
  `coupled`);
- the factor ranges (§4.4);
- the split sizes;
- root entropies (one for train and val, **a separate one for test**);
- spin-up mode and storage resolution;
- `GENERATOR_VERSION`.

Generation writes a `manifest.json` per split. For every window it records
the spawn key, the drawn factors, a hash of the initial ocean and atmosphere
state, and summary statistics (rms curl, KE, regime occupancy). The spec
hash is the cache key. This replaces the whole-`QGConfig` hash for new
datasets; legacy datasets keep theirs.

### 4.2 Seeds

A hierarchical `numpy.random.SeedSequence(entropy=root, spawn_key=(split_id,
window_idx, stream_id))`:
- `split_id`: train = 0, val = 1, test = 2, and OOD sets 10+;
- `stream_id`: one per random stream (factors, ocean initial state,
  atmosphere, observations, S1 corruption).

**Splits are disjoint by construction.** Test uses its own entropy, so **the
test set does not change when train or val sizes change**, and train
generation can never touch test streams.

**The legacy scheme, for reference.** Today window `i` uses
`RandomState(seed + i)` for its factors, `seed + 2000 + 101 i` for the wind
and `seed + 3000 + 101 i` for the ocean, with split seeds 42 / 10042 / 20042
(production 1000/100/100). Checked arithmetically:
- **no exact seed collides** across splits or streams, because offsets that
  are multiples of 1000 are never multiples of 101;
- but the seed ranges **interleave** (the train wind seeds 2042–102941
  contain every val and test wind seed);
- and independence holds only because the arithmetic happens to work, and
  nothing tests it.

It stays for the legacy `ou` benchmark datasets, since changing it would
change every published QG number.

### 4.3 Independence rules (enforced, not assumed)

1. **One independent spinup per window**, which is today's rule, kept: no
   two windows share an ocean trajectory. If windows are ever cut from one
   long run to save compute, **all windows of a trajectory go to the same
   split** (group split). §5 shows batching makes that saving unnecessary.
2. **Atmosphere:** each window's gyrostat state comes from its own split's
   attractor trajectory, at offsets more than 15 Lyapunov times apart. The
   gyrostat amplitudes stay correlated at about 0.3 for 60–90 days, so
   windows cut from one atmospheric run within a few months of each other
   would **not** be independent.
3. **Observations and S1 corruption** use per-split streams.
4. **Normalization statistics come from train only** (already true through
   `precompute_qg_norm_stats.py`). The same goes for any data-driven choice:
   thresholds, early stopping on val, hyper-parameters on val.
5. **Test is generated by a separate job into a read-only directory.** The
   train and val loaders refuse a test manifest, and the evaluation script
   is the only reader.
6. **Automated leakage checks**, run at generation and in CI on small specs:
   - spawn keys disjoint across splits;
   - hashes of the initial ocean and atmosphere states disjoint;
   - nearest-neighbour distance from each test window to train (in factor
     space and on the initial ψ₁) no smaller than the distribution of
     within-train nearest-neighbour distances (a one-sided test).

### 4.4 Diversity

**Factors, sampled by a scrambled Sobol / Latin hypercube design per split**
(independent designs for train, val and test):

| factor | range | note |
|---|---|---|
| `rd`, `U1`, `rek` | ±15% of nominal | as today |
| wind level | 0 – 3×10⁻¹¹ s⁻², continuous, plus a **20% calm stratum** at 0 | today: a 5-level cycle with one in five calm |
| drift `cx`, `cy` | [0.25, 0.75], [−0.06, 0.06] m/s | as today |
| start phase `(x0, y0)` | uniform over the domain | as today |
| gyrostat time unit | 30–90 d per unit (memory 7–22 days) | new; diversity in forcing time scale |
| ring coupling ε | 0.3–0.8 | new; only once each ε's attractor constants are measured |
| Option C: `r_cf`, γ | ranges from decision 4 | new |
| observations | on-the-fly geometry and noise (train), fixed per window (val, test) | as today |

**Regime stratification.** The gyrostat's lobe state at the start of the
window is recorded. Train and val are stratified so that each regime
combination appears, and rare regimes are not left to chance.

**Diversity report,** written with each manifest:
- factor coverage (the design's discrepancy);
- histograms of rms curl, KE and regime occupancy;
- a comparison of val against train (Kolmogorov–Smirnov per statistic).

Test gets the same report, **read only after** the model is frozen.

**OOD test sets.** Optional and clearly labelled: factors at or beyond the
box edges (e.g. a time unit outside 30–90 d, or an unseen coupling). They are
reported separately, never pooled with the in-distribution test.

### 4.5 Storage format (new datasets)

- Store `true_state` thinned to the observation interval (every 6 steps,
  12 h), `init_lead_truth` thinned the same way, the wind state (3 or 12
  numbers per step), observations and seeds. **Recompute** the ψ/q targets
  and the curl field on load.
- That is ≈ 2.0 + 0.7 + 0.4 ≈ **3 MB per window** in float32, against
  33.9 MB today (~11× less).
- **Test sets at full time resolution** (small N; DA scoring may need it),
  but still without the recomputable fields: ≈ 11.8 + 4.0 + 0.4 ≈ 16 MB per
  window. 1000/100/100 comes to about 3.4 GB (train and val) + 1.6 GB
  (test) ≈ 5 GB.

### 4.6 Fresh train windows every epoch (decision 5)

> **Implemented in G5** (`data/qg_specwind_neural.py`,
> `train_qg_neural.py --specwind-spec`). Train windows are regenerated
> every `--regen-every` epochs (`--regen-windows` per round), from spawn
> keys at index ≥ 10⁹ in the train namespace, with a per-round Latin
> hypercube. Val is re-materialized at full resolution by deterministic
> regeneration and verified against the stored 12-hourly frames (bit-exact
> on GPU). Test is read from its full-resolution shards with purpose
> `test`. Normalization stats come from the first train draw. Only
> `cond_mode` `none` and `true` are supported until the spectral S1
> corruption lands (Option B PR-2).

Batched generation makes it cheap to **regenerate train windows on the fly**
from fresh spawn keys: about 4–6 min per 1000 windows on one RTX 8000.
- Training can then draw new windows every epoch or every k epochs, which is
  diversity limited only by the factor box.
- Val and test stay frozen.
- Nothing is stored for train.
- Useful for the neural rows, since train diversity is otherwise capped by
  disk.

## 5. Compute time

Steps per window: 8760 spin-up (2 years) + 481 forced (10-day lead + 30-day
window). Batch size 256. Batched time per window: 0.027 ms per step one-way,
0.036 ms coupled. A ×1.5 margin covers per-window parameters and
post-processing.

| generation | per window | 1000/100/100 (1200 windows) | 5000/500/500 (6000 windows) |
|---|---|---|---|
| **legacy, serial, GPU** (measured) | 28.4 s (spectral) – 31.7 s (storm) | 9.5 – 10.6 h | 47 – 53 h |
| legacy, serial, CPU (~4.5 min per window, from `data/qg.py`) | ~270 s | ~90 h | ~450 h |
| **Option B, batched**, unforced spin-up | 0.25 s (0.37 with margin) | **5 min (7.5)** | **25 min (37)** |
| Option B, batched, forced spin-up | 0.25 s (0.37) | 5 min (7.5) | 25 min (37) |
| **Option C, batched**, coupled spin-up | 0.33 s (0.50) | **6.7 min (10)** | **33 min (50)** |
| gyrostat wind, CPU numpy (5-unit burn-in) | 0.15 s | 3 min | 15 min |
| gyrostat wind, batched on GPU (§3.2) | ≈ 0 | seconds | seconds |
| train regeneration (§4.6) | — | 4–6 min per 1000 train windows | — |

| storage | per window | 1200 windows | 6000 windows |
|---|---|---|---|
| today's format | 33.9 MB | 41 GB | 203 GB |
| §4.5 format (test at full resolution) | ~3 MB (test ~16 MB) | ~5 GB | ~25 GB |

**Reading the tables:**
- With batching, a production-size dataset takes minutes of GPU time for
  either option.
- Coupling adds ~35% at saturation.
- Not included: neural training and DA evaluation, which will dominate the
  budget (DA on 100 test windows × methods × scenarios), and the H100 / H200
  speed-up, which is not measured.

## 6. Work breakdown

| PR | content | depends on |
|---|---|---|
| PR-2 (Option B plan) | the dynamics, data and S1 hooks for the spectral drivers on `QGDynamics` (unchanged scope) | #271 |
| G1 | `BatchedQGDynamics` (with the eddy-drag term `r_cf`, §8.1), batched GPU wind (gyrostat, OU, surrogate, attractor-state cache), parity tests | #271 |
| G2 | `QGDatasetSpec`, manifests, `SeedSequence` streams, independence rules and leakage checks, the §4.5 storage format, the diversity report, sharded generation script + sbatch | G1 |
| G3 | Option C coupled step (§3.4) on `BatchedQGDynamics`, energy-budget and zero-coupling tests | G1; decision 4 |
| G4 | generate the datasets (B first, then C) and record the manifests and diversity reports in `docs/results/` | G2 (+ G3 for C) |
| G5 | train-time regeneration in the QG neural loader (decision 5) | G2 |

The Option B plan's PR-3 (long forced runs and ensembles) also benefits
from G1: its ensembles become one batch.

## 7. Risks

| # | risk | mitigation |
|---|---|---|
| R1 | Batched and serial results differ at float32 level, and a reviewer expects bit-identity. | Parity at tolerance, stated in the manifest; frozen test sets; the legacy path untouched. |
| R2 | Per-window parameters change the benchmark's throughput. | Measure in G1 with per-window parameters before committing to §5's numbers (hence the ×1.5 margin). |
| R3 | The coupled step's physics is not settled (decision 4), so C datasets need regenerating. | Cheap: minutes. `GENERATOR_VERSION` and the spec hash invalidate caches. |
| R4 | Attractor-state cache offsets are correlated, which leaks within a split or across splits. | Per-split trajectories; offsets > 15 Lyapunov times; checked by the leakage tests. |
| R5 | Thinned storage breaks a consumer that expects every step. | Audit the QG neural and DA loaders in G2; test at full resolution; recompute on load behind one accessor. |
| R6 | Disk usage. | The §4.5 format; ensembles and diagnostics on node-local `/tmp` (the standing rule). |

## 8. Decisions (settled 2026-09-26)

| # | decision | outcome |
|---|---|---|
| 1 | dataset size | **5000/500/500** for the new drivers (~25 GB in the §4.5 format) |
| 2 | one-way spin-up | **forced** for new datasets (same cost once batched) |
| 3 | factor box | **gyrostat time unit (30–90 d per unit) added now**; ring coupling ε added only after its attractor constants are measured |
| 4 | Option C coupling | **physical bulk coefficients plus a labelled feedback-gain sweep**; see below |
| 5 | train-time regeneration | **build it** (G5 is no longer optional) |
| 6 | legacy datasets | **unchanged**: the `ou` benchmark keeps its legacy seeds and format |

### 8.1 Decision 4: the Option C coupling coefficients

Both exchange terms follow from one bulk relative-wind stress
`τ = ρ_a C_D |U_a| (u_a − u_o)`, applied with momentum conservation between
an atmospheric slab (depth `H_a`) and the upper ocean layer (depth `H₁`):

| term | where | coefficient | value (`ρ_a` = 1.2, `C_D` = 1.2×10⁻³, `|U_a|` = 10 m/s, `H_a` = 10 km, `H₁` = 500 m) |
|---|---|---|---|
| eddy drag `−r_cf ζ₁` | ocean upper-layer PV | `r_cf = ρ_a C_D |U_a| / (ρ₀ H₁)` | **2.8×10⁻⁸ s⁻¹** (damping time ~1.1 yr; bottom drag `rek` = 5.8×10⁻⁷) |
| `+γ_a r_cf ζ_{o,k}` | tendency of wind amplitude `a_k` | `γ_a = C_D |U_a| / H_a` | **1.2×10⁻⁶ s⁻¹** (9.6-day spin-down) |

**How ψ̂₁ is scaled into the atmospheric modes.** The wind amplitude `a_k`
is a PV source, `a = r_cf ζ_a`. The atmospheric drag
`−γ_a (ζ_a − ζ_o)` therefore becomes `da_k/dt ⊃ −γ_a a_k + γ_a r_cf ζ_{o,k}`,
where `ζ_{o,k} = −|k|² ψ̂₁,k` is the ocean's relative vorticity projected on
mode k. The `−γ_a a_k` part is the atmosphere's own drag, already inside the
gyrostat's damping, so it is not added a second time. Only the
ocean-dependent term `+γ_a r_cf ζ_{o,k}` is new. In standardized gyrostat
units it is divided by the mode's std `s_k` and multiplied by the time unit.

**Measured magnitudes** (one-year gyrostat-forced run from the shared
spin-up; nominal ocean, level 10⁻¹¹):

| quantity | value |
|---|---|
| ocean vorticity projected on the 12 modes (rms) | 1.0–4.5 × 10⁻⁷ s⁻¹ |
| **ocean → atmosphere** feedback / the amplitudes' own tendency (`s_k` per 15 d) | **0.2–1% at `|k|` = 1–√2, 3% at `|k|` = 2** |
| **atmosphere → ocean** eddy drag on the full field: `r_cf ζ₁` rms / mean rms curl | 3.2×10⁻¹³ / 3×10⁻¹² ≈ **10%** |

**Consequences:**
- With physical coefficients the coupling is **one-sided**. The eddy drag
  on the ocean is a real, ~10% effect. The feedback on the atmosphere is at
  the percent level, as expected for purely mechanical coupling: the strong
  real-world feedback is thermal (Option E, QG-TC).
- **The eddy drag does not need co-stepping.** It depends only on the
  ocean, so "Option B + eddy drag" reproduces physical Option C to within
  the measured ≤ 3.5% feedback at the cost of Option B. G1 therefore adds
  `r_cf` to the batched one-way path.
- **The co-stepped C path (G3) is kept for the feedback-gain sweep:**
  `κ_fb ∈ {1 (physical), 10, 100}` multiplies only the ocean → atmosphere
  term. `κ_fb > 1` is **explicitly non-physical**: a sensitivity study of how
  much a learned DA/forecast system gains when the atmosphere carries ocean
  information. It is labelled as such in every manifest and report.
- Diversity factor: `r_cf ∈ [0, 1×10⁻⁷] s⁻¹` (which spans `|U_a|` of about
  0–35 m/s or shallower effective layers). `κ_fb` is a fixed label per
  dataset, not a sampled factor.
