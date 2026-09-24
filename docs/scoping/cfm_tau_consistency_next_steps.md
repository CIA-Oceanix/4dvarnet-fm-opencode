# τ-consistency of the CFM operator — next steps

**Status:** DRAFT for review, **v3 2026-09-23**. Nothing implemented. Follows
`docs/results/cfm_tau_consistency.md`; section references (§2, §3) and property
labels (A1, B1, B4, D1, E1…) are that doc's.

**Version history:**
- **v1:** the first plan: probes, reseed, low-τ sampling, bootstrapped targets.
- **v2:** re-derived from the tower property. Added the truth-marginal test NS1 and split the bootstrap into T2a / T2b.
- **v3 (this):**
  - adds **T1′**, the exact control for T2a;
  - unifies T2a/T2b in **one pair construction** in x-space;
  - replaces the mean-slot architecture with **input zeroing (T4a)**;
  - makes two cautions explicit: no EMA infrastructure exists, and the T3 penalty can be satisfied by shrinkage;
  - grounds costs in the measured run time;
  - adds an **execution plan with parallel tracks (§6)**.

---

## 1. The two defects, restated through the tower property

The exact operator `D(x, tau, y) = E[x1 | x_tau = x, y]` satisfies the tower property
at every level of conditioning:

- **Marginal form (B1):** `E_{x_tau ~ p_tau(·|y)}[D(x_tau, tau, y)] = m(y)` for every τ.
- **Pairwise form (D1, the martingale property):** for `s < tau`, where `x_s` is a further-noised copy of `x_tau`, `E[D(x_tau, tau, y) | x_s, y] = D(x_s, s, y)`.
- **Second moment (B4, the law of total variance):** `P(y) = Var(D(x_tau)) + (b_tau²/tau)·E[∇_x D]`.

Measured (`docs/results/cfm_tau_consistency.md` §3), on both P1 M-tier flows:

1. **Defect 1 — a biased τ=0 prediction.** B1 drifts by 0.20 RMSE between τ=0 and the endpoint. The E1 slope is ≈ 1 where it should be 0. The flow corrects the τ=0 estimate within 1–2 Euler steps.
2. **Defect 2 — variance destroyed along τ ∈ [0.1, 0.4].** B4 decays 2.6–3.2× where it should be flat. The transport is self-consistent (B2), so this is where the ensemble under-dispersion (spread/RMSE ≈ 0.4–0.55) comes from.

**The unresolved caveat.** B1 and B4 were evaluated on the ODE's own states, not on
the true `p_tau(·|y)`. A violation therefore mixes operator inconsistency with a wrong
distribution of the ODE's states. NS1 resolves it.

## 2. What the tower property allows us to train

Each training window supplies one `(x1, y)` pair, i.e. **one** draw from `p(x1|y)`.

| form | needs | trainable? |
|---|---|---|
| B1 per `y` | several posterior samples of `x1` | **No.** Only with self-generated samples, which inherit the defect |
| D1 pairwise: `D(x_s) ≈ sg D(x_tau)` | one `x1`, two noise draws | **Yes, exactly** (pair construction in §3, T2) |
| D1 at `s = 0` | one `x1`; `x0` independent | **Yes.** The τ=0 special case of the row above |
| B4 per `y` | K ≥ 4 draws of x0 per `y` plus a JVP | yes, but costly and noisy (T3) |

---

## 3. Steps

**Common recipe.** All training arms use **PredictStateCFM-M** with the exact P1 recipe:
- monai backbone, `data.normalize: true`, cosine LR, 400 epochs, batch 16, lr 1e-3, clip 10.
- Base config: `config/experiment/A2_predictstatecfm_monaiM_l96.yaml`.
- Every new option defaults to today's behaviour, so no existing config retrains.

**Measured cost.**
- One training run is **about 36 min on an A100** (job 54694: 400 epochs).
- Evaluating one run takes about 1 hour on an RTX 8000: the P1 `ens30_no10` evaluation plus the two probes.
- Evaluation, not training, dominates wall time.

### NS1 — truth-marginal tower test (no retraining; ≈ 30 GPU-min)

Build `x_tau = tau·x1 + (1 − tau)·x0` from the **test-set truth** `x1`. This is the
exact distribution the network was trained on; no ODE is involved.

- **NS1a. Per-τ orthogonality curve on true marginals.** Regress the residual `x1 − D(x_tau^true, tau, y)` on `g(y) = ODE mean − D0`, pooled over windows, for τ ∈ {0, 0.1, …, 0.9}.
  - For an exact operator the slope is 0 at every τ, since `g(y)` is a function of `(x_tau, y)`.
  - At τ = 0 the curve is the measured E1 slope (≈ 1). As τ → 1 it goes to 0 trivially.
  - **The τ at which it falls to 0 is where the operator becomes consistent on its own training distribution.**
  - Regressing `D(x_tau) − D0` on `g` instead would be wrong: it inherits `D0`'s bias.
- **NS1b. Truth-marginal martingale test.** Build `x_s` from `x_tau^true` with the §3 T2 pair construction (s < τ). Regress `D(x_tau, tau) − D(x_s, s)` on features of `x_s` (`D(x_s, s) − D0` and `x_s` itself), over a grid of (s, τ) pairs. Every slope must be 0.
- **Reading the outcome:**
  - NS1a falls to 0 by τ ≈ 0.1 and NS1b ≈ 0: the defect is confined to τ = 0. Batch 1 stands, and **Batch 2 is dropped**: the B4 decay then comes from the ODE's states, so a sampler-side check comes first.
  - NS1a and NS1b clearly nonzero over [0.1, 0.4] (expected, given B4): the operator is at fault at low and mid τ. Batch 1, then Batch 2.
- **Implementation:** a flag on `reports/l96/probe_tau_consistency.py` (or a sibling script). Forward passes only.

### NS0 — sampler-side diagnostics (no retraining)

- **NS0a. Deterministic path from x0 = 0.** Is the ODE gain refinement or ensembling?
  - An `--x0-zero` flag on `probe_ode_mean_along_path.py`, one member.
  - ≈ 0.40 pooled means refinement; ≈ 0.44 means ensembling is needed.
  - **Caveat:** x0 = 0 is an atypical input (norm 0, vs about `sqrt(n)·s0` for a draw). Report it next to the single-random-member score (≈ 0.45).
- **NS0c. Accuracy vs cost of cheap point estimators.** An RMSE and network-call grid over N ∈ {1, 2, 3} steps × M ∈ {1, 4, 8, 30} members.
  - It decides what replaces the P1 "τ=0 mean" row. For example, 2 steps × 8 members is 16 calls.
- **NS0b. Martingale check under an ancestral stochastic sampler** (about 20 steps, Gaussian steps centred on `D`). **Optional:** NS1b tests the same identity more cleanly. Run only if NS1b and the ODE-based B4 disagree.

### Batch 1 — defect 1 (3 runs)

**T0 — baseline reseed.**
- The P1 config with `training.seed: 1`. The field exists; its default is `None`, so the P1 baseline was unseeded.
- With the P1 run, it gives a first estimate of run-to-run noise on every §4 metric. That estimate is weak (2 runs): if T2a's gain is under ~5%, a second T2a seed is needed before claiming anything.

**T1′ — the exact control for T2a (new in v3).**
- A fraction `f = 0.25` of each batch goes to **τ = 0 exactly**, with the ordinary target `x1`.
- T2a adds the same training mass at τ = 0 and only changes the target. So T2a vs T1′ isolates the target-variance mechanism from the extra-training-at-τ=0 effect.

**T2a — bootstrapped τ=0 targets.**
- **Loss.** For the same `f = 0.25` at τ = 0 with fresh x0, the target is `sg D_teacher(x_tau', tau', y)` instead of `x1`. Here `x_tau' = tau' x1 + (1 − tau') x0'`, with `tau' ~ U[0.2, 0.5]` and `x0'` independent of x0. The rest of the batch keeps the standard loss.
- **Why it is unbiased.** By the tower property, `E[D(x_tau') | y] = m(y)`, so the optimum is unchanged if the teacher is exact.
- **Why it should help.** The target's spread about `m(y)` falls ≈ 3.5× (measured at τ' = 0.2), and each window supplies many effective target draws instead of one `x1`.
- **Teacher — new infrastructure.** `LitModel` has **no EMA** today.
  - Add a deep copy of the model: frozen, in eval mode so dropout 0.1 is off, updated after each optimiser step with decay 0.999. With 63 steps per epoch that is a horizon of ≈ 16 epochs.
  - Keep it **out of the model's `state_dict`**, so `evaluation/neural_inference.load_model` still loads checkpoints. Check how strict that loader is about extra keys before implementing.
- **Warm start.** Until epoch 50 the f-fraction uses `x1` as the target, so the run is identical to T1′ up to that point.
- **Early abort.** Probe the epoch-100 checkpoint on 50 windows (≈ 5 min).
  - Abort if E1 or the τ=0 RMSE is worse than T1′ at the same epoch.
  - Watch the prediction RMSE at τ ∈ [0.2, 0.5]: if it degrades, the teacher is collapsing.

**Readout of Batch 1** (the guard-rail applies to every arm):

| outcome | meaning |
|---|---|
| T2a ≫ T1′ ≈ T0 on τ=0 RMSE and E1 | target variance was the cause. Citable mechanism for P2 |
| T1′ ≈ T2a ≫ T0 | capacity at τ=0 was the cause; the teacher adds nothing |
| none beyond T0 noise | not fixable at the loss level → T4a |
| ODE ensemble-mean RMSE or CRPS worse beyond T0 noise | the arm is rejected, whatever it does to τ=0 |

### Batch 2 — defect 2 (2–3 runs; only if NS1b is nonzero over [0.1, 0.4])

**T2b — general pairwise martingale target.**
- **Sampling.** Take a fraction `f = 0.25` with `tau ~ U[0.05, 0.4]` and `tau' = tau + Δ`, `Δ ~ U[0.1, 0.3]`.
- **Pair construction** (x-space, with no division by τ; shared with T2a and NS1b):
  - `x_tau' = tau' x1 + b_tau' ε'`
  - `x_tau = (tau/tau') x_tau' + sqrt(b_tau² − (tau/tau')² b_tau'²) ξ`
  - The square root is real for `tau < tau'`. The pair has exactly the joint law the martingale needs, and at τ = 0 the construction reduces to pure noise, i.e. T2a. **One code path serves T2a, T2b and NS1b.**
- **Target:** `(1 − α) x1 + α sg D_teacher(x_tau', tau')`, with α = 0.5 to start (or whatever Batch 1 favours) and optionally α = 1.
- **Readouts:** NS1b slopes over [0.1, 0.4], B4 flatness, spread/RMSE.

**T1 — low-τ sampling mixture (the capacity control for T2b).**
- `tau ~ (1 − f)·U[0,1] + f·U[0, 0.4]`, with `f = 0.5`.

### Batch 3 — fallbacks (not scheduled)

**T4a — zero the input at τ=0.**
- Take T1′ and feed `x0 := 0` whenever τ = 0, so A1 holds exactly by construction.
- It is one line, replacing v2's separate mean-slot architecture.
- The τ=0-only control failed (0.489) because its trunk never saw τ > 0; here it does.
- Run if T2a fixes the RMSE but A1 stays large, or for P2's "mean slot" story.

**T3 — B4 penalty.**
- A penalty on `(V(tau) − V(tau'))²`, with `V = Var_{x0}(D) + (b²/tau)·tr J / n`, K ≥ 4 draws of x0 per window plus a finite-difference JVP: 3–5× the training cost.
- **Caution: variance conservation constrains only the equality, not the level.** Shrinking both sides satisfies it. It needs an anchor, e.g. a stop-gradient on the τ' side or a calibration term.
- Only if T2b fixes NS1b but B4 still decays.

### Code changes (Batch 1 and 2 infrastructure; one PR)

| where | change |
|---|---|
| `conf/schema.py` | A `bootstrap:` block on the PredictStateCFM schemas: `frac`, `tau_min`, `tau_max`, `tau_prime_min`, `tau_prime_max`, `alpha`, `start_epoch`, `ema_decay`. `tau0_frac` for T1′ and T4a, `tau0_zero_input` for T4a, and `tau_sampling: uniform \| low_mix` for T1. All off or uniform by default |
| `models/vanilla_cfm.py` | `PredictStateCFM.compute_loss` gains the τ=0 fraction, the x-space pair construction, and an optional `teacher` argument. `MonaiPredictStateCFM` inherits all of it |
| `training/lightning_module.py` | The EMA teacher (a frozen copy, outside the model's `state_dict`), updated in `on_train_batch_end`, passed to `compute_loss`, and switched on at `start_epoch` |
| `config/experiment/` | `A2_predictstatecfm_monaiM_{seed1,tau0frac,boot0}_l96.yaml` for Batch 1, plus Batch-2 configs later |
| `batch/` | sbatch scripts cloned from `run_l96_a2_ps_m_train.sbatch` |
| `tests/` | Off-by-default gives a bit-identical loss under a fixed seed; no gradient reaches the teacher; the pair construction has mean `tau x1` and variance `b_tau²` given `x1` (statistical check); the teacher stays out of the checkpoint keys, and `load_model` round-trips; shapes and finiteness |

---

## 4. Evaluation (identical for every run)

1. P1 `ens30_no10`: RMSE, CRPS, spread/RMSE — S0 only.
2. `reports/l96/probe_ode_mean_along_path.py` and `reports/l96/probe_tau_consistency.py`.
3. NS1a and NS1b.

**Success criteria vs the T0 reseed pair:**

| metric | target | aimed at |
|---|---|---|
| τ=0 prediction RMSE | closes ≥ 50% of the gap to the ODE ensemble mean | defect 1 |
| E1 slope; NS1a curve | E1 from ≈ 1 toward 0; the curve at 0 for τ ≤ 0.5 | defect 1 |
| NS1b slopes over [0.1, 0.4] | toward 0 | defect 2 |
| B4 flatness, `B4(0.1) / B4(0.9)` | from 2.6 toward 1 | defect 2 |
| spread/RMSE (P1 convention) | rises from 0.40 | defect 2 |
| ODE ensemble-mean RMSE / CRPS | no regression beyond T0 noise | guard-rail |

## 5. Cost

| item | compute |
|---|---|
| NS1 + NS0a/c | ≈ 1.5 RTX-8000 hours, no training |
| Batch 1 (T0, T1′, T2a) | 3 × ≈ 40 min A100 (T2a ≈ +15% for the teacher pass), plus 3 × ≈ 1 h evaluation |
| Batch 2 (T2b, T1) | 2–3 runs, same unit cost, conditional |
| Batch 3 | not scheduled |

## 6. Execution plan — what runs in parallel

Nothing in Batch 1 depends on NS1: E1 ≈ 1 already establishes defect 1 at τ = 0.
NS1 gates only **Batch 2**. So the work splits into three tracks that start together.

| track | contents | depends on | runs on |
|---|---|---|---|
| **A — diagnostics** | NS1 plus the NS0a/c flags (reports-only PR), then the runs | nothing | local RTX 8000 |
| **B — training code** | the §3 code changes (one PR: models, training, conf, tests, configs) | nothing | dev + CI |
| **C — T0 reseed** | launch immediately: config only, no new code | nothing | SLURM A100 |

Then:

1. **Batch 1 launch:** T1′ and T2a as two parallel SLURM jobs, as soon as track B's PR is merged. T0 is already running or done.
2. **Batch 1 evaluation:** the three runs evaluate in parallel, each on its own GPU. **Never run two probes on one 48 GB card** (the Jacobian-transpose pass ran out of memory when two probes shared one). The T2a epoch-100 abort check runs mid-training on the local GPU.
3. **Gate 1:** Batch 1 readout (§3 table), combined with NS1 from track A, which is finished by then.
4. **Batch 2 launch:** only if NS1b is nonzero over [0.1, 0.4]. T2b and T1 run as parallel jobs, reusing track B's code; each is just a config.
5. **Gate 2:** Batch 2 readout decides whether Batch 3 (T4a or T3) is needed.

**Critical path:** track B's PR → Batch 1 training (≈ 40 min) → evaluation (≈ 1 h, in parallel) → Gate 1. Track A and T0 are off the critical path.

**PR hygiene:** tracks A and B are separate PRs, **both based on master** (stacked PRs get no CI). Each carries its own `CHANGELOG.d/` fragment. Checkpoints are archived to `experiments/l96/`, following the standing rule for benchmark-relevant runs.

## 7. Decisions needed

1. **Start tracks A, B and C together** (proposed), or run NS1 alone first?
2. **Batch 1 as T0 + T1′ + T2a** (proposed), or T2a alone?
3. **Seeds:** one per arm plus the T0 pair (proposed), or two per arm (≈ 2× cost).
4. **Paper home** (unchanged): the self-conditioning mechanism and T2a go to P2 (`docs/scoping/p2_unrolled_solvers_and_flows.md`); the B4 variance-decay diagnosis and T2b go to P1's calibration section.
