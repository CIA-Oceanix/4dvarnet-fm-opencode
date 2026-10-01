# L96 P1 — FM-operator score with window-bootstrap intervals; P1 report and paper inserts

**Status:** PLAN v1 2026-10-01 — (A) the overnight rerun, (B) the P1-report section, (C) the paper
inserts. Branch `feature/p1-fm-score` (worktree `4dvarnet-fm-p1-fm-score`). Builds on
`docs/results/l96_p1_fm_score.md`; numbers quoted in B/C are the current z-unit means and are
replaced by physical-unit values with 95% intervals once A has run.

## Decisions taken (2026-10-01 discussion)

- **Uncertainty = 95% window-bootstrap interval** (2000 replicates, percentile), applied to the
  statistic as reported, whatever its averaging convention. Resampling unit = the test window
  (time steps and channels within a window are correlated). Ratios (spread/skill, calibration loss,
  non-Gaussian) are ratios of resampled per-window totals; c* is re-fitted in every replicate.
- **Comparisons are paired**: the same resampled window indices for both schemes, interval on the
  difference, plus the per-window win count out of 200.
- **Seeds**: per-window values averaged over the 3 seeds before resampling; the seed range is
  reported alongside.
- **Units**: FMS reported in **physical units**, FMS_phys(τ) = mean_c σ_c² FMS_z,c(τ) (the operator
  is affine-equivariant, so this is the same score with the error measured in physical units; τ keeps
  its z-space SNR meaning). z-units kept in the JSONs.
- **Open — RMSE convention (user decision):** keep the benchmark RMSE (mean over windows × channels
  of the per-window per-channel RMSE) or move P1 to the pooled RMSE √(mean MSE_w) (consistent with
  FMS(τ = 0) and the error decomposition; e.g. PSC-M 0.340 → 0.389, ETKF 0.610 → 0.714). Both are
  computable from stored per-window scores; B and C are written so either drops in.

## A. Overnight run

### A1. Code (before submission)

1. `reports/l96/probe_fm_score.py::score_members` additionally returns per-window arrays, written
   to `fmscore_<case>.npz` next to the JSON (float32):
   - `fms_<variant>` (W, 24, n_τ) for variant ∈ {gauss, ens, kde, snap}, τ ∈ {0, .25, .5, .75, .95}
     (time-mean per window and channel, z-units);
   - `se` (W, 24) time-mean squared error of the ensemble mean, `var` (W, 24) time-mean member
     variance (z-units) — spread/skill numerator and denominator;
   - `scale_z`, `scale_phys` (W, 3 groups, 4 τ, 42 scales) per-window sums for the variance-rescaling
     sweep (z- and σ²-weighted), so c* can be re-fitted per bootstrap replicate;
   - `snap_ess` (W, n_τ).
   ≈ 1 MB per run and case. The pooled JSON is unchanged (regression check, A3).
2. `reports/l96/probe_fm_score_point.py` (CPU): FMS of point estimates = squared error at every τ,
   per window and channel, for DirectUNet-M (1200 ep, 3 seeds, `estimates_s*.npz`) and Strong-4DVar
   (regular grid only; the random-layout trajectories were not kept).
3. `reports/l96/probe_fm_score_da_gauss.py` (CPU): Gaussian FMS per window from the **stored**
   benchmark ETKF / EnKF / ETKS per-time means and variances (`da_current_2026-09-29`, and the
   random-layout bundle) — the benchmark-file DA rows, Gaussian variant; complements the member
   re-run (which gives all four variants on a fresh realisation).
4. `evaluation/bootstrap.py`: `window_bootstrap(per_window: dict[str, ndarray], stat, n=2000,
   seed=0)` → (value, lo, hi); `paired_bootstrap(a, b, stat, ...)` → (Δ, lo, hi, wins). Statistics:
   mean (FMS, MSE, CRPS), sqrt-of-mean (pooled RMSE), mean of per-window RMSE (benchmark RMSE),
   ratio of sums (spread/skill, non-Gaussian), argmin-refit (calibration loss).
5. `tests/test_bootstrap.py`: the interval of a mean matches the analytic ±1.96 σ/√W on Gaussian
   data; paired interval excludes 0 for a constant shift hidden under large between-window
   variance; mean of per-window Gaussian FMS = pooled Gaussian FMS (linearity).
6. `reports/l96/summarise_fm_score_current.py` → reads the npz, writes
   `reports/l96/outputs/fm_score/fm_score_current.md` in the layout of section B and a
   `fm_score_current.json` of every (value, lo, hi). Per-run npz stay in the report-input bundle
   `experiments/l96/report_inputs/fm_score_current/` (not committed; ~160 MB); JSON summaries are
   committed.

### A2. Runs

| job | tasks | where | estimate |
|---|---|---|---|
| learned rows (`batch/run_l96_fm_score_current.sbatch`, unchanged rows) | 36: PSC-M, Van-M (1200 ep), SDA1/2/3-fix-M (1200 ep, gw 25), DU-M → SDA3-fix hybrid; seeds 1–3 × {regular, random} | split as last night: a100 on br-206/207 (`%6`), L40S on br-209 (`%2`) | 19 min (L40S) – 70 min (shared a100) per task; done by morning if the queue allows |
| DA with members (`batch/run_l96_fm_score_da.sbatch`) | 2 (regular, random) | br-209 L40S | ~3 h |
| point rows + Gaussian DA from stored files | CPU, local | sl-mee-br-202 | < 1 h |

GPU share: my jobs stay on ≤ 7 of 15 Odyssey_GPU nodes (memory `feedback_gpu_cluster_share`).
Members never leave node-local storage.

### A3. Checks before anything is reported

- Every learned row reproduces its benchmark RMSE (as on 2026-09-30/10-01).
- Pooled values from the new npz equal last night's pooled JSONs: exactly for `gauss` (closed form)
  given identical members; within sampling noise otherwise (GPU sampling is not bit-reproducible).
- Mean of per-window FMS = pooled FMS for every row (linearity).
- Bootstrap: interval widths scale ~1/√W (spot check at W = 50 / 200).

### A4. Out of scope tonight

Integrated-over-τ summary; time-since-last-observation breakdown; 100-member ETKS; Weak-4DVar.

## B. P1 report insert

**Where:** a new shared section `## Distributional score FMS_τ (current benchmark)`, generated by a
new module `reports/l96/fm_score_summary.py` (same pattern as `per_window_summary.py`) and appended
to `p1_l96_benchmark.md` right after "Per-window RMSE / EV / CRPS summary"; optionally also to
`l96_benchmark_extended.md`. Inputs: the `fm_score_current` bundle. Rows (same order and labels as
the per-window section): ETKF, EnKF (Gaussian only, stored variances), ETKS, Strong-4DVar (point,
regular only), DirectUNet-M (point), PredictStateCFM-M, VanillaCFM-M, SDA1-M, SDA2-M, SDA3-fix-M
(1200 ep), hybrid (reference). Columns: regular S0 | regular S1 | random S0 | random S1. Cells:
`value [lo, hi]`; best per column **bold**, second *italic* (as in the per-window section).

**Prerequisite (separate, small):** the per-window section still lists the 400-epoch SDA rows;
#307 has the 1200-epoch priors. Refresh it in the same PR so both sections show the same rows.

**Text and tables (exact content):**

1. Definition paragraph (≤ 8 lines): path x_τ = τ x₁ + (1−τ) x₀ in z-scored space; FMS_τ(q) =
   E‖E_q[x₁|x_τ] − x₁*‖², reported in physical units² (σ_c²-weighted); τ = 0 is the MSE of the
   ensemble mean; each τ ∈ (0,1) strictly proper; for a Gaussian forecast minimised at
   spread² = MSE at every τ, so τ > 0 rewards calibration; point estimates score their MSE at every
   τ; τ = 0.95 omitted (dominated by the universal 1/snr floor). Variant: Gaussian (per-element
   N(mean, var)) for every row, so DA, flows and SDA are scored alike; empirical/KDE variants in
   the JSON. Pooled over windows × time × 24 channels; seeds averaged per window; 95% window
   bootstrap.
2. **Table B1–B4: FMS at τ = 0, 0.25, 0.5, 0.75** (one table per τ, rows × 4 columns).
3. **Table B5: calibration** — pooled spread/skill (target √(N/(N+1)) = 0.984 for N = 30) and
   calibration loss at τ = 0.5 / 0.75 with c*, regular and random, S0 and S1.
4. **Table B6: paired differences vs PredictStateCFM-M** at τ = 0 and τ = 0.75: Δ [lo, hi] and wins
   / 200, for every row — this is where the hybrid's crossover is tested (better at τ = 0, worse or
   tied at τ = 0.75).
5. **Table B7: non-Gaussian check** — KDE vs Gaussian at the KDE's variance, τ = 0.5 / 0.75 (ensemble
   rows only; DA from the member re-run).
6. **Reading** (5 bullets, provisional values in z-units from `docs/results/l96_p1_fm_score.md`):
   - Accuracy and distributional skill rank differently: the hybrid has the best τ = 0 (MSE) on
     every column but is overtaken by the flows at τ = 0.75 (regular S0 0.0200 vs 0.0195; random
     0.029 vs 0.025) — under-dispersed sampler (calibration loss 15–28% vs 0–3%).
   - Misspecification signature: DA calibration loss grows S0 → S1 (ETKF 14 → 23% regular,
     3 → 14% random; ETKS 34 → 63%, 21 → 47%); learned rows invariant; SDA2 falls (7 → 3%).
   - ETKS has the best DA mean but the worst DA score from τ = 0.75 (S0) / 0.5 (S1).
   - Flows gain 1.4–3.4% from non-Gaussian shape (KDE); DA ensembles are short-tailed
     (+18–31% at S1, τ = 0.75).
   - Training budget: PSC-M 400 → 1200 ep cuts its τ = 0.75 calibration loss 11% → 2%.
7. **Caveats** (3 bullets): DA ensemble rows are a fresh realisation (RMSE within sampling noise of
   the benchmark files); EnKF and Strong-4DVar have no stored ensemble (Gaussian / point only);
   the window bootstrap does not include seed variability (seed ranges shown separately).

## C. Paper inserts (`docs/papers/p1_structural_hypotheses/`)

Placement follows the current LaTeX; the mapping to the revised structure
(`docs/plans/paper/l96_p1_structure_revision.md` on `feature/p1-structure-revision`) is given per
item. Values in brackets are filled from A.

### C1. §3 — new subsection "The flow operator as a scoring rule" (revised structure: §3, after Ψ_mean + Ψ_anom)

> **Draft.** The operator that defines a conditional flow also scores one. For a forecast
> distribution $q$ of the state given the context, let $\Psi_q(\mathbf{x}_\tau,\tau) =
> \mathbb{E}_q[\mathbf{x}_1 \mid \mathbf{x}_\tau]$ on the path $\mathbf{x}_\tau = \tau\mathbf{x}_1 +
> (1-\tau)\mathbf{x}_0$, $\mathbf{x}_0\sim\mathcal{N}(0,I)$. Given the truth $\mathbf{x}_1^\star$,
> \begin{equation} \mathrm{FMS}_\tau(q) = \mathbb{E}\,\|\Psi_q(\mathbf{x}_\tau,\tau) -
> \mathbf{x}_1^\star\|^2 = \mathbb{E}\,\|\Psi_q - \Psi_p\|^2 + \mathrm{mmse}_p(\tau),
> \label{eq:fms}\end{equation}
> where $p$ is the posterior given all the information available to any scheme, so the constant is
> common to all schemes and differences in $\mathrm{FMS}_\tau$ are exact operator-divergence
> differences. Three properties make it the instrument this paper needs. (i) At $\tau = 0$,
> $\Psi_q$ is the forecast mean and $\mathrm{FMS}_0$ is its MSE: the score of $\Psi_{\mathrm{mean}}$.
> (ii) For $\tau\in(0,1)$ it is strictly proper, and for a Gaussian forecast $\mathcal{N}(m, v)$ with
> error $e$ and $s = \tau^2/(1-\tau)^2$,
> $\mathrm{FMS}_\tau = (e^2 + s v^2)/(1 + s v)^2$, minimised at $v = \mathbb{E}e^2$ for every
> $\tau$: the restriction identity of Section~X is the optimality condition of the score, so
> $\tau > 0$ scores $\Psi_{\mathrm{anom}}$. (iii) A point estimate is a degenerate flow,
> $\Psi_q \equiv m$, and scores its MSE at every $\tau$; deterministic and probabilistic schemes
> share one axis. Integrated against $\mathrm{d}s$ the score recovers the logarithmic score
> \citep{verdu2010mismatched}; as $\tau\to1$ its leading informative term is the Hyvärinen score.
> We report $\tau\in\{0, 0.25, 0.5, 0.75\}$ and decompose $\mathrm{FMS}_\tau$ into a calibration
> loss --- the share removed by the best scalar rescaling of the forecast variance --- and the rest.

Plus **Appendix A′** (½–1 page): propriety; Gaussian closed form and optimum; mixture/ensemble
operator (softmax over members) and its KDE version; the 1/snr floor and Hyvärinen limit; the
log-score integral; why a block-local evaluation is needed (weight degeneracy: effective sample
size 28 / 19 / 5 of 30 at τ = 0.25 / 0.5 / 0.75 on 24-channel snapshots); the affine-equivariance
argument for physical units. Bib: Verdú (2010), Hyvärinen (2005), Vincent (2011), Gneiting &
Raftery (2007), Scheuerer & Hamill (2015) for context.

### C2. §4 "Metrics" (`04_evaluation.tex`; revised structure §4.5) — replace the paragraph

> **Draft.** RMSE is computed [per window per channel and averaged | pooled as
> $\sqrt{\overline{\mathrm{MSE}}}$ — pending decision] on the 24 observed channels of the 200 test
> windows, with the same convention for every scheme. Ensemble schemes are scored with the CRPS
> and with $\mathrm{FMS}_\tau$ (Section~\ref{sec:fms}), both proper, both in physical units;
> spread/skill is the pooled ratio $\sqrt{\mathbb{E}[\mathrm{var}]/\mathbb{E}[e^2]}$ over windows,
> time and channels, whose calibrated value is $\sqrt{N/(N+1)} = 0.984$ for $N = 30$ (a per-window
> ratio is not that quantity). Every reported value carries a 95\% bootstrap interval over test
> windows (2000 replicates; seeds averaged per window); comparisons between schemes use the paired
> bootstrap of the per-window difference.

(Removes the old "spread/skill vs √((N−1)/(N+1))" and "energy scores" sentences; rank histograms move
to the appendix sentence.)

### C3. §5.5 "Dispersion" (`05_results.tex` §`sec:nongaussian`; revised §5.5 + §6.3) — rewrite

- **Replace Table `tab:calibration`** (per-window spread/RMSE, pre-#291 ETKF 0.687, "filters within 3%
  of the target" — all three superseded) with **Table: FMS_τ at S0, regular grid** — rows ETKF, ETKS,
  DirectUNet-M, VanillaCFM-M, PredictStateCFM-M, SDA1/2/3-fix-M, hybrid; columns FMS at τ = 0, 0.5,
  0.75 [CI], pooled spread/skill, calibration loss (τ = 0.75).
- **New text (draft):** "Scored by FMS, the classes separate on calibration, not on the Gaussian /
  non-Gaussian divide the operator partition suggested. The flows are calibrated (calibration loss
  [0–2]%); the filter is mildly over-confident ([14]%); the smoother and the guided samplers are
  over-confident ([34]% and [7–18]%). Accuracy and distributional skill rank differently: the
  hybrid has the lowest $\mathrm{FMS}_0$ of every scheme but is overtaken by the flows at
  $\tau = 0.75$ ([0.0200] vs [0.0195]; random layouts [0.029] vs [0.025]), the paired difference
  [Δ, CI, wins]." Keep the rank-histogram paragraph, add: "The kernel-smoothed flows beat a Gaussian
  of the same variance by [1.4–3.4]%, so the flows' non-Gaussian structure is real but small; the
  DA ensembles are short-tailed."
- **C4 sentence** stays "not supported as a discriminating claim" — now with FMS as the evidence
  rather than spread/RMSE.

### C4. Model-error results (`05_results.tex` §5.2/§5.3; revised §6.3 "Calibration under model error")

New **Table: calibration loss at τ = 0.75, S0 → S1**, regular and random, all ensemble rows [CI]; text:
"Model error turns the filters' and smoother's residual calibration loss into a dominant one (ETKF
[14 → 23]%, ETKS [34 → 63]% on the regular grid; [3 → 14]%, [21 → 47]% on the random layouts),
while every learned sampler is unchanged — except SDA2, whose loss falls ([7 → 3]%) because
conditioning on the biased parameters widens its prior. This is the misspecification signature of
Section~X measured with a proper score: the state-space inflation tuned at S0 cannot absorb a
parameter error at S1." Supports K6 (and K4/K5 framing) of the structure doc.

### C5. "Blending" paragraph (`05_results.tex`; revised §6.5)

Replace "most accurate scheme measured (0.312 …) at spread/RMSE 0.53" by: "the most accurate scheme
(FMS₀ [ ]; RMSE [0.293]) — but not the best distribution: its tempered-likelihood sampler is
over-confident (calibration loss [16]%) and the flows overtake it at τ = 0.75. Blending composes
Ψ_mean from one scheme and Ψ_anom from another; FMS scores the two separately and shows the
composition inherits the second's calibration." (Hybrid numbers move to the 1200-epoch SDA prior,
#307.)

### C6. Abstract / claims

- Abstract: one sentence — "A proper score derived from the flow operator separates the accuracy of
  the mean from the calibration of the anomaly, and shows that the most accurate scheme is not the
  best-calibrated one, and that model error degrades the filters' calibration while leaving the
  learned samplers' unchanged."
- Claims list: K6 restated on FMS; new **K9** "accuracy and distributional skill rank differently;
  composition inherits the anomaly's calibration" (hybrid crossover).

### C7. Not touched

The C4/C5 history, QG sections, L63 — unchanged by this plan.

## Order of work

1. Tonight: A1 → A2 (submit) → A3 checks in the morning.
2. B as one PR (report module + tables + per-window SDA-1200 refresh), CI green + reviewer approval,
   user merges.
3. C as one PR on the LaTeX, after the user has picked the RMSE convention and reviewed B.
