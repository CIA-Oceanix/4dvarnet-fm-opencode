## 2026-09-14: N=100 confirmation of the loc_radius-tuned obs-density configs

**Summary:** Runs the two loc_radius-tuned high-density ETKF/EnKF obs
configs found by the earlier N=10 correction (cols=16, loc_radius=2.0;
cols=64, loc_radius=1.0) at full N=100, for both ETKF and EnKF, on both
S0 and S1. EnKF's own `loc_radius` optimum was checked separately at N=10
first rather than assumed to match ETKF's (the untuned loc=6.0 EnKF
cross-check had collapsed *more* severely than ETKF at the same density)
-- it matched ETKF's exactly at both densities.

**Result: fully confirmed.** Both tuned configs decisively beat the
canonical cols=4/loc=6.0 baseline, for both methods, on every field and
both scenarios, including the previously-collapsing unobserved deep
layer (q layer2, which roughly doubles at cols=64). New at N=100 (not
visible at N=10): EnKF edges out ETKF+ridge=1.0 at these higher
densities (modest ~0.01-0.02 margin) -- "ETKF+ridge=1.0 beats EnKF" is a
cols=4-specific finding, not universal.

**Files modified:**
- `reports/qg/outputs/qg_obs_density_sweep_n100/*.json` (new, N=100, 8
  files: ETKF/EnKF x cols={16,64} x S0/S1).
- `reports/qg/generate_da_sensitivity_report.py` -- new "N=100
  confirmation" subsection under the obs-density section; Synthesis
  bullet updated.
- `reports/qg/outputs/da_sensitivity_s0_s1_report.md` -- regenerated.
- `PLAN.md` -- N=100 confirmation subsection added under the "MAJOR
  CORRECTION" section.

**Rationale:** The N=10 correction's headline claim ("more density,
properly localized, is unambiguously better") was screening-scale;
closing the "N=100 confirmation is the natural next step" caveat it left
open.

**Not yet decided:** whether to promote one of these configs (most
plausibly cols=64/loc=1.0) into the canonical S0/S1 benchmark -- unlike
`etkf_ridge=1.0`'s promotion, this changes the observation configuration
itself, a benchmark-design call left open pending discussion.

**Verification:** `ruff check` passes; report regenerated and manually
reviewed against the underlying JSON (all tables computed directly, not
transcribed). Both SLURM jobs (53537 ETKF 1h50m, 53538 EnKF 2h36m)
completed successfully (`sacct` state COMPLETED).
