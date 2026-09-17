## 2026-09-16: loc_radius/etkf_ridge re-tune at the reference density; full 1-64 cols/day density curve

**Summary:** Re-checks `loc_radius=6.0` -- the project's own reference-case
default, never itself swept -- at the reference density (cols=4) and finds
it was untuned there too (worst point in a {1,2,3,4,6} N=10 grid, both
methods). Re-tunes `etkf_ridge` at the new `loc_radius=2.0` (its benefit
had inverted from the 2026-09-12 tuning done at `loc_radius=6.0`) and
`inflation` (unaffected, remains optimal). Promotes `loc_radius=2.0`
(cols<=32) / `loc_radius=1.0` (cols=64) + `etkf_ridge=0.1` (was 1.0) into
the canonical `qg_repro_validation{,_s1}/{etkf,enkf}.json`, and extends
the obs-density curve to the full cols=1/2/4/8/16/32/64 range (previously
only 1,2,4,16,64 had been tested), re-confirming every existing point at
the new `etkf_ridge=0.1` for full internal consistency.

**Two striking results**: (1) a perfectly monotonic dose-response curve
across the whole 1-64 cols/day range, once `loc_radius` is tuned per
density instead of held at one project-wide default; (2) ETKF and EnKF
are now nearly indistinguishable at every density (typically within
0.001-0.006 EV) -- both the original EnKF>ETKF gap (2026-09-11) and the
later ETKF>EnKF gap (after `etkf_ridge=1.0`'s 2026-09-12 promotion) were
largely artifacts of the untuned `loc_radius=6.0`, not a real method
difference.

**Files modified:**
- `reports/qg/outputs/qg_repro_validation{,_s1}/{etkf,enkf}.json` --
  promoted to the new config; old `loc_radius=6.0` results archived as
  `{etkf,enkf}_loc6_default.json` in both directories, not deleted.
- `reports/qg/generate_qg_da_report.py`, `reports/qg/generate_qg_neural_report.py`
  -- updated hardcoded config/comparison strings to the new default;
  regenerated `qg_da_report.md`/`qg_neural_report.md`.
- `reports/qg/generate_da_sensitivity_report.py` -- new "Reference density
  (cols=4) re-check, ridge re-tune, and final promotion" subsection;
  updated the previously-open "Not yet decided: promote a high-density
  config?" callout (resolved differently than originally framed -- the
  reference density itself was re-tuned instead); regenerated
  `da_sensitivity_s0_s1_report.md`.
- `reports/qg/generate_qg_obs_density_report.py` (new) ->
  `reports/qg/outputs/qg_obs_density_report.md` (new) -- dedicated report
  covering the complete cols=1-64 curve for both methods at N=100.
- `reports/qg/outputs/qg_obs_density_sweep/*.json` (174 new N=10 files:
  cols=4/8/32 `loc_radius` screens, ridge re-checks at both `loc_radius`
  values, inflation re-check) + `reports/qg/outputs/qg_obs_density_sweep_n100/*.json`
  (new N=100 confirmations, including the cols=1/2/16/64 ETKF re-runs at
  `ridge=0.1`).
- `PLAN.md` -- full narrative, including two mid-session incidents caught
  and fixed: a squash-merge branch-divergence gotcha (rebased cleanly
  onto `origin/master`, resolving one real conflict against an unrelated
  concurrently-merged PR) and a `sed`-rename bug that briefly had two
  SLURM jobs writing to colliding stale filenames (caught from job logs,
  cancelled before damage, verified no committed data was touched).

**Rationale:** The obs-density sensitivity study's N=100 confirmation
found EnKF edging out ETKF+ridge=1.0 at high density -- a wrinkle that
turned out to be explained by `loc_radius=6.0` being untuned at the
reference density too, not just at high density. Rather than treating
"promote a high-density config" as a benchmark-design call (changing the
observation configuration), the actual fix was simpler and more general:
re-tune the reference config itself.

**Verification:** `ruff check` passes on all touched Python files; all
three regenerated reports reproduce byte-identical output from their
generators (no manual drift). Every `loc_radius`/`etkf_ridge`/`inflation`
value was independently screened at N=10 and confirmed at N=100 for both
methods separately, not assumed or transferred without checking. CI-gated
test list unaffected (docs/report + JSON-data-only change, no QG/neural
production code touched).
