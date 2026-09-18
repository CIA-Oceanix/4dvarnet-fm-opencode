## 2026-09-18: Refresh the Stier values the scoping docs cite, after #219

**Summary:** #219's regeneration of the L96 consolidated benchmark shifted three
Stier rows by 1 in the last decimal. The scoping docs quote those values by
digit, so they were superseded on arrival. Refreshed, and the coupling recorded.

**Files modified:** `docs/scoping/da_paper_structural_hypotheses.md`,
`docs/scoping/ml_paper_exploiting_prior_knowledge.md`,
`docs/scoping/phase_D_physics_priors_under_uncertainty.md` (3 lines each);
`docs/scoping/README.md` — new standing caveat.

| row | was cited | now |
|---|---|---|
| `FDV1-Stier(monai)` | 0.4011 / 0.3993 | 0.4012 / 0.3994 |
| `subgrad+state-Stier(monai)` | 0.3728 / 0.3729 | 0.3729 / 0.3730 |
| `subgrad+state-Stier+SDA3(monai)` | 0.3410 / 0.3402 | 0.3411 / 0.3403 |

**Rationale:** #219 rebuilt the report to recompute every row from stored
trajectory arrays under a uniform pooled convention, which moved these three in
the fourth decimal. The DA baselines (0.8116 / 0.8883 / 0.9131), the SDA monai
trio and `DirectUNet-M(monai,cos)` are unchanged. **No conclusion moves:** the
ML doc's headline split recomputes to 93.24% / 6.76% on either set of values, so
the "~93% / ~7%" statement stands as written.

`phase_D_physics_priors_under_uncertainty.md` was included because it cites the
same rows; its framing is superseded but its work packages are live, so leaving
one of the four docs stale would have been worse than the edit.

**Standing caveat added** to `docs/scoping/README.md`: nothing guards the docs
against the report. `tests/test_l96_report_consistency.py` guards the report
against the archive, but the docs→report direction is unprotected, so any
regeneration needs a re-grep of the cited values.

**Known and deliberately not fixed:** `reports/l96/generate_l96_fm_sampler_report.py`
hardcodes `BENCHMARK_ROWS` copied from the consolidated benchmark, including
`subgrad+state-Stier+SDA3(monai) = 0.3410`, now likewise stale by one digit.
Updating it would desynchronize the generator from its checked-in output, which
**cannot be regenerated** — the source JSONs are gitignored SLURM artifacts no
longer on disk (recorded in the #214 fragment). Left for whoever next has those
inputs.

**Verification:** every cited value re-extracted from `origin/master`'s report
and compared; `grep -rnE "0\.4011|0\.3993|0\.3728|0\.3410|0\.3402" docs/scoping/`
returns nothing. Docs only, no code touched.
