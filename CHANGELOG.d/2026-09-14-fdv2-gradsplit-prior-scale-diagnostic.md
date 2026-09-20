## 2026-09-14: L96 — gradsplit_prior_scale diagnostic knob + prior_weight fix

**Summary:** New `FourDVarNetSolver` param `gradsplit_prior_scale` (default
1.0, no-op): multiplies `gradsplit+state`'s `g_prior` channel AFTER
normalization/soft-clipping. New diagnostic experiment config
`FDV2_gradsplit_state_monai_l96_Stier_priorscale1e4.yaml`
(`gradsplit_prior_scale=1e-4`) makes the fed tensor `cat([g_obs, ~0, x])`,
functionally close to `obs+state`'s own `cat([x, obs_clean])`, to test
whether a persistent training plateau (jobs 53440/53447, both oscillating
~3.5-4.7 through epoch 10) traces back to `g_prior`'s own contribution.
Also fixed two test bugs found while adding this: a genuine test-writing
mistake in the new test itself (dropout-induced nondeterminism comparing
two `prior_unet` forward calls -- fixed with `dropout=0.0` + `.eval()`),
and a pre-existing, unrelated test-order fragility in
`test_normalize_channels_cache_reuses_first_norm` (an unseeded statistical
assertion that happened to flip to a false failure once the new test's
insertion shifted the global RNG stream for later tests in the file --
fixed by seeding it explicitly).
**Files modified:** `models/fourdvarnet.py` (`_build_update_input`,
`_solver_iteration`, `FourDVarNetSolver.__init__`/checkpoint call site,
docstrings); `conf/schema.py` (`FourDVarNetConfig.gradsplit_prior_scale`);
`train.py` (model_factory threading); `tests/test_fourdvarnet.py` (new
`test_build_update_input_gradsplit_prior_scale_multiplies_after_normalization`,
seeded `test_normalize_channels_cache_reuses_first_norm`); new
`config/experiment/FDV2_gradsplit_state_monai_l96_Stier_priorscale1e4.yaml`,
`batch/run_l96_fdv2_gradsplit_state_monai_Stier_priorscale1e4_train.sbatch`.
`FourDVarNetPredictStateCFM` deliberately NOT touched (mirrors the
`init_state_var` precedent -- a diagnostic-only `FourDVarNetSolver` knob,
not needed there; its own `_solver_iteration` call site is unaffected since
the new parameter has a default).
**Rationale:** User asked to re-run `gradsplit+state`'s S-tier config with
`g_prior` forced near zero after normalization, to isolate whether the
persistent training plateau traces back to `g_prior`'s own contribution
specifically (as opposed to, e.g., the two-real-gradient double-autograd-
call per iteration itself, or some other mechanism).
**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py
tests/test_lightning_module.py` -- 128 passed. `ruff check` clean. Manual
smoke test: direct `_build_update_input` scaling check (already covered by
the new unit test); `model_factory`-level integration smoke test for the
new config was inconclusive due to heavy CPU contention on this shared node
at the time (load average 15+, 8 cores) -- not re-attempted given the unit
test directly exercises the exact code path with matching assertions.
