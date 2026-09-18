## 2026-09-18: L63 — per-window observation noise; delete an orphaned module

**Summary:** `Lorenz63Dataset` gave every window the same observation-noise
realization. Fixed to draw per window, matching the two sibling datasets. Also
deletes `evaluation/experiment.py`, which nothing imported and which would have
crashed if called.

**Files modified:**
- `data/lorenz63.py` — per-window `obs_seed`; the seed is recorded in each window dict
- `tests/test_lorenz63.py` — `test_observations_noise` pools across windows
- `tests/test_data_leakage.py` — inject `dynamics=` into two slow 4DVar tests
- `evaluation/baselines.py` — `_ESAccumulator.step` accepts a numpy ensemble
- `.github/workflows/ci.yml` — drop the now-obsolete `test_observations_noise` deselect
- `evaluation/experiment.py` — **deleted**
- `docs/scoping/refactor_plan.md` — record both decisions

**Rationale:** `obs_seed = cfg.seed + 1` was a constant, and
`generate_observations` was then called once per window with it, so all windows
received an identical noise draw. `data/lorenz96.py:437` and
`data/random_param_dataset.py:35` already used `cfg.seed + i * 100 + 1`; the
forcing seed three lines below the bug also varied per window, which is what
made the asymmetry an oversight rather than a design choice.

Measured, pooled per-dimension variance against a target `R_var = 0.5`:

| | before | after |
|---|---|---|
| 5 windows (n=125) | 0.319 / 0.506 / 0.331 | — |
| 40 windows (n=1000) | 0.316 / 0.502 / 0.329 | **0.524 / 0.492 / 0.495** |

Before, pooling 8x more windows did not move the numbers, because the samples
were not independent — the noise vector in window 5 matched window 0 to float32
rounding.

`test_observations_noise` detected this and had been failing for as long as it
had been ungated, but it was *also* under-powered: it checked a single window's
~25 observations against a +/-30% tolerance, when the sample-variance relative
standard error at n=25 is ~29%. It now pools across windows — which is only a
valid thing to do *because* of this fix — with a tolerance derived from the
pooled sample size.

`evaluation/experiment.py` (`run_baselines`/`run_full_experiment`) was
unreferenced anywhere in the repo, untouched since the initial commit apart from
one ruff auto-fix, recorded at 0% coverage in `tests/TEST_SUITE_SUMMARY.md`, and
built `Weak4DVar`/`Strong4DVar`/`EnKF` without `dynamics=` — so it would have
raised `AttributeError` on its first `assimilate()` call.

**Impact:** results built on `Lorenz63Dataset` — the L63 S0/S1 DA baselines and
the E/F/G/S series — need re-running before their published numbers are valid
again. Window 0 is bit-identical (its seed is unchanged); windows 1+ move. L96
and QG are unaffected: they already drew per-window seeds.

**Product bug found by the newly-runnable slow tests:**
`_ESAccumulator.step` guarded `ref_t` with an `isinstance(..., torch.Tensor)`
check but called `.detach()` on `ensemble_t` unconditionally. The ensemble
filters pass tensors, but `Strong4DVar`/`Weak4DVar` pass a slice of an
already-converted numpy analysis, so either deterministic method raised
`AttributeError: 'numpy.ndarray' object has no attribute 'detach'` whenever ES
accumulation was enabled. Both branches are now symmetric. This is the same
error the retired `tests/test_equiv_report.py` script hit, and it survived
because the only tests exercising that path are `@pytest.mark.slow` and so have
never run in CI.

**Verification:** `pytest tests/test_lorenz63.py tests/test_random_param_dataset.py
tests/test_refactoring_equivalence.py tests/test_data_leakage.py` → all pass,
including the two slow 4DVar tests that had never run in CI.
